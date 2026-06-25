"""
QR Code Replacer Pipeline v0.3.0
===============================
Deterministic pipeline for detecting and replacing QR codes in images
with realistic local appearance matching.

Key improvements over v0.2:
- Carrier patch replacement (not just tight QR bounding box)
- Local brightness/contrast/color matching from surrounding area
- Automatic blur estimation to match local sharpness
- JPEG-like texture simulation
- Seamless blending (OpenCV seamlessClone) as alternative
- Multiple decode retries with fallback strategies
"""

import cv2
import numpy as np
import qrcode
import os
import json
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QRReplacementResult:
    output_image_path: str
    detected_points: Optional[np.ndarray]
    old_decoded_text: Optional[str]
    new_decoded_text: Optional[str]
    success: bool
    method_used: Optional[str] = None
    selected_preprocess_variant: Optional[str] = None
    expanded_points: Optional[np.ndarray] = None
    selected_blend_mode: str = "feather"
    retries_used: int = 0
    reason_if_failed: Optional[str] = None
    debug_report: dict = field(default_factory=dict)


# ─── Detection ────────────────────────────────────────────────────────────────

def _detect_raw(img: np.ndarray) -> tuple:
    """Low-level detect + decode using separate calls. Returns (points, decoded_text)."""
    detector = cv2.QRCodeDetector()
    try:
        retval, points = detector.detect(img)
        if retval and points is not None:
            text, _ = detector.decode(img, points)
            return points, text
    except Exception:
        pass
    try:
        retval, decoded_list, points, _ = detector.detectAndDecodeMulti(img)
        if retval and decoded_list:
            text = decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
            return points[0] if points is not None else None, text
    except Exception:
        pass
    return None, None


def _detect_multi(img: np.ndarray) -> tuple:
    """Try detectAndDecodeMulti for multiple QR codes."""
    detector = cv2.QRCodeDetector()
    try:
        retval, decoded_list, points, _ = detector.detectAndDecodeMulti(img)
        if retval and decoded_list:
            return points, decoded_list
    except Exception:
        pass
    return None, None


def _preprocess_variants(img: np.ndarray):
    """Generate multiple preprocessing variants for detection."""
    variants = []

    # BGR original
    variants.append(("bgr", img))

    # Grayscale variants
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    # Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
    )
    variants.append(("adaptive", adaptive))

    # Inverted grayscale
    inv_gray = cv2.bitwise_not(gray)
    variants.append(("inv_gray", inv_gray))

    # CLAHE grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cl))

    return variants


def detect_qr_chain(img: np.ndarray, debug_dir: Optional[str] = None, mode: str = "single") -> tuple:
    """
    Detection pipeline with multiple preprocessing variants.
    Returns (points, decoded_text, method_used).
    """
    detector = cv2.QRCodeDetector()
    all_results = []

    for name, variant in _preprocess_variants(img):
        try:
            if variant is img or len(variant.shape) == 2:
                retval, points = detector.detect(variant)
            else:
                retval, points = detector.detect(cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if len(variant.shape) == 3 else variant)

            if retval and points is not None:
                # Try to decode
                decoded = None
                try:
                    if len(variant.shape) == 2:
                        text, _ = detector.decode(variant, points)
                    else:
                        text, _ = detector.decode(img, points)
                    decoded = text
                except Exception:
                    pass

                all_results.append({
                    "name": name,
                    "points": points,
                    "decoded": decoded,
                    "variant": variant,
                })
        except Exception:
            pass

    if not all_results:
        return None, None, None

    # For single mode: pick best result (prefer decoded > non-decoded, then largest area)
    if mode == "single" or len(all_results) == 1:
        # Score: decoded gets priority, then largest quadrilateral
        def score(r):
            area = cv2.contourArea(r["points"].astype(np.float32).reshape(4, 2))
            return (1 if r["decoded"] else 0, area)

        all_results.sort(key=score, reverse=True)
        best = all_results[0]
        return best["points"], best["decoded"], best["name"]

    # Multi mode
    multi_points, multi_decoded = [], []
    for r in all_results:
        if r["decoded"]:
            multi_points.append(r["points"])
            multi_decoded.append(r["decoded"])

    if multi_points:
        return multi_points, multi_decoded, "multi"

    return None, None, None


# ─── Point ordering ───────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],      # top-left (min sum = closest to origin)
        pts[np.argmin(diff)],   # top-right (min y difference)
        pts[np.argmax(s)],      # bottom-right (max sum)
        pts[np.argmin(-diff)],  # bottom-left (max y difference)
    ], dtype=np.float32)


# ─── Carrier patch expansion ───────────────────────────────────────────────────

def expand_carrier_quad(pts: np.ndarray, ratio: float, img_h: int, img_w: int) -> np.ndarray:
    """
    Expand the detected QR quadrilateral outward from its center by ratio.
    Returns the expanded carrier quad clamped to image bounds.
    """
    center = pts.mean(axis=0)

    # Vector from center to each corner
    dirs = pts - center

    # Scale outward
    expanded = center + dirs * ratio

    # Clamp to image bounds with 5px margin
    margin = 5
    expanded[:, 0] = np.clip(expanded[:, 0], margin, img_w - margin)
    expanded[:, 1] = np.clip(expanded[:, 1], margin, img_h - margin)

    return expanded.astype(np.float32)


def estimate_local_appearance(img: np.ndarray, quad: np.ndarray, ring_width: int = 15) -> dict:
    """
    Sample the surrounding ring region around a quad to estimate local appearance.
    Returns dict with mean_bgr, contrast, color_cast, sharpness_estimate.
    """
    # Create mask for the quad
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)

    # Dilate to get ring
    kernel = np.ones((ring_width, ring_width), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    ring_mask = cv2.subtract(dilated, mask)

    # Sample ring pixels
    ring_pixels = img[ring_mask > 0]

    if len(ring_pixels) == 0:
        return {"mean_bgr": np.array([220, 220, 220]),
                "contrast": 20.0, "color_cast": np.array([0, 0, 0]),
                "sharpness": 1.0}

    mean_bgr = ring_pixels.mean(axis=0).astype(np.float32)

    # Contrast: std dev of grayscale ring
    gray_ring = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[ring_mask > 0]
    contrast = gray_ring.std()

    # Color cast: deviation of mean from luminance-only
    luminance = mean_bgr.mean()
    color_cast = mean_bgr - luminance

    # Rough sharpness estimate: Laplacian variance on ring
    try:
        ring_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(ring_gray, cv2.CV_64F, mask=ring_mask).var()
        sharpness = min(lap_var / 100.0, 3.0)  # Normalize
    except Exception:
        sharpness = 1.0

    return {
        "mean_bgr": mean_bgr,
        "contrast": float(contrast),
        "color_cast": color_cast,
        "sharpness": float(sharpness),
    }


# ─── QR Generation ─────────────────────────────────────────────────────────────

def _qr_version_capacity(version: int) -> int:
    """Approximate max payload length for a QR version at M error correction."""
    capacities = {1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134, 7: 154, 8: 192, 9: 230,
                  10: 271, 11: 321, 12: 367, 13: 425, 14: 458, 15: 520, 16: 586,
                  17: 644, 18: 718, 19: 792, 20: 858}
    return capacities.get(version, 1000)


def generate_qr_image(
    payload: str,
    quiet_zone: int = 4,
    fg_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Generate a clean QR code image. warpPerspective handles all scaling.
    """
    payload_len = len(payload)
    version = 1
    for v in range(1, 40):
        capacity = _qr_version_capacity(v)
        if payload_len <= capacity:
            version = v
            break

    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=quiet_zone,
    )
    qr.add_data(payload)
    qr.make()

    img = qr.make_image(fill_color=fg_color, back_color=bg_color)
    return np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR


def generate_carrier_patch(
    payload: str,
    carrier_size: tuple,
    fg_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    local_appearance: Optional[dict] = None,
    add_texture: bool = True,
    jpeg_simulation: bool = True,
) -> tuple:
    """
    Generate a carrier patch: paper-colored background with full-size QR centered.
    The QR is NOT resized — warpPerspective handles all scaling to the destination.
    Returns (patch, qr_only_mask, carrier_mask).
    """
    patch_w, patch_h = carrier_size

    # Base carrier: sampled paper color (NOT lighter/whiter than paper)
    # The carrier should look like the paper it's printed on, not a bright sticker
    if local_appearance is not None:
        mean_bgr = local_appearance["mean_bgr"]  # BGR
        # Match paper color exactly - no brightening
        base_color = mean_bgr.astype(np.uint8)
        patch = np.full((patch_h, patch_w, 3), base_color[::-1], dtype=np.uint8)  # BGR
    else:
        patch = np.full((patch_h, patch_w, 3), 255, dtype=np.uint8)

    # Generate QR at FULL size (warpPerspective will handle scaling)
    qr_img = generate_qr_image(payload, quiet_zone=4, fg_color=fg_color, bg_color=bg_color)
    qr_h, qr_w = qr_img.shape[:2]

    # Center QR in carrier (no resize!)
    x_offset = max(0, (patch_w - qr_w) // 2)
    y_offset = max(0, (patch_h - qr_h) // 2)

    # Place QR in patch
    y_end = min(y_offset + qr_h, patch_h)
    x_end = min(x_offset + qr_w, patch_w)
    patch[y_offset:y_end, x_offset:x_end] = qr_img[:y_end-y_offset, :x_end-x_offset]

    # Create masks
    qr_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, qr_only_mask = cv2.threshold(qr_gray, 128, 255, cv2.THRESH_BINARY_INV)
    carrier_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.rectangle(carrier_mask, (0, 0), (patch_w - 1, patch_h - 1), 255, -1)

    # Add subtle texture for realism to background only (not QR modules)
    if add_texture:
        bg_mask = (qr_gray > 200).astype(np.float32)
        noise = np.random.normal(0, 3.0, patch.shape).astype(np.float32)
        textured = np.clip(patch.astype(np.float32) + noise * bg_mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
        textured = cv2.GaussianBlur(textured, (3, 3), 0.4)
        # Only apply texture to background, not QR modules
        patch = np.where(bg_mask[:, :, np.newaxis] > 0, textured, patch)

    # JPEG-like compression simulation (light)
    if jpeg_simulation:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, enc = cv2.imencode(".jpg", patch, encode_param)
        patch = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return patch, qr_only_mask, carrier_mask


# ─── Perspective warp ─────────────────────────────────────────────────────────

def perspective_warp(
    src_img: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple,
) -> np.ndarray:
    """
    Warp src_img to fit the destination quadrilateral dst_points.
    dst_points must be ordered: TL, TR, BR, BL.
    """
    h, w = src_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)
    dst_pts = np.array([
        dst_points[0], dst_points[1], dst_points[2], dst_points[3]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return cv2.warpPerspective(src_img, H, (output_size[1], output_size[0]))


def perspective_warp_with_mask(
    src_img: np.ndarray,
    src_mask: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple,
) -> tuple:
    """Warp image and mask together, returning (warped_img, warped_mask)."""
    h, w = src_img.shape[:2]
    src_pts = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=np.float32)
    dst_pts = np.array([
        dst_points[0], dst_points[1], dst_points[2], dst_points[3]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(src_img, H, (output_size[1], output_size[0]))
    warped_mask = cv2.warpPerspective(src_mask, H, (output_size[1], output_size[0]))
    return warped_img, warped_mask


# ─── Appearance matching ───────────────────────────────────────────────────────

def match_appearance_to_local(
    warped_patch: np.ndarray,
    warped_mask: np.ndarray,
    original_img: np.ndarray,
    quad: np.ndarray,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Apply local appearance matching to the warped patch.
    - Blur slightly to match local sharpness
    - Slight luminance matching
    """
    result = warped_patch.copy().astype(np.float32)
    original = original_img.copy().astype(np.float32)

    # Create a soft mask for the patch region
    patch_mask = (warped_mask > 0).astype(np.float32)

    # Estimate local stats from just outside the quad
    outer_ring = cv2.subtract(
        cv2.dilate(warped_mask, np.ones((7, 7), np.uint8), iterations=2),
        warped_mask
    )
    outer_pixels = original_img[outer_ring > 0].astype(np.float32)
    if len(outer_pixels) > 0:
        local_mean = outer_pixels.mean(axis=0)
        patch_mean = result[patch_mask > 0].mean(axis=0)
        adjust = local_mean - patch_mean
        # Apply brightness/color adjustment inside mask
        for c in range(3):
            result[:, :, c] = np.where(
                patch_mask > 0,
                np.clip(result[:, :, c] + adjust[c] * 0.6, 0, 255),
                original[:, :, c]
            )

    # Apply subtle blur to match local sharpness
    if blur_sigma > 0.1:
        result = cv2.GaussianBlur(result.astype(np.uint8), (3, 3), blur_sigma)
        result = result.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Blending ─────────────────────────────────────────────────────────────────

def blend_feather(
    original: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 3,
) -> np.ndarray:
    """
    Feather blend patch onto original using a Gaussian-blurred mask.
    """
    sigma = max(1, feather_px)
    blur_ks = sigma * 6 + 1
    mask_f = cv2.GaussianBlur(mask, (blur_ks, blur_ks), sigma).astype(np.float32) / 255.0

    result = original.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (
            original[:, :, c].astype(np.float32) * (1 - mask_f) +
            patch[:, :, c].astype(np.float32) * mask_f
        )
    return result.astype(np.uint8)


def blend_seamless(
    original: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """
    OpenCV seamlessClone blending. Tries MIXED_CLONE first, falls back to NORMAL_CLONE.
    """
    center = quad.mean(axis=0)[::-1]  # (x,y) -> (y,x) for opencv
    center = (int(center[0]), int(center[1]))

    try:
        # Ensure center is within bounds
        h, w = original.shape[:2]
        cx = max(1, min(w - 2, center[0]))
        cy = max(1, min(h - 2, center[1]))
        result = cv2.seamlessClone(
            patch.astype(np.uint8), original, mask,
            (cx, cy), cv2.MIXED_CLONE
        )
        return result
    except Exception:
        pass

    try:
        h, w = original.shape[:2]
        cx = max(1, min(w - 2, center[0]))
        cy = max(1, min(h - 2, center[1]))
        result = cv2.seamlessClone(
            patch.astype(np.uint8), original, mask,
            (cx, cy), cv2.NORMAL_CLONE
        )
        return result
    except Exception:
        pass

    # Fallback to feather
    return blend_feather(original, patch, mask, feather_px=3)


# ─── Final decode ─────────────────────────────────────────────────────────────

def _final_decode(result_img: np.ndarray) -> Optional[str]:
    """Re-validate output by decoding the QR."""
    detector = cv2.QRCodeDetector()

    try:
        retval, points = detector.detect(result_img)
        if retval and points is not None:
            text, _ = detector.decode(result_img, points)
            if text:
                return text
    except Exception:
        pass

    try:
        retval, decoded_list, points, _ = detector.detectAndDecodeMulti(result_img)
        if retval and decoded_list:
            return decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
    except Exception:
        pass

    return None


# ─── Main pipeline ────────────────────────────────────────────────────────────

def replace_qr_pipeline(
    input_image_path: str,
    new_payload: str,
    output_path: str,
    debug_dir: Optional[str] = None,
    mode: str = "single",
    replace_all: bool = False,
    blend_mode: str = "feather",       # "feather" or "seamless"
    carrier_expand_ratio: float = 1.45,
    qr_scale_within_patch: float = 0.72,
    edge_feather_px: int = 3,
    blur_sigma: float = 0.0,            # 0 = auto
    jpeg_simulation: bool = True,
    verify: bool = True,
    qr_fg: tuple = (0, 0, 0),
    qr_bg: tuple = (255, 255, 255),
) -> QRReplacementResult:
    """
    Full pipeline:
    1. Multi-preprocessing QR detection
    2. Carrier patch expansion
    3. Local appearance estimation
    4. Carrier patch generation with realism
    5. Perspective warp
    6. Appearance matching
    7. Blending (feather or seamless)
    8. Verification with retries
    """
    os.makedirs(debug_dir, exist_ok=True) if debug_dir else None

    img = cv2.imread(input_image_path)
    if img is None:
        return QRReplacementResult(
            output_image_path=output_path, detected_points=None,
            old_decoded_text=None, new_decoded_text=None, success=False,
            reason_if_failed=f"Cannot load image: {input_image_path}"
        )

    h, w = img.shape[:2]
    debug_report = {"steps": []}

    # ─ Detect ─
    points, old_decoded, method_used = detect_qr_chain(img, debug_dir=debug_dir, mode=mode)

    if points is None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_no_detection.png"), img)
        return QRReplacementResult(
            output_image_path=output_path, detected_points=None,
            old_decoded_text=old_decoded, new_decoded_text=None, success=False,
            method_used=None, reason_if_failed="No QR code detected in any preprocessing variant"
        )

    # Multi-QR handling
    multi_mode = isinstance(points, np.ndarray) and len(points.shape) == 3 and points.shape[1] > 1
    if multi_mode and not replace_all:
        # Pick the most central QR
        img_center = np.array([w / 2, h / 2])
        best_idx = min(range(len(points)), key=lambda i: np.linalg.norm(points[i].mean(axis=0) - img_center))
        points = points[best_idx]

    pts_ordered = order_points(np.array(points))

    if debug_dir:
        dbg = img.copy()
        cv2.polylines(dbg, [pts_ordered.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(debug_dir, "03_detected_region.png"), dbg)

    debug_report["steps"].append({"step": "detect", "method": method_used, "decoded": old_decoded})

    # ─ Estimate local appearance BEFORE replacing ─
    local_appearance = estimate_local_appearance(img, pts_ordered, ring_width=20)
    debug_report["steps"].append({
        "step": "local_appearance",
        "mean_bgr": local_appearance["mean_bgr"].tolist(),
        "contrast": local_appearance["contrast"],
        "sharpness": local_appearance["sharpness"]
    })

    # ─ Expand to carrier patch ─
    expanded_pts = expand_carrier_quad(pts_ordered, carrier_expand_ratio, h, w)

    if debug_dir:
        dbg2 = img.copy()
        cv2.polylines(dbg2, [expanded_pts.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.polylines(dbg2, [pts_ordered.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.imwrite(os.path.join(debug_dir, "04_carrier_expanded.png"), dbg2)

    # Compute carrier bounding box size
    carrier_bb = cv2.boundingRect(expanded_pts.astype(np.int32))
    carrier_w = max(carrier_bb[2], 10)
    carrier_h = max(carrier_bb[3], 10)

    # Ensure carrier is at least as large as the QR (330x330 for typical payloads)
    # so the QR doesn't need to be resized before warping
    qr_dummy = generate_qr_image(new_payload)
    qr_h, qr_w = qr_dummy.shape[:2]
    carrier_size = (max(carrier_w, qr_w), max(carrier_h, qr_h))

    debug_report["steps"].append({
        "step": "carrier_expand",
        "ratio": carrier_expand_ratio,
        "carrier_size": carrier_size,
        "expanded_points": expanded_pts.tolist()
    })

    # ─ Generate carrier patch with QR (full-size, no resize) ─
    carrier_patch, qr_mask, patch_mask = generate_carrier_patch(
        new_payload,
        carrier_size,
        fg_color=qr_fg,
        bg_color=qr_bg,
        local_appearance=local_appearance,
        add_texture=True,
        jpeg_simulation=jpeg_simulation,
    )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "05_carrier_patch.png"), carrier_patch)

    # ─ Warp carrier patch to destination ─
    warped_patch = perspective_warp(carrier_patch, expanded_pts, (h, w))
    warped_qr_mask = perspective_warp(qr_mask, expanded_pts, (h, w))
    warped_carrier_mask = perspective_warp(patch_mask, expanded_pts, (h, w))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "06_warped_patch.png"), warped_patch)
        cv2.imwrite(os.path.join(debug_dir, "07_warped_qr_mask.png"), warped_qr_mask)
        cv2.imwrite(os.path.join(debug_dir, "08_warped_carrier_mask.png"), warped_carrier_mask)

    # ─ Auto blur sigma if not set ─
    if blur_sigma <= 0:
        # Estimate blur from local sharpness
        blur_sigma = max(0.3, min(local_appearance["sharpness"] * 0.4, 1.5))

    debug_report["steps"].append({"step": "appearance", "blur_sigma": blur_sigma})

    # ─ Appearance matching ─
    matched_patch = match_appearance_to_local(
        warped_patch, warped_carrier_mask, img, expanded_pts, blur_sigma=blur_sigma
    )

    # ─ Blending ─
    if blend_mode == "seamless":
        blended = blend_seamless(img, matched_patch, warped_carrier_mask, expanded_pts)
        used_blend = "seamless"
    else:
        blended = blend_feather(img, matched_patch, warped_carrier_mask, feather_px=edge_feather_px)
        used_blend = "feather"

    debug_report["steps"].append({"step": "blend", "mode": used_blend})

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "09_blended.png"), blended)

    # ─ Verify with retries ─
    final_result = blended.copy()
    success = False
    new_decoded = None
    retries = 0
    last_error = None

    # Retry strategies in order
    retry_strategies = [
        {"blur_sigma": max(0.1, blur_sigma - 0.2), "edge_feather_px": max(1, edge_feather_px - 1)},
        {"blur_sigma": max(0.1, blur_sigma - 0.4), "edge_feather_px": max(1, edge_feather_px - 2)},
        {"blend_mode": "feather", "edge_feather_px": 2},
        {"blend_mode": "feather", "edge_feather_px": 1},
    ]

    for strategy in retry_strategies:
        if not verify:
            break
        retries += 1
        s_blur = strategy.get("blur_sigma", blur_sigma)
        s_feather = strategy.get("edge_feather_px", edge_feather_px)
        s_blend = strategy.get("blend_mode", blend_mode)

        # Re-apply appearance matching with new sigma
        retried_patch = match_appearance_to_local(
            warped_patch, warped_carrier_mask, img, expanded_pts, blur_sigma=s_blur
        )
        if s_blend == "seamless":
            retried = blend_seamless(img, retried_patch, warped_carrier_mask, expanded_pts)
        else:
            retried = blend_feather(img, retried_patch, warped_carrier_mask, feather_px=s_feather)

        new_decoded = _final_decode(retried)
        if new_decoded == new_payload:
            final_result = retried
            success = True
            used_blend = s_blend
            debug_report["steps"].append({
                "step": "verify_success", "strategy": strategy, "retries": retries
            })
            break
        else:
            last_error = f"decode={new_decoded}, expected={new_payload}"

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "10_final.png"), final_result)
        with open(os.path.join(debug_dir, "pipeline_report.json"), "w") as f:
            json.dump(debug_report, f, indent=2)

    return QRReplacementResult(
        output_image_path=output_path,
        detected_points=pts_ordered,
        old_decoded_text=old_decoded,
        new_decoded_text=new_decoded if success else None,
        success=success,
        method_used=method_used,
        selected_preprocess_variant=method_used,
        expanded_points=expanded_pts,
        selected_blend_mode=used_blend,
        retries_used=retries,
        reason_if_failed=last_error if not success else None,
        debug_report=debug_report,
    )
