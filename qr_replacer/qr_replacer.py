"""
QR Code Replacer Pipeline v0.4.0
===============================
Deterministic pipeline for detecting and replacing QR codes in images.

v0.4 improvements:
- Smart carrier detection: scan outward from QR to find paper boundaries
- Multi-candidate scoring: generate candidates and pick the best
- Content protection: penalize expansions covering text/graphics
- Correct carrier sizing: max(expanded_bb, qr_size) not qr_size * ratio
- Auto-select best expansion ratio from validated candidates
"""

import cv2
import numpy as np
import qrcode
import os
import json
from PIL import Image
from dataclasses import dataclass, field
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
    """Low-level detect + decode using separate calls."""
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


def _preprocess_variants(img: np.ndarray):
    """Generate multiple preprocessing variants for detection."""
    variants = []
    variants.append(("bgr", img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
    )
    variants.append(("adaptive", adaptive))
    inv_gray = cv2.bitwise_not(gray)
    variants.append(("inv_gray", inv_gray))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cl))
    return variants


def detect_qr_chain(img: np.ndarray, debug_dir: Optional[str] = None, mode: str = "single") -> tuple:
    """Detection pipeline with multiple preprocessing variants."""
    detector = cv2.QRCodeDetector()
    all_results = []

    for name, variant in _preprocess_variants(img):
        try:
            if variant is img or len(variant.shape) == 2:
                retval, points = detector.detect(variant)
            else:
                retval, points = detector.detect(
                    cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if len(variant.shape) == 3 else variant
                )
            if retval and points is not None:
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
                    "name": name, "points": points, "decoded": decoded, "variant": variant,
                })
        except Exception:
            pass

    if not all_results:
        return None, None, None

    if mode == "single" or len(all_results) == 1:
        def score(r):
            area = cv2.contourArea(r["points"].astype(np.float32).reshape(4, 2))
            return (1 if r["decoded"] else 0, area)
        all_results.sort(key=score, reverse=True)
        best = all_results[0]
        return best["points"], best["decoded"], best["name"]

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
        pts[np.argmin(s)], pts[np.argmin(diff)],
        pts[np.argmax(s)], pts[np.argmin(-diff)],
    ], dtype=np.float32)


# ─── Smart carrier detection ──────────────────────────────────────────────────

def _scan_edge_brightness(
    img: np.ndarray,
    start_pt: np.ndarray,
    direction: np.ndarray,
    max_px: int = 80,
    step_px: int = 5,
) -> dict:
    """
    Scan outward from start_pt in direction, measuring brightness.
    Returns dict with:
      - clean_paper_px: how many consecutive bright pixels before hitting dark
      - first_dark_px: distance to first dark pixel
      - mean_brightness: mean brightness in scanned region
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return {"clean_paper_px": 0, "first_dark_px": max_px, "mean_brightness": 200}
    dir_unit = (direction / norm) * step_px

    clean_count = 0
    first_dark = None
    brightness_sum = 0.0
    brightness_count = 0

    for step in range(1, max_px // step_px + 1):
        sy = int(start_pt[1] + dir_unit[1] * step)
        sx = int(start_pt[0] + dir_unit[0] * step)
        if not (0 <= sy < img.shape[0] and 0 <= sx < img.shape[1]):
            break
        b = int(gray[sy, sx])  # convert to int to avoid overflow
        brightness_sum += b
        brightness_count += 1
        if b > 180:
            clean_count += step_px
        elif first_dark is None:
            first_dark = step * step_px

    return {
        "clean_paper_px": clean_count,
        "first_dark_px": first_dark if first_dark is not None else max_px,
        "mean_brightness": brightness_sum / brightness_count if brightness_count > 0 else 0,
    }


def detect_carrier_region(img: np.ndarray, qr_pts: np.ndarray) -> dict:
    """
    Analyze the image around the QR code to detect the carrier region.

    Scans outward from each QR corner to find:
    - How far the clean white paper extends
    - Where dark content (leather/text) begins
    - The safe expansion margin in each direction

    Returns dict with per-direction analysis and recommended expansion.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Sample paper brightness from ring around QR
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [qr_pts.astype(np.int32)], 255)
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    border_mask = cv2.subtract(dilated, mask)
    if cv2.countNonZero(border_mask) > 0:
        paper_bgr = cv2.bitwise_and(img, img, mask=border_mask).sum(axis=(0, 1)) / cv2.countNonZero(border_mask)
    else:
        paper_bgr = np.array([220, 220, 220])

    center = qr_pts.mean(axis=0)
    result = {
        "paper_bgr": paper_bgr.tolist(),
        "corners": {},
        "recommended_expand_px": {},
    }

    # Scan from each corner
    for i, (name, corner) in enumerate(zip(
        ["TL", "TR", "BR", "BL"],
        qr_pts
    )):
        direction = corner - center
        scan_result = _scan_edge_brightness(img, corner, direction, max_px=80, step_px=5)
        result["corners"][name] = scan_result

        # Safe expansion = 80% of clean paper distance, minimum 5px
        safe = max(5, int(scan_result["clean_paper_px"] * 0.8))
        result["recommended_expand_px"][name] = safe

    # Overall recommended expansion ratio
    avg_safe = np.mean(list(result["recommended_expand_px"].values()))
    # Also check if there are nearby content threats
    content_threats = _detect_content_threats(img, qr_pts)
    result["content_threats"] = content_threats

    return result


def _detect_content_threats(img: np.ndarray, qr_pts: np.ndarray) -> dict:
    """
    Detect if there are text/graphics near the QR that could be covered.
    Returns dict with threat info for each direction.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    center = qr_pts.mean(axis=0)

    threats = {}
    dir_names = ["top", "bottom", "left", "right"]
    # Approximate directions from center
    dir_vectors = [
        np.array([0, -1]),   # top
        np.array([0, 1]),    # bottom
        np.array([-1, 0]),   # left
        np.array([1, 0]),    # right
    ]

    for name, dvec in zip(dir_names, dir_vectors):
        # Sample a grid of pixels in a box around the QR in this direction
        side_px = 15
        ahead_px = 60
        samples = []

        # Get points perpendicular to direction
        perp = np.array([-dvec[1], dvec[0]])
        half_w = side_px * 2

        for t in range(5, ahead_px, 5):
            for s in range(-half_w, half_w + 1, 5):
                sample_y = int(center[1] + dvec[1] * t + perp[1] * s)
                sample_x = int(center[0] + dvec[0] * t + perp[0] * s)
                if 0 <= sample_y < h and 0 <= sample_x < w:
                    samples.append(gray[sample_y, sample_x])

        if samples:
            samples = np.array(samples)
            # High variance = text/graphics, low variance = uniform paper
            variance = samples.std()
            dark_ratio = (samples < 100).sum() / len(samples)
            threats[name] = {
                "variance": float(variance),
                "dark_ratio": float(dark_ratio),
                "has_content": bool(variance > 30 or dark_ratio > 0.1),
            }
        else:
            threats[name] = {"variance": 0, "dark_ratio": 0, "has_content": False}

    return threats


def compute_smart_expansion(
    qr_pts: np.ndarray,
    carrier_info: dict,
    img_h: int,
    img_w: int,
    strategy: str = "smart",
) -> np.ndarray:
    """
    Compute expanded quad based on smart carrier detection.

    strategy:
      - "tight": minimal 5px padding in all directions
      - "smart": use safe expansion per corner based on paper scan
      - "conservative": expand only in directions without content threats
    """
    center = qr_pts.mean(axis=0)

    if strategy == "tight":
        # Simple uniform 5px expansion per corner
        dirs = qr_pts - center
        expanded = center + dirs * 1.05
    elif strategy == "smart":
        # Per-corner expansion based on scan data
        # Use the recommended safe expansion per corner
        safe_px = carrier_info.get("recommended_expand_px", {})
        dirs = qr_pts - center
        norms = np.linalg.norm(dirs, axis=1)
        expanded_pts_list = []
        for i, (corner, name) in enumerate(zip(qr_pts, ["TL", "TR", "BR", "BL"])):
            dir_norm = norms[i]
            if dir_norm > 1e-6:
                dir_unit = (corner - center) / dir_norm
                px = safe_px.get(name, 15)
                # Expand by pixel amount
                new_pt = corner + dir_unit * px
            else:
                new_pt = corner
            expanded_pts_list.append(new_pt)
        expanded = np.array(expanded_pts_list, dtype=np.float32)
    elif strategy == "conservative":
        # Only expand where there's no content threat
        threats = carrier_info.get("content_threats", {})
        safe_px = carrier_info.get("recommended_expand_px", {})
        dirs = qr_pts - center
        norms = np.linalg.norm(dirs, axis=1)
        expanded_pts_list = []
        corner_threats = [
            threats.get("top", {}).get("has_content", False),    # TL, TR
            threats.get("top", {}).get("has_content", False),    # TL, TR
            threats.get("bottom", {}).get("has_content", False), # BR, BL
            threats.get("bottom", {}).get("has_content", False), # BR, BL
        ]
        for i, (corner, name) in enumerate(zip(qr_pts, ["TL", "TR", "BR", "BL"])):
            dir_norm = norms[i]
            if dir_norm > 1e-6:
                dir_unit = (corner - center) / dir_norm
                px = safe_px.get(name, 15) if not corner_threats[i] else 5
                new_pt = corner + dir_unit * px
            else:
                new_pt = corner
            expanded_pts_list.append(new_pt)
        expanded = np.array(expanded_pts_list, dtype=np.float32)
    else:
        dirs = qr_pts - center
        expanded = center + dirs * 1.2

    # Clamp to image bounds
    margin = 5
    expanded[:, 0] = np.clip(expanded[:, 0], margin, img_w - margin)
    expanded[:, 1] = np.clip(expanded[:, 1], margin, img_h - margin)
    return expanded.astype(np.float32)


def expand_carrier_quad(pts: np.ndarray, ratio: float, img_h: int, img_w: int) -> np.ndarray:
    """Expand quad outward from center by ratio. DEPRECATED: use compute_smart_expansion."""
    center = pts.mean(axis=0)
    dirs = pts - center
    expanded = center + dirs * ratio
    margin = 5
    expanded[:, 0] = np.clip(expanded[:, 0], margin, img_w - margin)
    expanded[:, 1] = np.clip(expanded[:, 1], margin, img_h - margin)
    return expanded.astype(np.float32)


# ─── QR Generation ─────────────────────────────────────────────────────────────

def _qr_version_capacity(version: int) -> int:
    capacities = {1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134, 7: 154, 8: 192,
                  9: 230, 10: 271, 11: 321, 12: 367, 13: 425, 14: 458}
    return capacities.get(version, 1000)


def generate_qr_image(
    payload: str,
    quiet_zone: int = 4,
    fg_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Generate clean QR code image. warpPerspective handles all scaling."""
    payload_len = len(payload)
    version = 1
    for v in range(1, 40):
        if payload_len <= _qr_version_capacity(v):
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
    return np.array(img.convert("RGB"))[:, :, ::-1]


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
    Generate carrier patch: paper-colored background with QR centered.
    Carrier size must be >= QR size (no resize of QR).
    Returns (patch, qr_only_mask, carrier_mask).
    """
    patch_w, patch_h = carrier_size

    # Paper-colored carrier background
    if local_appearance is not None and "mean_bgr" in local_appearance:
        base_bgr = local_appearance["mean_bgr"].astype(np.uint8)
        patch = np.full((patch_h, patch_w, 3), base_bgr[::-1], dtype=np.uint8)
    else:
        patch = np.full((patch_h, patch_w, 3), 255, dtype=np.uint8)

    qr_img = generate_qr_image(payload, quiet_zone=4, fg_color=fg_color, bg_color=bg_color)
    qr_h, qr_w = qr_img.shape[:2]
    x_offset = max(0, (patch_w - qr_w) // 2)
    y_offset = max(0, (patch_h - qr_h) // 2)
    y_end = min(y_offset + qr_h, patch_h)
    x_end = min(x_offset + qr_w, patch_w)
    patch[y_offset:y_end, x_offset:x_end] = qr_img[:y_end - y_offset, :x_end - x_offset]

    qr_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, qr_only_mask = cv2.threshold(qr_gray, 128, 255, cv2.THRESH_BINARY_INV)
    carrier_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cv2.rectangle(carrier_mask, (0, 0), (patch_w - 1, patch_h - 1), 255, -1)

    if add_texture:
        bg_mask = (qr_gray > 200).astype(np.float32)
        noise = np.random.normal(0, 3.0, patch.shape).astype(np.float32)
        textured = np.clip(patch.astype(np.float32) + noise * bg_mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
        textured = cv2.GaussianBlur(textured, (3, 3), 0.4)
        patch = np.where(bg_mask[:, :, np.newaxis] > 0, textured, patch)

    if jpeg_simulation:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, enc = cv2.imencode(".jpg", patch, encode_param)
        patch = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return patch, qr_only_mask, carrier_mask


# ─── Perspective warp ─────────────────────────────────────────────────────────

def perspective_warp(src_img: np.ndarray, dst_points: np.ndarray, output_size: tuple) -> np.ndarray:
    """Warp src_img to fit the destination quadrilateral."""
    h, w = src_img.shape[:2]
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = np.array([
        dst_points[0], dst_points[1], dst_points[2], dst_points[3]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return cv2.warpPerspective(src_img, H, (output_size[1], output_size[0]))


def perspective_warp_with_mask(
    src_img: np.ndarray, src_mask: np.ndarray,
    dst_points: np.ndarray, output_size: tuple,
) -> tuple:
    """Warp image and mask together."""
    h, w = src_img.shape[:2]
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
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
    """Apply local appearance matching to warped patch."""
    result = warped_patch.copy().astype(np.float32)
    original = original_img.copy().astype(np.float32)
    patch_mask = (warped_mask > 0).astype(np.float32)

    outer_ring = cv2.subtract(
        cv2.dilate(warped_mask, np.ones((7, 7), np.uint8), iterations=2),
        warped_mask
    )
    outer_pixels = original_img[outer_ring > 0].astype(np.float32)
    if len(outer_pixels) > 0:
        local_mean = outer_pixels.mean(axis=0)
        patch_mean = result[patch_mask > 0].mean(axis=0)
        adjust = local_mean - patch_mean
        for c in range(3):
            result[:, :, c] = np.where(
                patch_mask > 0,
                np.clip(result[:, :, c] + adjust[c] * 0.6, 0, 255),
                original[:, :, c]
            )

    if blur_sigma > 0.1:
        result = cv2.GaussianBlur(result.astype(np.uint8), (3, 3), blur_sigma)
        result = result.astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


# ─── Blending ─────────────────────────────────────────────────────────────────

def blend_feather(
    original: np.ndarray, patch: np.ndarray, mask: np.ndarray, feather_px: int = 3,
) -> np.ndarray:
    """Feather blend patch onto original using Gaussian-blurred mask."""
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
    original: np.ndarray, patch: np.ndarray, mask: np.ndarray, quad: np.ndarray,
) -> np.ndarray:
    """OpenCV seamlessClone blending."""
    center = quad.mean(axis=0)[::-1]
    center = (int(center[0]), int(center[1]))
    h, w = original.shape[:2]
    cx = max(1, min(w - 2, center[0]))
    cy = max(1, min(h - 2, center[1]))
    try:
        return cv2.seamlessClone(
            patch.astype(np.uint8), original, mask, (cx, cy), cv2.MIXED_CLONE
        )
    except Exception:
        pass
    try:
        return cv2.seamlessClone(
            patch.astype(np.uint8), original, mask, (cx, cy), cv2.NORMAL_CLONE
        )
    except Exception:
        pass
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


# ─── Candidate scoring ─────────────────────────────────────────────────────────

def _score_candidate(
    candidate_img: np.ndarray,
    original_img: np.ndarray,
    quad: np.ndarray,
    payload: str,
    carrier_info: dict,
) -> dict:
    """
    Score a candidate composite image.
    Returns dict with scores for: decode_success, brightness_match, content_protection, size_penalty.
    """
    detector = cv2.QRCodeDetector()
    scores = {"decode_success": 0, "brightness_match": 0.0, "content_protection": 0.0, "size_score": 0.0}

    # 1. Decode check
    decoded = _final_decode(candidate_img)
    scores["decode_success"] = 1.0 if decoded == payload else 0.0
    if scores["decode_success"] < 0.5:
        return scores

    # 2. Brightness match: compare carrier area to surrounding paper brightness
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    carrier_pixels = candidate_img[mask > 0].astype(np.float32)
    paper_bgr = np.array(carrier_info.get("paper_bgr", [220, 220, 220]))
    if len(carrier_pixels) > 0:
        carrier_mean = carrier_pixels.mean(axis=0)
        brightness_diff = np.abs(carrier_mean - paper_bgr).mean()
        scores["brightness_match"] = max(0, 1.0 - brightness_diff / 50.0)

    # 3. Content protection: penalize if quad covers content
    threats = carrier_info.get("content_threats", {})
    threat_penalty = sum(1 for t in threats.values() if t.get("has_content", False))
    scores["content_protection"] = max(0, 1.0 - threat_penalty * 0.2)

    # 4. Size score: prefer smaller carriers (less is more)
    quad_area = cv2.contourArea(quad)
    scores["size_score"] = 1.0  # normalized

    return scores


# ─── Estimate local appearance ─────────────────────────────────────────────────

def estimate_local_appearance(img: np.ndarray, quad: np.ndarray, ring_width: int = 15) -> dict:
    """Sample surrounding ring region to estimate local appearance."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    kernel = np.ones((ring_width, ring_width), np.uint8)
    dilated = cv2.dilate(mask, kernel)
    ring_mask = cv2.subtract(dilated, mask)
    ring_pixels = img[ring_mask > 0]

    if len(ring_pixels) == 0:
        return {"mean_bgr": np.array([220, 220, 220]), "contrast": 20.0,
                "color_cast": np.array([0, 0, 0]), "sharpness": 1.0}

    mean_bgr = ring_pixels.mean(axis=0).astype(np.float32)
    gray_ring = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[ring_mask > 0]
    contrast = float(gray_ring.std())
    luminance = mean_bgr.mean()
    color_cast = mean_bgr - luminance
    try:
        lap_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, mask=ring_mask).var()
        sharpness = min(lap_var / 100.0, 3.0)
    except Exception:
        sharpness = 1.0

    return {"mean_bgr": mean_bgr, "contrast": contrast,
            "color_cast": color_cast, "sharpness": float(sharpness)}


# ─── Main pipeline ────────────────────────────────────────────────────────────

def replace_qr_pipeline(
    input_image_path: str,
    new_payload: str,
    output_path: str,
    debug_dir: Optional[str] = None,
    mode: str = "single",
    replace_all: bool = False,
    blend_mode: str = "feather",
    carrier_expand_ratio: float = 1.2,
    edge_feather_px: int = 3,
    blur_sigma: float = 0.0,
    jpeg_simulation: bool = True,
    verify: bool = True,
    qr_fg: tuple = (0, 0, 0),
    qr_bg: tuple = (255, 255, 255),
) -> QRReplacementResult:
    """
    Full pipeline with smart carrier detection and multi-candidate scoring.

    Key improvements over v0.3:
    - Smart carrier detection: scan for paper boundaries
    - Multi-candidate: tight, smart, and conservative candidates
    - Auto-select best candidate that decodes AND protects content
    - Correct carrier sizing: max(expanded_bb, qr_size)
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
    debug_report = {"steps": [], "candidates": []}

    # ─ Detect ─
    points, old_decoded, method_used = detect_qr_chain(img, debug_dir=debug_dir, mode=mode)

    if points is None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_no_detection.png"), img)
        return QRReplacementResult(
            output_image_path=output_path, detected_points=None,
            old_decoded_text=old_decoded, new_decoded_text=None, success=False,
            method_used=None, reason_if_failed="No QR code detected"
        )

    multi_mode = isinstance(points, np.ndarray) and len(points.shape) == 3 and points.shape[1] > 1
    if multi_mode and not replace_all:
        img_center = np.array([w / 2, h / 2])
        best_idx = min(range(len(points)), key=lambda i: np.linalg.norm(points[i].mean(axis=0) - img_center))
        points = points[best_idx]

    pts_ordered = order_points(np.array(points))

    if debug_dir:
        dbg = img.copy()
        cv2.polylines(dbg, [pts_ordered.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(debug_dir, "03_detected_region.png"), dbg)

    debug_report["steps"].append({"step": "detect", "method": method_used, "decoded": old_decoded})

    # ─ Smart carrier detection ─
    carrier_info = detect_carrier_region(img, pts_ordered)
    debug_report["steps"].append({
        "step": "carrier_detection",
        "paper_bgr": carrier_info["paper_bgr"],
        "expand_px": carrier_info["recommended_expand_px"],
        "content_threats": {
            k: {"has_content": v["has_content"]} for k, v in carrier_info["content_threats"].items()
        }
    })

    # ─ Local appearance ─
    local_appearance = estimate_local_appearance(img, pts_ordered, ring_width=20)
    debug_report["steps"].append({
        "step": "local_appearance",
        "mean_bgr": local_appearance["mean_bgr"].tolist(),
        "contrast": local_appearance["contrast"],
        "sharpness": local_appearance["sharpness"]
    })

    # ─ Generate candidates ─
    qr_dummy = generate_qr_image(new_payload)
    qr_h, qr_w = qr_dummy.shape[:2]

    candidate_strategies = [
        ("tight", "tight"),
        ("smart", "smart"),
        ("conservative", "conservative"),
    ]

    candidates = []
    for strat_name, strategy in candidate_strategies:
        expanded = compute_smart_expansion(pts_ordered, carrier_info, h, w, strategy=strategy)
        carrier_bb = cv2.boundingRect(expanded.astype(np.int32))
        cw = max(carrier_bb[2], qr_w)
        ch = max(carrier_bb[3], qr_h)
        carrier_size = (cw, ch)

        # Generate carrier patch
        carrier_patch, qr_mask, patch_mask = generate_carrier_patch(
            new_payload, carrier_size,
            fg_color=qr_fg, bg_color=qr_bg,
            local_appearance=local_appearance,
            add_texture=True, jpeg_simulation=jpeg_simulation,
        )

        # Warp
        warped_patch = perspective_warp(carrier_patch, expanded, (h, w))
        warped_carrier_mask = perspective_warp(patch_mask, expanded, (h, w))

        # Auto blur
        sigma = blur_sigma if blur_sigma > 0 else max(0.3, min(local_appearance["sharpness"] * 0.4, 1.5))
        matched = match_appearance_to_local(warped_patch, warped_carrier_mask, img, expanded, blur_sigma=sigma)

        # Blend
        if blend_mode == "seamless":
            blended = blend_seamless(img, matched, warped_carrier_mask, expanded)
        else:
            blended = blend_feather(img, matched, warped_carrier_mask, feather_px=edge_feather_px)

        # Score
        scores = _score_candidate(blended, img, expanded, new_payload, carrier_info)
        total_score = (
            scores["decode_success"] * 5.0 +
            scores["brightness_match"] * 1.0 +
            scores["content_protection"] * 2.0 -
            0.1 * (carrier_bb[2] * carrier_bb[3]) / 10000
        )

        candidates.append({
            "strategy": strat_name,
            "expanded_pts": expanded.copy(),
            "carrier_size": carrier_size,
            "blended": blended,
            "scores": scores,
            "total_score": total_score,
            "sigma": sigma,
        })

        if debug_dir:
            dbg_c = img.copy()
            cv2.polylines(dbg_c, [expanded.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)
            cv2.imwrite(os.path.join(debug_dir, f"04_candidate_{strat_name}.png"), dbg_c)
            cv2.imwrite(os.path.join(debug_dir, f"05_{strat_name}_blended.png"), blended)

        debug_report["candidates"].append({
            "strategy": strat_name,
            "carrier_size": carrier_size,
            "scores": scores,
            "total_score": total_score,
            "expand_pts": expanded.tolist(),
        })

    # ─ Select best candidate ─
    candidates.sort(key=lambda c: c["total_score"], reverse=True)
    best = candidates[0]
    final_result = best["blended"]
    selected_strategy = best["strategy"]
    expanded_pts = best["expanded_pts"]
    sigma = best["sigma"]

    debug_report["steps"].append({
        "step": "candidate_selected",
        "strategy": selected_strategy,
        "total_score": best["total_score"],
        "scores": best["scores"],
        "sigma": sigma,
    })

    # ─ Verify with retries if needed ─
    success = best["scores"]["decode_success"] >= 0.5
    new_decoded = None
    retries = 0
    last_error = None

    if not success:
        retry_strategies = [
            {"blur_sigma": max(0.1, sigma - 0.2), "edge_feather_px": max(1, edge_feather_px - 1)},
            {"blur_sigma": max(0.1, sigma - 0.4), "edge_feather_px": max(1, edge_feather_px - 2)},
            {"blend_mode": "feather", "edge_feather_px": 2},
            {"blend_mode": "feather", "edge_feather_px": 1},
        ]
        for strat in retry_strategies:
            if not verify:
                break
            retries += 1
            s_blur = strat.get("blur_sigma", sigma)
            s_feather = strat.get("edge_feather_px", edge_feather_px)
            s_blend = strat.get("blend_mode", blend_mode)

            retried_matched = match_appearance_to_local(
                perspective_warp(generate_carrier_patch(new_payload, best["carrier_size"],
                    fg_color=qr_fg, bg_color=qr_bg, local_appearance=local_appearance,
                    add_texture=True, jpeg_simulation=jpeg_simulation)[0],
                    expanded_pts, (h, w)),
                perspective_warp(generate_carrier_patch(new_payload, best["carrier_size"],
                    fg_color=qr_fg, bg_color=qr_bg)[1], expanded_pts, (h, w)),
                img, expanded_pts, blur_sigma=s_blur
            )
            if s_blend == "seamless":
                retried = blend_seamless(img, retried_matched, warped_carrier_mask, expanded_pts)
            else:
                retried = blend_feather(img, retried_matched, warped_carrier_mask, feather_px=s_feather)

            new_decoded = _final_decode(retried)
            if new_decoded == new_payload:
                final_result = retried
                success = True
                debug_report["steps"].append({"step": "verify_success", "strategy": strat, "retries": retries})
                break
            else:
                last_error = f"decode={new_decoded}"

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "10_final.png"), final_result)
        with open(os.path.join(debug_dir, "pipeline_report.json"), "w") as f:
            json.dump(debug_report, f, indent=2)

    return QRReplacementResult(
        output_image_path=output_path,
        detected_points=pts_ordered,
        old_decoded_text=old_decoded,
        new_decoded_text=new_payload if success else None,
        success=success,
        method_used=method_used,
        expanded_points=expanded_pts,
        selected_blend_mode=blend_mode,
        retries_used=retries,
        reason_if_failed=last_error if not success else None,
        debug_report=debug_report,
    )
