"""
QR Code Replacer Pipeline
=========================
Deterministic pipeline for detecting and replacing QR codes in images.

Improvements over v0.1:
- Proper QR version/size selection to fill detected region
- Debug image saving for each detection stage
- Robust decode fallback chain
- Color QR generation support
- Better feather blending with edge smoothing
"""

import cv2
import numpy as np
import qrcode
import os
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class QRReplacementResult:
    output_image_path: str
    detected_points: Optional[np.ndarray]
    old_decoded_text: Optional[str]
    new_decoded_text: Optional[str]
    success: bool
    method_used: Optional[str] = None


def _detect_raw(img: np.ndarray) -> tuple:
    """
    Low-level detect + decode using separate calls.
    Returns (points, decoded_text) or (None, None).
    """
    detector = cv2.QRCodeDetector()
    try:
        retval, points = detector.detect(img)
        if retval and points is not None and len(points) > 0:
            decoded_text, _ = detector.decode(img, points)
            return points, decoded_text
    except Exception:
        pass
    return None, None


def _detect_multi(img: np.ndarray) -> tuple:
    """detectAndDecodeMulti fallback. Returns (points, decoded_text)."""
    detector = cv2.QRCodeDetector()
    try:
        retval, decoded_list, points, _ = detector.detectAndDecodeMulti(img)
        if retval and points is not None and len(points) > 0:
            text = decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
            return points, text
    except Exception:
        pass
    return None, None


def detect_qr_chain(img: np.ndarray, debug_dir: Optional[str] = None) -> tuple:
    """
    Robust detection fallback chain using separate detect() + decode().
    Returns: (points, decoded_text, method_used)
    """
    h, w = img.shape[:2]

    # Preprocessing variants
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_gray = cv2.bitwise_not(gray)

    # CLAHE enhanced grayscale (helps with uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)

    variants = [
        (img, "bgr"),
        (gray, "gray"),
        (thresh, "otsu"),
        (inv_gray, "inv_gray"),
        (clahe_gray, "clahe"),
    ]

    # Save debug preprocess images
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for proc_img, name in variants:
            disp = proc_img if len(proc_img.shape) == 2 else cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(debug_dir, f"01_preproc_{name}.png"), disp)

    # Try raw detect+decode on each variant
    for proc_img, name in variants:
        points, text = _detect_raw(proc_img)
        if points is not None:
            if debug_dir:
                _save_detection_overlay(img, points, os.path.join(debug_dir, f"02_detected_{name}.png"))
            return points, text, name

    # Multi decode fallback
    for proc_img, name in variants:
        points, text = _detect_multi(proc_img)
        if points is not None:
            if debug_dir:
                _save_detection_overlay(img, points, os.path.join(debug_dir, f"02_detected_multi_{name}.png"))
            return points, text, f"multi_{name}"

    return None, None, None


def _save_detection_overlay(img: np.ndarray, points: np.ndarray, output_path: str) -> None:
    """Draw detected QR corners and bounding box onto image for debug."""
    debug_img = img.copy()
    pts = points.reshape(4, 2).astype(np.int32)

    # Draw bounding polygon
    cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw corner markers
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, pt in enumerate(pts):
        cv2.circle(debug_img, tuple(pt), 8, colors[i], -1)
        cv2.putText(debug_img, str(i), tuple(pt + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    cv2.imwrite(output_path, debug_img)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order the four detected corner points consistently:
    Top-left, Top-right, Bottom-right, Bottom-left.
    """
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]       # top-left (smallest sum)
    ordered[2] = pts[np.argmax(s)]        # bottom-right (largest sum)
    ordered[1] = pts[np.argmin(diff)]    # top-right (smallest diff)
    ordered[3] = pts[np.argmax(diff)]    # bottom-left (largest diff)

    return ordered


def compute_qr_region_size(dst_points: np.ndarray) -> tuple:
    """
    Compute the width and height of the QR region from ordered corner points.
    Returns (width, height) in pixels.
    """
    (tl, tr, br, bl) = dst_points

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = int(max(width_a, width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = int(max(height_a, height_b))

    # QR codes are square — use the larger dimension
    size = max(max_width, max_height)
    return size, size


def generate_qr_image(
    payload: str,
    target_size: Optional[int] = None,
    quiet_zone: int = 4,
    fg_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """
    Generate a QR code image. The warpPerspective call handles all scaling
    to fit the detected region, so we always generate at auto size with
    box_size >= 10 for reliable scanning.

    Args:
        payload: The string to encode
        target_size: Ignored (kept for API compatibility — warp handles sizing)
        quiet_zone: Border size in QR modules
        fg_color: RGB foreground color
        bg_color: RGB background color
    """
    payload_len = len(payload)
    version = 1
    for v in range(1, 40):
        capacity = _qr_version_capacity(v)
        if payload_len <= capacity:
            version = v
            break
        version = v

    box_size = 10  # minimum for reliable scanning; warp handles final size

    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=quiet_zone,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fg_color, back_color=bg_color).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _qr_version_capacity(version: int) -> int:
    """
    Approximate numeric data capacity per QR version (bytes, M-level ECC).
    """
    # Rough estimates for M-level ECC
    caps = {
        1: 25, 2: 47, 3: 77, 4: 114, 5: 154, 6: 195,
        7: 224, 8: 279, 9: 335, 10: 395, 11: 468, 12: 535,
        13: 619, 14: 667, 15: 758, 16: 854, 17: 938, 18: 1053,
        19: 1159, 20: 1264, 21: 1373, 22: 1455, 23: 1541, 24: 1633,
        25: 1725, 26: 1812, 27: 1914, 28: 1992, 29: 2102, 30: 2214,
    }
    return caps.get(version, 2000)


def perspective_warp(src_img: np.ndarray, dst_points: np.ndarray, output_size: tuple) -> np.ndarray:
    """
    Warp src_img onto the quadrilateral defined by dst_points (TL, TR, BR, BL).
    Returns warped image at output_size dimensions.
    """
    src_pts = np.array([
        [0, 0],
        [src_img.shape[1] - 1, 0],
        [src_img.shape[1] - 1, src_img.shape[0] - 1],
        [0, src_img.shape[0] - 1]
    ], dtype=np.float32)

    dst_pts = np.array([dst_points[0], dst_points[1], dst_points[2], dst_points[3]], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(src_img, H, (output_size[1], output_size[0]))
    return warped


def composite_qr(
    output_img: np.ndarray,
    warped_qr: np.ndarray,
    dst_points: np.ndarray,
    feather: bool = False,
    feather_px: int = 5
) -> np.ndarray:
    """
    Composite the warped QR onto the original using a polygon mask.

    Args:
        output_img: Original image
        warped_qr: Warped replacement QR
        dst_points: Corner points of QR region (4x2)
        feather: Enable feathered (soft) blending
        feather_px: Feather radius in pixels
    """
    dst_int = dst_points.astype(np.int32)

    # Build mask
    mask = np.zeros(output_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [dst_int], 255)

    if feather:
        # Soft edge via Gaussian blur on mask
        blur_ks = feather_px * 6 + 1
        mask_blurred = cv2.GaussianBlur(mask, (blur_ks, blur_ks), feather_px)
        mask_f = mask_blurred.astype(np.float32) / 255.0

        # Fade outer edge to 0 over feather_px pixels
        # Create a slightly shrunken mask for the inner region
        mask_inner = cv2.erode(mask, np.ones((feather_px * 2 + 1, feather_px * 2 + 1), np.uint8))
        mask_inner_f = mask_inner.astype(np.float32) / 255.0

        result = output_img.copy().astype(np.float32)
        for c in range(3):
            # Blend: background * (1 - mask_f) + warped * mask_f
            result[:, :, c] = (
                output_img[:, :, c].astype(np.float32) * (1 - mask_f) +
                warped_qr[:, :, c].astype(np.float32) * mask_f
            )
        return result.astype(np.uint8)
    else:
        # Hard mask composite
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(output_img, output_img, mask=inv_mask)
        fg = cv2.bitwise_and(warped_qr, warped_qr, mask=mask)
        return cv2.add(bg, fg)


def _final_decode(result_img: np.ndarray) -> Optional[str]:
    """
    Re-validate output by decoding the QR.
    Tries detect+decode first, then multi, then raw detect + decode.
    """
    detector = cv2.QRCodeDetector()

    # Try detect + decode (separate calls)
    try:
        retval, points = detector.detect(result_img)
        if retval and points is not None:
            text, _ = detector.decode(result_img, points)
            if text:
                return text
    except Exception:
        pass

    # Try multi
    try:
        retval, decoded_list, points, _ = detector.detectAndDecodeMulti(result_img)
        if retval and decoded_list:
            text = decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
            return text
    except Exception:
        pass

    return None


def replace_qr_pipeline(
    input_image_path: str,
    new_payload: str,
    output_path: str,
    debug_dir: Optional[str] = None,
    feather: bool = False,
    replace_all: bool = False,
    qr_fg: tuple = (0, 0, 0),
    qr_bg: tuple = (255, 255, 255),
) -> QRReplacementResult:
    """
    Full pipeline:
    1. Load image
    2. Detect existing QR (fallback chain)
    3. Extract old decoded text
    4. Generate new QR sized to fill the detected region
    5. Perspective warp to detected quadrilateral
    6. Composite onto original
    7. Re-validate output by decoding
    8. Return structured result

    Args:
        input_image_path: Path to input image
        new_payload: New string to encode in the QR
        output_path: Where to save the result
        debug_dir: If set, saves debug images at each stage
        feather: Enable soft edge blending
        replace_all: Replace all detected QR codes (not just first)
        qr_fg: RGB foreground color for new QR
        qr_bg: RGB background color for new QR

    Returns:
        QRReplacementResult dataclass
    """
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {input_image_path}")

    original_h, original_w = img.shape[:2]

    # --- Detect ---
    points, old_decoded, method_used = detect_qr_chain(img, debug_dir=debug_dir)

    if points is None:
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_no_detection.png"), img)
        return QRReplacementResult(
            output_image_path=output_path,
            detected_points=None,
            old_decoded_text=old_decoded,
            new_decoded_text=None,
            success=False,
            method_used=None,
        )

    # --- Handle multi-QR ---
    multi_mode = isinstance(points, np.ndarray) and len(points.shape) == 3 and points.shape[1] > 1

    if multi_mode and not replace_all:
        points = points[0]

    pts_ordered = order_points(np.array(points))

    # --- Size new QR to match detected region ---
    target_size, _ = compute_qr_region_size(pts_ordered)

    if debug_dir:
        # Save detected region overlay
        dbg = img.copy()
        cv2.polylines(dbg, [pts_ordered.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(debug_dir, "03_detected_region.png"), dbg)

    # --- Generate new QR ---
    new_qr = generate_qr_image(
        new_payload,
        target_size=target_size,
        quiet_zone=4,
        fg_color=qr_fg,
        bg_color=qr_bg
    )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "04_new_qr_raw.png"), new_qr)

    # --- Warp ---
    warped = perspective_warp(new_qr, pts_ordered, (original_h, original_w))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "05_warped_qr.png"), warped)

    # --- Composite ---
    result_img = composite_qr(img, warped, pts_ordered, feather=feather)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "06_composited.png"), result_img)

    # --- Re-validate ---
    final_text = _final_decode(result_img)

    cv2.imwrite(output_path, result_img)

    success = (final_text == new_payload)

    return QRReplacementResult(
        output_image_path=output_path,
        detected_points=pts_ordered,
        old_decoded_text=old_decoded,
        new_decoded_text=final_text,
        success=success,
        method_used=method_used,
    )
