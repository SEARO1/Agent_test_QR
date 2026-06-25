"""
QR Code Replacer Pipeline
=========================
Deterministic pipeline for detecting and replacing QR codes in images.
"""

import cv2
import numpy as np
from PIL import Image
import qrcode
from dataclasses import dataclass
from typing import Optional


@dataclass
class QRReplacementResult:
    output_image_path: str
    detected_points: Optional[np.ndarray]
    old_decoded_text: Optional[str]
    new_decoded_text: Optional[str]
    success: bool


def _normalize_detect_result(retval, decoded_info, points) -> tuple:
    """Normalize detection result - decoded_info may be string or list depending on method."""
    if points is None or (isinstance(points, np.ndarray) and points.size == 0):
        return None, None
    # decoded_info is string for detectAndDecode, list for detectAndDecodeMulti
    text = None
    if decoded_info is not None:
        if isinstance(decoded_info, (list, tuple)):
            text = decoded_info[0] if len(decoded_info) > 0 else None
        else:
            text = decoded_info
    return points, text


def detect_qr_chain(img: np.ndarray, debug_prefix: str = "debug") -> tuple:
    """
    Detection fallback chain using separate detect() + decode() calls for robustness.
    Falls back through multiple image preprocessing methods.

    Returns: (points, decoded_text, method_used)
    """
    detector = cv2.QRCodeDetector()

    # Preprocess images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_gray = cv2.bitwise_not(gray)

    # Try detect + decode (separate calls) on each preprocessing variant
    for proc_img, name in [(img, "bgr"), (gray, "gray"), (thresh, "otsu"), (inv_gray, "inv")]:
        try:
            retval, points = detector.detect(proc_img)
            if retval and points is not None and len(points) > 0:
                decoded_text, _ = detector.decode(proc_img, points)
                return points, decoded_text, name
        except Exception:
            pass

    # Try detectAndDecodeMulti as final fallback
    for proc_img, name in [(img, "bgr"), (gray, "gray"), (thresh, "otsu"), (inv_gray, "inv")]:
        try:
            # detectAndDecodeMulti returns 4 values: retval, decoded_list, points, straight_qr
            retval, decoded_list, points, _ = detector.detectAndDecodeMulti(proc_img)
            if retval and points is not None and len(points) > 0:
                decoded_text = decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
                return points, decoded_text, f"multi_{name}"
        except Exception:
            pass

    return None, None, []


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order the four detected corner points consistently:
    Top-left, Top-right, Bottom-right, Bottom-left.
    """
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum and difference to identify corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]       # top-left (smallest sum)
    ordered[2] = pts[np.argmax(s)]       # bottom-right (largest sum)
    ordered[1] = pts[np.argmin(diff)]    # top-right (smallest diff)
    ordered[3] = pts[np.argmax(diff)]    # bottom-left (largest diff)

    return ordered


def generate_qr_image(payload: str, quiet_zone: int = 4) -> np.ndarray:
    """
    Generate a clean black-on-white QR code with enough quiet zone.
    Returns BGR image.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=quiet_zone,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def perspective_warp(src_img: np.ndarray, dst_points: np.ndarray, output_size: tuple) -> np.ndarray:
    """
    Warp src_img onto the quadrilateral defined by dst_points.
    dst_points should be ordered: TL, TR, BR, BL.
    Returns the warped image (same size as output_size).
    """
    (tl, tr, br, bl) = dst_points

    # Source points for the new QR (top-left, top-right, bottom-right, bottom-left)
    src_pts = np.array([
        [0, 0],
        [src_img.shape[1] - 1, 0],
        [src_img.shape[1] - 1, src_img.shape[0] - 1],
        [0, src_img.shape[0] - 1]
    ], dtype=np.float32)

    # Destination points
    dst_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Warp the new QR
    warped = cv2.warpPerspective(src_img, H, (output_size[1], output_size[0]))

    return warped


def feather_blend(warped: np.ndarray, dst_points: np.ndarray, output_shape: tuple, feather_amount: int = 5) -> np.ndarray:
    """
    Apply feather blending at the edges of the warped QR region.
    """
    mask = np.zeros(output_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [dst_points.astype(np.int32)], 255)

    # Feather by blurring the mask
    kernel = feather_amount * 2 + 1
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mask = mask.astype(np.float32) / 255.0

    # Apply mask to warped image
    blended = warped.copy().astype(np.float32)
    for c in range(3):
        blended[:, :, c] = blended[:, :, c] * mask

    return blended.astype(np.uint8)


def composite_qr(output_img: np.ndarray, warped_qr: np.ndarray, dst_points: np.ndarray, feather: bool = False) -> np.ndarray:
    """
    Composite the warped QR back onto the original image using proper masking.
    """
    # Create binary mask of the QR region
    mask = np.zeros(output_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [dst_points.astype(np.int32)], 255)

    if feather:
        # Feather the mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        mask = mask.astype(np.float32) / 255.0
        # Soft blend
        result = output_img.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (output_img[:, :, c].astype(np.float32) * (1 - mask)) + \
                              (warped_qr[:, :, c].astype(np.float32) * mask)
        return result.astype(np.uint8)
    else:
        # Hard mask composite
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(output_img, output_img, mask=inv_mask)
        fg = cv2.bitwise_and(warped_qr, warped_qr, mask=mask)
        return cv2.add(bg, fg)


def validate_qr_at_points(img: np.ndarray, points: np.ndarray) -> Optional[str]:
    """
    Try to decode QR code at the specific point region.
    """
    detector = cv2.QRCodeDetector()
    try:
        retval, decoded_list, _, _ = detector.detectAndDecodeMulti(img)
        if retval and decoded_list:
            return decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
    except Exception:
        pass
    return None


def _normalize_decoded_text(decoded) -> Optional[str]:
    """Normalize decoded text from various formats to a string."""
    if decoded is None:
        return None
    if isinstance(decoded, (list, tuple)):
        return decoded[0] if len(decoded) > 0 else None
    return str(decoded) if decoded else None


def replace_qr_pipeline(
    input_image_path: str,
    new_payload: str,
    output_path: str,
    debug_dir: Optional[str] = None,
    feather: bool = False,
    replace_all: bool = False
) -> QRReplacementResult:
    """
    Main pipeline:
    1. Load image.
    2. Detect existing QR code using fallback chain.
    3. Store old decoded text.
    4. Generate new QR.
    5. Warp onto detected quadrilateral.
    6. Composite back.
    7. Re-validate output.
    8. Return result.
    """

    # Load image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {input_image_path}")

    # Step 1: Detect QR
    points, old_decoded_raw, method_used = detect_qr_chain(img)
    old_decoded = _normalize_decoded_text(old_decoded_raw)

    if points is None or (isinstance(points, np.ndarray) and points.size == 0):
        return QRReplacementResult(
            output_image_path=output_path,
            detected_points=None,
            old_decoded_text=old_decoded,
            new_decoded_text=None,
            success=False
        )

    # Handle multi-QR case
    if isinstance(points, np.ndarray) and len(points.shape) == 3 and points.shape[1] > 1:
        # Multiple QR codes detected
        if not replace_all:
            # Only replace the first one
            points = points[0]
        else:
            # Process all
            for i, pts in enumerate(points):
                pts_ordered = order_points(pts)
                new_qr = generate_qr_image(new_payload)
                output_shape = (img.shape[0], img.shape[1])
                warped = perspective_warp(new_qr, pts_ordered, output_shape)
                img = composite_qr(img, warped, pts_ordered, feather=feather)

            # Validate final output using detectAndDecodeMulti
            detector = cv2.QRCodeDetector()
            try:
                retval, decoded_list, _, _ = detector.detectAndDecodeMulti(img)
                final_text = decoded_list[0] if retval and decoded_list else None
            except Exception:
                final_text = None
            cv2.imwrite(output_path, img)
            final_text = _normalize_decoded_text(final_text)
            return QRReplacementResult(
                output_image_path=output_path,
                detected_points=points,
                old_decoded_text=old_decoded,
                new_decoded_text=final_text,
                success=(final_text == new_payload)
            )

    # Step 2: Order points
    pts_ordered = order_points(np.array(points))

    # Step 3: Generate new QR
    new_qr = generate_qr_image(new_payload)

    # Step 4: Warp new QR onto detected region
    output_shape = (img.shape[0], img.shape[1])
    warped = perspective_warp(new_qr, pts_ordered, output_shape)

    # Step 5: Composite back
    result_img = composite_qr(img, warped, pts_ordered, feather=feather)

    # Step 6: Re-validate output using detectAndDecodeMulti (more reliable)
    detector = cv2.QRCodeDetector()
    final_text = None
    try:
        retval, decoded_list, _, _ = detector.detectAndDecodeMulti(result_img)
        if retval and decoded_list:
            final_text = decoded_list[0] if isinstance(decoded_list, (list, tuple)) else decoded_list
    except Exception:
        pass

    final_text = _normalize_decoded_text(final_text)

    # Write output
    cv2.imwrite(output_path, result_img)

    # Determine success
    success = (final_text == new_payload)

    return QRReplacementResult(
        output_image_path=output_path,
        detected_points=pts_ordered,
        old_decoded_text=old_decoded,
        new_decoded_text=final_text,
        success=success
    )
