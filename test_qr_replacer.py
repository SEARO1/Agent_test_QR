"""
Quick test script for qr_replacer pipeline.
"""
import cv2
import numpy as np
from qr_replacer import replace_qr_pipeline, generate_qr_image, detect_qr_chain

# Generate a test image with a QR code (simple standalone QR)
qr = generate_qr_image("https://example.com/old")
cv2.imwrite("test_qr.png", qr)
print("Generated test QR:", qr.shape)

# Run detection
pts, text, method = detect_qr_chain(qr)
print(f"Detected: pts={pts is not None}, text={text}, method={method}")

# Manually run through the pipeline to debug
from qr_replacer import order_points, perspective_warp, composite_qr

detector = cv2.QRCodeDetector()
retval, decoded_info, points, _ = detector.detectAndDecodeMulti(qr)
print(f"Multi decode: retval={retval}, decoded_info={decoded_info}, points_shape={points.shape if points is not None else None}")

# Now test the replacement
result = replace_qr_pipeline(
    input_image_path="test_qr.png",
    new_payload="https://newhref.com/replaced",
    output_path="test_output.png"
)

print(f"\n=== Result ===")
print(f"Old text: {result.old_decoded_text}")
print(f"New text: {result.new_decoded_text}")
print(f"Success: {result.success}")

# Try decoding the output manually
output_img = cv2.imread("test_output.png")
if output_img is not None:
    r2, di2, p2, _ = detector.detectAndDecodeMulti(output_img)
    print(f"Output decode attempt: retval={r2}, decoded_info={di2}")
