"""
CLI entry point for qr_replacer package.
Run with: python -m qr_replacer -i input.png -p "payload" -o output.png
"""

import argparse
from .qr_replacer import replace_qr_pipeline

parser = argparse.ArgumentParser(description="QR Code Replacement Pipeline")
parser.add_argument("-i", "--input", required=True, help="Input image path")
parser.add_argument("-p", "--payload", required=True, help="New QR payload string")
parser.add_argument("-o", "--output", default="output_qr_replaced.png", help="Output image path")
parser.add_argument("-d", "--debug", help="Debug output directory")
parser.add_argument("-f", "--feather", action="store_true", help="Enable feather blending")
parser.add_argument("-a", "--replace-all", action="store_true", help="Replace all detected QR codes")

args = parser.parse_args()

result = replace_qr_pipeline(
    input_image_path=args.input,
    new_payload=args.payload,
    output_path=args.output,
    debug_dir=args.debug,
    feather=args.feather,
    replace_all=args.replace_all
)

print(f"Output: {result.output_image_path}")
print(f"Detected points:\n{result.detected_points}")
print(f"Old decoded: {result.old_decoded_text}")
print(f"New decoded: {result.new_decoded_text}")
print(f"Success: {result.success}")
