"""
CLI entry point for qr_replacer package.

Usage:
    python -m qr_replacer -i input.png -p "payload" -o output.png
    python -m qr_replacer -i input.png -p "payload" -o output.png -f     # feather blending
    python -m qr_replacer -i input.png -p "payload" -o output.png -d debug/   # save debug images
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
parser.add_argument("--fg", dest="fg", default="0,0,0", help="QR foreground color as R,G,B (default: 0,0,0)")
parser.add_argument("--bg", dest="bg", default="255,255,255", help="QR background color as R,G,B (default: 255,255,255)")

args = parser.parse_args()

# Parse color tuples
def parse_color(s):
    return tuple(int(x) for x in s.split(","))

result = replace_qr_pipeline(
    input_image_path=args.input,
    new_payload=args.payload,
    output_path=args.output,
    debug_dir=args.debug,
    feather=True,
    replace_all=args.replace_all,
    qr_fg=parse_color(args.fg),
    qr_bg=parse_color(args.bg),
)

print(f"Output:        {result.output_image_path}")
print(f"Detected pts:  {result.detected_points}")
print(f"Method used:   {result.method_used}")
print(f"Old decoded:   {result.old_decoded_text}")
print(f"New decoded:   {result.new_decoded_text}")
print(f"Success:       {result.success}")
