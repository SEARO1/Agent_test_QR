"""
CLI entry point for qr_replacer package.

Usage:
    python -m qr_replacer -i input.png -p "payload" -o output.png
    python -m qr_replacer -i input.png -p "payload" -o output.png -f     # feather blending (default)
    python -m qr_replacer -i input.png -p "payload" -o output.png --seamless  # seamless blending
    python -m qr_replacer -i input.png -p "payload" -o output.png -d debug/   # save debug images
    python -m qr_replacer -i input.png -p "payload" -o output.png -a          # replace all QR codes
"""

import argparse
import json
from .qr_replacer import replace_qr_pipeline

parser = argparse.ArgumentParser(description="QR Code Replacement Pipeline v0.3.0")
parser.add_argument("-i", "--input", required=True, help="Input image path")
parser.add_argument("-p", "--payload", required=True, help="New QR payload string")
parser.add_argument("-o", "--output", default="output_qr_replaced.png", help="Output image path")
parser.add_argument("-d", "--debug", help="Debug output directory")
parser.add_argument("-f", "--feather", action="store_true", default=True, help="Use feather blending (default)")
parser.add_argument("--seamless", action="store_true", help="Use seamlessClone blending")
parser.add_argument("-a", "--replace-all", action="store_true", help="Replace all detected QR codes")
parser.add_argument("--expand", type=float, default=1.45, help="Carrier patch expansion ratio (default: 1.45)")
parser.add_argument("--qr-scale", type=float, default=0.72, help="QR scale within carrier (default: 0.72, unused in v0.3)")
parser.add_argument("--feather-px", type=int, default=3, help="Edge feather radius in pixels (default: 3)")
parser.add_argument("--blur", type=float, default=0.0, help="Blur sigma (0=auto, default: auto)")
parser.add_argument("--no-texture", action="store_true", help="Disable texture simulation")
parser.add_argument("--no-jpeg", action="store_true", help="Disable JPEG simulation")
parser.add_argument("--fg", dest="fg", default="0,0,0", help="QR foreground color as R,G,B (default: 0,0,0)")
parser.add_argument("--bg", dest="bg", default="255,255,255", help="QR background color as R,G,B (default: 255,255,255)")


def parse_color(s):
    return tuple(int(x) for x in s.split(","))


args = parser.parse_args()

# Determine blend mode
blend_mode = "seamless" if args.seamless else "feather"

result = replace_qr_pipeline(
    input_image_path=args.input,
    new_payload=args.payload,
    output_path=args.output,
    debug_dir=args.debug,
    mode="single",
    replace_all=args.replace_all,
    blend_mode=blend_mode,
    carrier_expand_ratio=args.expand,
    qr_scale_within_patch=args.qr_scale,
    edge_feather_px=args.feather_px,
    blur_sigma=args.blur,
    jpeg_simulation=not args.no_jpeg,
    verify=True,
    qr_fg=parse_color(args.fg),
    qr_bg=parse_color(args.bg),
)

print(f"Output:        {result.output_image_path}")
print(f"Detected pts:  {result.detected_points.tolist() if result.detected_points is not None else None}")
print(f"Expanded pts:  {result.expanded_points.tolist() if result.expanded_points is not None else None}")
print(f"Method used:   {result.method_used}")
print(f"Old decoded:   {result.old_decoded_text}")
print(f"New decoded:   {result.new_decoded_text}")
print(f"Blend mode:    {result.selected_blend_mode}")
print(f"Retries used:  {result.retries_used}")
print(f"Success:       {result.success}")
if result.reason_if_failed:
    print(f"Failure:       {result.reason_if_failed}")

if args.debug and result.debug_report:
    report_path = args.debug.rstrip("/\\") + "/pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(result.debug_report, f, indent=2)
    print(f"Debug report:  {report_path}")
