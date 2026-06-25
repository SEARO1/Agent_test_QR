"""
QR Code Replacer
================
A deterministic pipeline for detecting and replacing QR codes in images.

Usage:
    from qr_replacer import replace_qr_pipeline, QRReplacementResult

    result = replace_qr_pipeline(
        input_image_path="input.png",
        new_payload="https://example.com",
        output_path="output.png",
        debug_dir="debug_output/",   # optional: save debug images
        feather=True,                # optional: soft edge blending
        qr_fg=(0, 0, 0),            # optional: QR foreground color
        qr_bg=(255, 255, 255),       # optional: QR background color
    )

    print(f"Success: {result.success}")
"""

from .qr_replacer import (
    QRReplacementResult,
    replace_qr_pipeline,
    detect_qr_chain,
    generate_qr_image,
    order_points,
    perspective_warp,
    composite_qr,
    compute_qr_region_size,
)

__version__ = "0.2.0"
__all__ = [
    "QRReplacementResult",
    "replace_qr_pipeline",
    "detect_qr_chain",
    "generate_qr_image",
    "order_points",
    "perspective_warp",
    "composite_qr",
    "compute_qr_region_size",
]
