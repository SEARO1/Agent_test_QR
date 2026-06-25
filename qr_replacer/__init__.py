"""
QR Code Replacer
================
A deterministic pipeline for detecting and replacing QR codes in images.

Usage:
    from qr_replacer import replace_qr_pipeline, QRReplacementResult

    result = replace_qr_pipeline(
        input_image_path="input.png",
        new_payload="https://example.com",
        output_path="output.png"
    )
"""

from .qr_replacer import (
    QRReplacementResult,
    replace_qr_pipeline,
    detect_qr_chain,
    generate_qr_image,
    order_points,
    perspective_warp,
    composite_qr,
    validate_qr_at_points,
)

__version__ = "0.1.0"
__all__ = [
    "QRReplacementResult",
    "replace_qr_pipeline",
    "detect_qr_chain",
    "generate_qr_image",
    "order_points",
    "perspective_warp",
    "composite_qr",
    "validate_qr_at_points",
]
