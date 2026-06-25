"""QR Replacer Package."""

from .qr_replacer import (
    replace_qr_pipeline,
    QRReplacementResult,
    detect_qr_chain,
    generate_qr_image,
    perspective_warp,
    order_points,
    estimate_local_appearance,
    expand_carrier_quad,
    generate_carrier_patch,
    blend_feather,
    blend_seamless,
)

__all__ = [
    "replace_qr_pipeline",
    "QRReplacementResult",
    "detect_qr_chain",
    "generate_qr_image",
    "perspective_warp",
    "order_points",
    "estimate_local_appearance",
    "expand_carrier_quad",
    "generate_carrier_patch",
    "blend_feather",
    "blend_seamless",
    "compute_smart_expansion",
    "detect_carrier_region",
]
