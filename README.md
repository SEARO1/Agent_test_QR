# QR Code Replacer

A deterministic Python pipeline for detecting and replacing QR codes inside images, with full re-validation.

## Features

- **Robust detection** — 5-stage fallback chain using separate `detect()` + `decode()` calls (handles ECI-encoded QR codes that `detectAndDecode` fails on)
- **Perspective-correct warp** — preserves detected position, scale, and angle via homography
- **Re-validation** — decodes output to confirm replacement succeeded
- **Feather blending** — optional soft-edge blending at QR boundary for natural composites
- **Color QR support** — specify foreground/background colors
- **Debug output** — saves intermediate images for each detection/preprocessing stage
- **Multi-QR support** — replace all detected codes with `--replace-all`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from qr_replacer import replace_qr_pipeline, QRReplacementResult

result = replace_qr_pipeline(
    input_image_path="input.png",
    new_payload="https://example.com",
    output_path="output.png",
    debug_dir="debug/",     # optional: save debug images
    feather=True,           # optional: soft edge blending
    qr_fg=(0, 0, 0),        # optional: foreground RGB
    qr_bg=(255, 255, 255),  # optional: background RGB
)

print(f"Old: {result.old_decoded_text}")
print(f"New: {result.new_decoded_text}")
print(f"Success: {result.success}")
```

### CLI

```bash
# Basic replacement
python -m qr_replacer -i input.png -p "Hello World" -o output.png

# With feather blending
python -m qr_replacer -i input.png -p "Hello World" -o output.png -f

# With debug images
python -m qr_replacer -i input.png -p "Hello World" -o output.png -d debug/

# Colored QR
python -m qr_replacer -i input.png -p "Hello" -o output.png --fg "0,0,128" --bg "255,255,224"

# Replace all QR codes
python -m qr_replacer -i input.png -p "Hello World" -o output.png -a
```

## Detection Fallback Chain

If detection fails on the original image, it automatically tries:

1. Grayscale
2. Otsu threshold
3. Inverted grayscale
4. CLAHE-enhanced grayscale
5. `detectAndDecodeMulti` on each of the above

Uses **separate `detect()` + `decode()` calls** instead of `detectAndDecode` for better ECI encoding support.

## Result Object

| Field | Type | Description |
|-------|------|-------------|
| `output_image_path` | `str` | Path to saved output image |
| `detected_points` | `np.ndarray` | 4 corner points (ordered TL→TR→BR→BL) |
| `old_decoded_text` | `str` | Original QR content |
| `new_decoded_text` | `str` | Decoded content of replaced QR |
| `success` | `bool` | `True` only if `new_decoded_text == new_payload` |
| `method_used` | `str` | Detection method that succeeded |

## Debug Output (`-d`)

When `debug_dir` is set, saves:

```
debug/
├── 01_preproc_bgr.png        # Original BGR
├── 01_preproc_gray.png       # Grayscale
├── 01_preproc_otsu.png       # Otsu threshold
├── 01_preproc_inv_gray.png   # Inverted grayscale
├── 01_preproc_clahe.png      # CLAHE enhanced
├── 02_detected_bgr.png       # Detection overlay (first success)
├── 03_detected_region.png    # Ordered corner overlay
├── 04_new_qr_raw.png         # Generated replacement QR
├── 05_warped_qr.png          # Warped QR (before composite)
└── 06_composited.png         # Final composited result
```

## Requirements

- Python 3.8+
- opencv-python-headless >= 4.8
- pillow >= 10.0
- qrcode >= 7.4
- numpy >= 1.24

## License

MIT
