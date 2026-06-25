# QR Code Replacer

A deterministic Python pipeline for detecting and replacing QR codes inside images.

## Features

- **Automatic detection** using OpenCV `QRCodeDetector` with 5-stage fallback chain
- **Perspective-correct replacement** — preserves position, scale, and angle
- **Re-validation** — decodes the output to confirm successful replacement
- **Feather blending** — optional smooth edge blending at QR boundary
- **Multi-QR support** — replace all detected QR codes with `--replace-all`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from qr_replacer import replace_qr_pipeline, QRReplacementResult

result = replace_qr_pipeline(
    input_image_path="photo_with_qr.png",
    new_payload="https://mynewlink.com",
    output_path="result.png",
    feather=False        # set True for soft edges
)

print(f"Old content: {result.old_decoded_text}")
print(f"New content: {result.new_decoded_text}")
print(f"Success:     {result.success}")
```

### CLI

```bash
# Basic replacement
python -m qr_replacer.qr_replacer -i input.png -p "Hello World" -o output.png

# With feather blending
python -m qr_replacer.qr_replacer -i input.png -p "Hello World" -o output.png -f

# Replace all QR codes
python -m qr_replacer.qr_replacer -i input.png -p "Hello World" -o output.png -a
```

## Detection Fallback Chain

If OpenCV can't detect the QR on the original BGR image, it automatically tries:

1. Original BGR image
2. Grayscale
3. Otsu threshold
4. Inverted grayscale
5. `detectAndDecodeMulti` on processed images

## Result Object

| Field | Type | Description |
|-------|------|-------------|
| `output_image_path` | `str` | Path to the saved output image |
| `detected_points` | `np.ndarray` | 4 corner points of detected QR (ordered TL→TR→BR→BL) |
| `old_decoded_text` | `str` | Original content encoded in the QR |
| `new_decoded_text` | `str` | Content decoded from the replaced QR |
| `success` | `bool` | `True` only if `new_decoded_text == new_payload` |

## Requirements

- Python 3.8+
- opencv-python-headless
- pillow
- qrcode
- numpy

## License

MIT
