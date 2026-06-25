# QR Code Replacer v0.4

A deterministic Python pipeline for detecting and replacing QR codes inside images, with smart carrier detection and multi-candidate scoring.

## Features

- **Smart carrier detection** — scans outward from QR to find paper boundaries, avoids covering text/graphics
- **Multi-candidate scoring** — generates tight/smart/conservative candidates, auto-selects best
- **Robust detection** — 5-stage fallback chain using separate `detect()` + `decode()` calls
- **Perspective-correct warp** — preserves detected position, scale, and angle via homography
- **Re-validation** — decodes output to confirm replacement succeeded
- **Feather blending** — Gaussian-blurred soft-edge blending at carrier boundary
- **Content protection** — penalizes carrier expansions that cover nearby text/graphics
- **Color QR support** — specify foreground/background colors
- **Debug output** — saves candidate comparisons and full pipeline report
- **Texture simulation** — Gaussian noise + blur + JPEG to match paper appearance

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
├── 03_detected_region.png       # Detection overlay on original
├── 04_candidate_tight.png       # Tight expansion overlay
├── 04_candidate_smart.png       # Smart expansion overlay
├── 04_candidate_conservative.png # Conservative expansion overlay
├── 05_tight_blended.png        # Tight candidate result
├── 05_smart_blended.png        # Smart candidate result
├── 05_conservative_blended.png  # Conservative candidate result
├── 10_final.png                # Best candidate (selected by scoring)
└── pipeline_report.json         # Full debug report with scores
```

### Scoring

Candidates are scored: `decode_success*5 + brightness_match + content_protection*2 - size_penalty`

## Requirements

- Python 3.8+
- opencv-python-headless >= 4.8
- pillow >= 10.0
- qrcode >= 7.4
- numpy >= 1.24

## License

MIT
