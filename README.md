# CPPN Art Generator

Generate mesmerizing abstract art and animations using Compositional Pattern Producing Networks (CPPNs).

<p align="center">
  <img src="cppn_example.gif" alt="CPPN Animation Example" width="70%">
</p>

## What is CPPN?

CPPNs create artistic images by passing coordinate information through neural networks with various activation functions. Think of it as evolution meets neural networks meets generative art.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate a single image
python cppn.py

# Generate an animated video
python render_video.py --n_frames 120 --output my_video.mp4
```

## Features

- **Hardware Accelerated**: Automatic detection and use of Apple MPS, CUDA, or CPU
- **Batch Processing**: Optimized video generation with efficient GPU utilization
- **Color Matching**: Match generated art to reference color palettes
- **Smooth Animations**: Spline interpolation for seamless video transitions
- **High Quality Output**: 16-bit image support for smooth gradients

## Basic Usage

### Generate Images

```python
from cppn import generate_image

# Simple generation
generate_image(width=512, height=512, seed=42, output_path="art.png")

# With color matching
generate_image(
    width=512,
    height=512,
    color_reference="reference.jpg",
    color_match_strength=0.8
)
```

### Generate Videos

```bash
# Full HD video with 240 frames
python render_video.py --n_frames 240 --width 1920 --height 1080 --fps 30

# With color matching
python render_video.py --n_frames 120 --color_reference colors.jpg --color_match_strength 0.7
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `net_size` | Network complexity (higher = more detail) | 32 |
| `h_size` | Latent vector size | 32 |
| `scaling` | Zoom level (higher = more zoomed out) | 10.0 |
| `seed` | Random seed for reproducibility | None |
| `num_layers` | Number of hidden layers | 3 |

## Requirements

- Python 3.7+
- NumPy
- Pillow
- PyTorch (optional, for GPU acceleration)
- SciPy (for video generation)
- FFmpeg (for video encoding)

## Credits

Inspired by [hardmaru's CPPN implementation](https://github.com/hardmaru/cppn-tensorflow). Modernized with PyTorch and optimized for 2025.
