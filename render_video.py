"""
render_video.py - Generate CPPN video with smooth spline interpolation

This script generates smooth videos by interpolating through latent space
using spline interpolation for seamless transitions between keyframes.

Usage:
    python render_video.py --n_frames 120 --output video.mp4
    python render_video.py --n_frames 240 --seed 42 --net_size 32
    python render_video.py --n_frames 180 --color_reference ref.jpg --color_match_strength 0.8
"""

import argparse
import numpy as np
import subprocess
import shutil
from pathlib import Path
from PIL import Image
from scipy.interpolate import CubicSpline
from cppn import CPPNGenerator


def generate_keypoints(h_size, num_points, seed=None):
    """
    Generate random keypoints in latent space for spline interpolation.

    Args:
        h_size: Dimension of the latent space
        num_points: Number of keypoints to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (num_points, h_size) containing latent vectors
    """
    rng = np.random.RandomState(seed)

    # Generate random points in latent space
    points = rng.uniform(-1.0, 1.0, size=(num_points, h_size)).astype(np.float32)

    return points


def generate_spline_trajectory(h_size, n_frames, num_keypoints=4, seed=None):
    """
    Generate a smooth trajectory through latent space using cubic spline interpolation.

    This creates a smooth path that passes through the keypoints without abrupt
    direction changes. The trajectory remains the same regardless of n_frames,
    just sampled more densely.

    Args:
        h_size: Dimension of latent space
        n_frames: Total number of frames to generate
        num_keypoints: Number of keypoints to interpolate through (default: 4)
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_frames, h_size) containing latent vectors for each frame
    """
    # Generate keypoints
    keypoints = generate_keypoints(h_size, num_keypoints, seed)

    # Create parameter values for keypoints (evenly spaced from 0 to 1)
    t_keypoints = np.linspace(0, 1, num_keypoints)

    # Create parameter values for all frames (densely sampled from 0 to 1)
    t_frames = np.linspace(0, 1, n_frames)

    # Create cubic spline interpolation for each dimension of the latent space
    # We interpolate each dimension independently to create a smooth path
    trajectory = np.zeros((n_frames, h_size), dtype=np.float32)

    for dim in range(h_size):
        # Get values for this dimension across all keypoints
        keypoint_values = keypoints[:, dim]

        # Create cubic spline for this dimension
        # bc_type='natural' gives smooth second derivatives at boundaries
        spline = CubicSpline(t_keypoints, keypoint_values, bc_type='natural')

        # Evaluate spline at all frame positions
        trajectory[:, dim] = spline(t_frames)

    return trajectory


def render_video(
    n_frames,
    output_path="cppn_video.mp4",
    width=1920,
    height=1080,
    fps=25,
    # CPPN parameters
    net_size=16,
    h_size=32,
    scaling=2.5,
    num_layers=3,
    seed=None,
    rgb=True,
    invert=False,
    # Color matching parameters
    color_reference=None,
    color_match_strength=1.0,
    color_match_mode='mean_std',
    # Spline parameters
    num_keypoints=4,
    # Memory optimization
    keep_frames_dir=False,
    jpeg_quality=95
):
    """
    Generate a CPPN video with smooth spline interpolation through keyframes.

    Frames are saved to disk as JPEGs to avoid memory issues with long videos,
    then combined using ffmpeg.

    Args:
        n_frames: Number of frames to generate
        output_path: Path to save the video file
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second (default: 25)
        net_size: Size of CPPN hidden layers
        h_size: Size of latent vector
        scaling: Coordinate scaling factor
        num_layers: Number of CPPN hidden layers
        seed: Random seed for reproducibility
        rgb: Generate RGB (True) or grayscale (False)
        invert: Invert colors
        color_reference: Path to reference image for color matching
        color_match_strength: Strength of color matching (0.0 to 1.0)
        color_match_mode: Mode for color matching ('mean_std' or 'percentile')
        num_keypoints: Number of keypoints for the spline trajectory
        keep_frames_dir: If True, keep the frames directory after video creation
        jpeg_quality: JPEG quality for frame storage (1-100, default: 95)

    Returns:
        Path to the generated video file
    """
    print("=" * 70)
    print("CPPN Video Generator")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Frames: {n_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Net size: {net_size}")
    print(f"  Latent size: {h_size}")
    print(f"  Scaling: {scaling}")
    print(f"  Layers: {num_layers}")
    print(f"  Seed: {seed}")
    print(f"  Color reference: {color_reference}")
    if color_reference:
        print(f"  Color match strength: {color_match_strength}")
        print(f"  Color match mode: {color_match_mode}")
    print(f"  Keypoints: {num_keypoints}")
    print(f"  JPEG quality: {jpeg_quality}")
    print()

    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg not found in PATH")
        print("Please install ffmpeg to generate videos")
        print("  macOS: brew install ffmpeg")
        print("  Linux: apt-get install ffmpeg or yum install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return None

    # Create temporary frames directory
    output_path_obj = Path(output_path)
    frames_dir = output_path_obj.parent / f"{output_path_obj.stem}_frames_temp"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary frames directory: {frames_dir}")

    # Create CPPN generator
    generator = CPPNGenerator(
        net_size=net_size,
        h_size=h_size,
        rgb=rgb,
        scaling=scaling,
        seed=seed,
        color_reference=color_reference,
        color_match_strength=color_match_strength,
        color_match_mode=color_match_mode
    )

    # Generate latent trajectory using spline interpolation
    print(f"Generating smooth spline trajectory with {num_keypoints} keypoints...")
    z_trajectory = generate_spline_trajectory(h_size, n_frames, num_keypoints, seed)

    print(f"Generated trajectory with {len(z_trajectory)} frames")

    # Generate and save frames to disk
    print("\nGenerating and saving frames to disk...")
    for i, z in enumerate(z_trajectory):
        progress = (i + 1) / len(z_trajectory) * 100
        print(f"  Frame {i+1}/{len(z_trajectory)} ({progress:.1f}%)...", end='\r')

        # Generate frame
        frame = generator.generate(
            width=width,
            height=height,
            z=z,
            num_layers=num_layers
        )

        # Process and save frame as JPEG
        img_data = 1 - frame if invert else frame
        img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)

        # Save as JPEG
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        img = Image.fromarray(img_data)
        img.save(frame_path, 'JPEG', quality=jpeg_quality)

    print(f"\n✓ Generated and saved {len(z_trajectory)} frames to disk")

    # Use ffmpeg to create video
    print(f"\nCreating video with ffmpeg (fps={fps})...")

    # Construct ffmpeg command
    # Using high quality settings for H.264
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%06d.jpg'),
        '-c:v', 'libx264',  # H.264 codec
        '-preset', 'slow',  # Better quality
        '-crf', '18',  # High quality (lower = better, 18 is visually lossless)
        '-pix_fmt', 'yuv420p',  # Compatibility with most players
        str(output_path)
    ]

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Video saved to {output_path}")

        # Print video info
        duration = len(z_trajectory) / fps
        print(f"\nVideo info:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total frames: {len(z_trajectory)}")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")

        # Check file size
        video_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {video_size_mb:.2f} MB")

    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print(f"stderr: {e.stderr}")
        print(f"\nFrames are saved in: {frames_dir}")
        return None

    # Clean up frames directory unless requested to keep
    if not keep_frames_dir:
        print(f"\nCleaning up temporary frames directory...")
        shutil.rmtree(frames_dir)
        print(f"✓ Removed {frames_dir}")
    else:
        print(f"\nFrames directory kept at: {frames_dir}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPPN video with smooth spline interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Output parameters
    parser.add_argument('--n_frames', type=int, default=120,
                        help='Number of frames to generate')
    parser.add_argument('--output', type=str, default='cppn_video.mp4',
                        help='Output video file path')
    parser.add_argument('--width', type=int, default=1920//3,
                        help='Video width in pixels')
    parser.add_argument('--height', type=int, default=1080//3,
                        help='Video height in pixels')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second (default: 25)')
    parser.add_argument('--keep_frames', action='store_true',
                        help='Keep the frames directory after video creation')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help='JPEG quality for frame storage (1-100, default: 95)')

    # Spline parameters
    parser.add_argument('--num_keypoints', type=int, default=5,
                        help='Number of keypoints for spline trajectory')

    # CPPN parameters
    parser.add_argument('--net_size', type=int, default=16,
                        help='Size of CPPN hidden layers')
    parser.add_argument('--h_size', type=int, default=24,
                        help='Size of latent vector')
    parser.add_argument('--scaling', type=float, default=2.5,
                        help='Coordinate scaling factor')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of CPPN hidden layers')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--grayscale', action='store_true',
                        help='Generate grayscale instead of RGB')
    parser.add_argument('--invert', action='store_true',
                        help='Invert colors (black becomes white, white becomes black)')

    # Color matching parameters
    parser.add_argument('--color_reference', type=str, default=None,
                        help='Path to reference image for color matching')
    parser.add_argument('--color_match_strength', type=float, default=1.0,
                        help='Strength of color matching (0.0 to 1.0)')
    parser.add_argument('--color_match_mode', type=str, default='mean_std',
                        choices=['mean_std', 'percentile'],
                        help='Mode for color matching')

    args = parser.parse_args()

    # Validate arguments
    if args.color_reference and not Path(args.color_reference).exists():
        print(f"Error: Color reference image not found: {args.color_reference}")
        return

    if args.color_match_strength < 0.0 or args.color_match_strength > 1.0:
        print("Error: color_match_strength must be between 0.0 and 1.0")
        return

    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        print("Error: jpeg_quality must be between 1 and 100")
        return

    # Generate video
    render_video(
        n_frames=args.n_frames,
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        net_size=args.net_size,
        h_size=args.h_size,
        scaling=args.scaling,
        num_layers=args.num_layers,
        seed=args.seed,
        rgb=not args.grayscale,
        invert=args.invert,
        color_reference=args.color_reference,
        color_match_strength=args.color_match_strength,
        color_match_mode=args.color_match_mode,
        num_keypoints=args.num_keypoints,
        keep_frames_dir=args.keep_frames,
        jpeg_quality=args.jpeg_quality
    )


if __name__ == '__main__':
    main()
