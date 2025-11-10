"""
generate_multiple_videos.py - Generate multiple CPPN videos with different seeds

This script runs render_video.py multiple times with different seeds to generate
a batch of videos with varying patterns but consistent parameters.

Usage:
    python generate_multiple_videos.py
    python generate_multiple_videos.py --seeds 0-50
    python generate_multiple_videos.py --seeds 10,20,30,40
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_seed_range(seed_spec):
    """
    Parse seed specification into a list of seed values.

    Args:
        seed_spec: Either a range like "0-20" or comma-separated values like "1,5,10"

    Returns:
        List of integer seed values
    """
    if '-' in seed_spec:
        # Parse range like "0-20"
        start, end = seed_spec.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in seed_spec:
        # Parse comma-separated list like "1,5,10"
        return [int(s.strip()) for s in seed_spec.split(',')]
    else:
        # Single value
        return [int(seed_spec)]


def generate_videos(
    seeds,
    n_frames=40,
    color_reference="reference_colors/082.jpg",
    color_match_strength=1.0,
    output_dir="videos",
    base_name="cppn_video",
    **extra_args
):
    """
    Generate multiple videos with different seeds.

    Args:
        seeds: List of seed values to use
        n_frames: Number of frames per video
        color_reference: Path to color reference image
        color_match_strength: Strength of color matching (0.0 to 1.0)
        output_dir: Directory to save videos
        base_name: Base name for output videos (will be suffixed with seed)
        **extra_args: Additional arguments to pass to render_video.py
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multiple CPPN Video Generator")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Seeds: {seeds}")
    print(f"  Frames per video: {n_frames}")
    print(f"  Color reference: {color_reference}")
    print(f"  Color match strength: {color_match_strength}")
    print(f"  Output directory: {output_dir}")
    print(f"  Base name: {base_name}")
    print(f"\nGenerating {len(seeds)} videos...\n")

    # Track successes and failures
    successful = []
    failed = []

    for i, seed in enumerate(seeds, 1):
        # Generate output filename with seed suffix
        output_file = output_path / f"{base_name}_seed{seed}.mp4"

        print(f"\n{'=' * 70}")
        print(f"Video {i}/{len(seeds)}: Seed {seed}")
        print(f"{'=' * 70}\n")

        # Build command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "render_video.py",
            "--n_frames", str(n_frames),
            "--color_match_strength", str(color_match_strength),
            "--color_reference", color_reference,
            "--seed", str(seed),
            "--output", str(output_file)
        ]

        # Add any extra arguments
        for key, value in extra_args.items():
            if value is not None:
                # Convert underscore to dash for CLI args
                arg_name = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:  # Only add flag if True
                        cmd.append(arg_name)
                else:
                    cmd.extend([arg_name, str(value)])

        # Run render_video.py
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True
            )
            successful.append((seed, output_file))
            print(f"\n✓ Successfully generated video for seed {seed}")

        except subprocess.CalledProcessError as e:
            failed.append((seed, str(e)))
            print(f"\n✗ Failed to generate video for seed {seed}")
            print(f"  Error: {e}")
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user")
            print(f"Generated {len(successful)} videos before interruption")
            break

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal videos requested: {len(seeds)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✓ Successfully generated videos:")
        for seed, output_file in successful:
            print(f"  Seed {seed}: {output_file}")

    if failed:
        print(f"\n✗ Failed videos:")
        for seed, error in failed:
            print(f"  Seed {seed}: {error}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple CPPN videos with different seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Seed specification
    parser.add_argument('--seeds', type=str, default='0-20',
                        help='Seeds to use: range like "0-20", list like "1,5,10", or single value')

    # Video parameters
    parser.add_argument('--n_frames', type=int, default=40,
                        help='Number of frames per video')
    parser.add_argument('--color_reference', type=str, default='reference_colors/082.jpg',
                        help='Path to reference image for color matching')
    parser.add_argument('--color_match_strength', type=float, default=1.0,
                        help='Strength of color matching (0.0 to 1.0)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='videos',
                        help='Directory to save videos')
    parser.add_argument('--base_name', type=str, default='cppn_video',
                        help='Base name for output videos (will be suffixed with _seedN)')

    # Pass-through parameters for render_video.py
    parser.add_argument('--width', type=int, default=1920//2,
                        help='Video width in pixels')
    parser.add_argument('--height', type=int, default=1080//2,
                        help='Video height in pixels')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second')
    parser.add_argument('--net_size', type=int, default=None,
                        help='Size of CPPN hidden layers')
    parser.add_argument('--h_size', type=int, default=None,
                        help='Size of latent vector')
    parser.add_argument('--scaling', type=float, default=None,
                        help='Coordinate scaling factor')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of CPPN hidden layers')
    parser.add_argument('--num_keypoints', type=int, default=None,
                        help='Number of keypoints for spline trajectory')
    parser.add_argument('--grayscale', action='store_true',
                        help='Generate grayscale instead of RGB')
    parser.add_argument('--invert', action='store_true',
                        help='Invert colors')
    parser.add_argument('--color_match_mode', type=str, default=None,
                        choices=['mean_std', 'percentile'],
                        help='Mode for color matching')
    parser.add_argument('--keep_frames', action='store_true',
                        help='Keep the frames directory after video creation')
    parser.add_argument('--jpeg_quality', type=int, default=None,
                        help='JPEG quality for frame storage (1-100)')

    args = parser.parse_args()

    # Parse seed specification
    try:
        seeds = parse_seed_range(args.seeds)
    except ValueError as e:
        print(f"Error parsing seeds: {e}")
        print("Use format like '0-20' for range or '1,5,10' for list")
        return

    # Validate color reference exists
    if not Path(args.color_reference).exists():
        print(f"Error: Color reference image not found: {args.color_reference}")
        return

    # Extract pass-through arguments
    extra_args = {
        'width': args.width,
        'height': args.height,
        'fps': args.fps,
        'net_size': args.net_size,
        'h_size': args.h_size,
        'scaling': args.scaling,
        'num_layers': args.num_layers,
        'num_keypoints': args.num_keypoints,
        'grayscale': args.grayscale,
        'invert': args.invert,
        'color_match_mode': args.color_match_mode,
        'keep_frames': args.keep_frames,
        'jpeg_quality': args.jpeg_quality,
    }

    # Generate videos
    generate_videos(
        seeds=seeds,
        n_frames=args.n_frames,
        color_reference=args.color_reference,
        color_match_strength=args.color_match_strength,
        output_dir=args.output_dir,
        base_name=args.base_name,
        **extra_args
    )


if __name__ == '__main__':
    main()
