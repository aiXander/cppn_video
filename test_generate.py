"""
Generate a batch of CPPN images with random settings to explore the parameter space.

This script generates multiple images with different random settings and saves
both the images and their parameters for exploration.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from cppn import CPPNGenerator


# ============================================================================
# CONFIGURATION - Customize these settings
# ============================================================================

# Number of images to generate
N = 20

# Output settings
OUTPUT_DIR = "exploration_random"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Random seed for reproducibility (None for random)
MASTER_SEED = 0

# ============================================================================
# PARAMETER RANGES - Customize these ranges for your explorations
# ============================================================================

PARAM_RANGES = {
    # Network architecture
    "net_size": [16],
    "h_size": [16, 32, 64],
    "num_layers": [2, 3, 4, 5],

    # Image properties
    "scaling_min": 2.5,
    "scaling_max": 5.0,
    "rgb_options": [True],

    # Latent vector
    "z_seed_min": 0,
    "z_seed_max": 10,

    # Visual settings
    "invert_options": [False],

    # Color matching (set to None to disable)
    "color_reference": "reference_colors/01.jpg",  # Path to reference image, or None to disable
    "color_match_strength_min": 1.0,  # Minimum strength (0.0 = no matching)
    "color_match_strength_max": 1.0   # Maximum strength (1.0 = full matching)
}


def generate_random_settings(rng):
    """
    Generate random settings for CPPN image generation using PARAM_RANGES.

    Args:
        rng: numpy RandomState instance

    Returns:
        Dictionary of settings
    """
    settings = {
        # Network architecture
        "net_size": int(rng.choice(PARAM_RANGES["net_size"])),
        "h_size": int(rng.choice(PARAM_RANGES["h_size"])),
        "num_layers": int(rng.choice(PARAM_RANGES["num_layers"])),

        # Image properties
        "scaling": float(rng.uniform(PARAM_RANGES["scaling_min"], PARAM_RANGES["scaling_max"])),
        "rgb": bool(rng.choice(PARAM_RANGES["rgb_options"])),

        # Latent vector
        "z_seed": int(rng.randint(PARAM_RANGES["z_seed_min"], PARAM_RANGES["z_seed_max"])),

        # Visual settings
        "invert": bool(rng.choice(PARAM_RANGES["invert_options"])),

        # Color matching
        "color_reference": PARAM_RANGES["color_reference"],
        "color_match_strength": float(rng.uniform(
            PARAM_RANGES["color_match_strength_min"],
            PARAM_RANGES["color_match_strength_max"]
        )) if PARAM_RANGES["color_reference"] is not None else 0.0
    }

    return settings


def settings_to_filename(settings, index):
    """
    Create a descriptive filename from settings.

    Args:
        settings: Dictionary of settings
        index: Image index

    Returns:
        Filename string
    """
    rgb_str = "rgb" if settings["rgb"] else "gray"
    inv_str = "inv" if settings["invert"] else "norm"

    filename = (
        f"cppn_"
        f"n{settings['net_size']}_"
        f"s{settings['scaling']:.1f}_"
        f"h{settings['h_size']}_"
        f"l{settings['num_layers']}_"
        f"{rgb_str}_"
        f"{inv_str}_"
        f"z{settings['z_seed']}"
    )

    # Add color matching info if enabled
    if settings.get("color_reference") is not None and settings.get("color_match_strength", 0.0) > 0:
        filename += f"_cm{settings['color_match_strength']:.2f}"

    return filename


def generate_batch(
    num_images=20,
    width=1920,
    height=1080,
    output_dir="exploration",
    master_seed=None
):
    """
    Generate a batch of images with random settings.

    Args:
        num_images: Number of images to generate
        width: Image width
        height: Image height
        output_dir: Output directory
        master_seed: Master random seed for reproducibility
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize random number generator
    rng = np.random.RandomState(master_seed)

    # Store all settings for later analysis
    all_settings = []

    print(f"Generating {num_images} images with random settings...")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Cache for reusing generators when parameters match
    # Key: (net_size, h_size, rgb, scaling, color_reference, color_match_strength)
    generator_cache = {}

    for i in range(num_images):
        # Generate random settings
        settings = generate_random_settings(rng)
        settings["index"] = i
        settings["width"] = width
        settings["height"] = height
        settings["timestamp"] = datetime.now().isoformat()

        # Create filename
        base_filename = settings_to_filename(settings, i)

        # Print progress
        print(f"\n[{i+1}/{num_images}] {base_filename}")
        print(f"  net_size={settings['net_size']}, h_size={settings['h_size']}, "
              f"layers={settings['num_layers']}, scaling={settings['scaling']:.1f}")

        try:
            # Create cache key for generator reuse (excludes z_seed and num_layers)
            cache_key = (
                settings["net_size"],
                settings["h_size"],
                settings["rgb"],
                settings["scaling"],
                settings.get("color_reference"),
                settings.get("color_match_strength", 0.0)
            )

            # Get or create generator
            if cache_key not in generator_cache:
                generator = CPPNGenerator(
                    net_size=settings["net_size"],
                    h_size=settings["h_size"],
                    rgb=settings["rgb"],
                    scaling=settings["scaling"],
                    seed=settings["z_seed"],  # Initial seed for RNG
                    color_reference=settings.get("color_reference"),
                    color_match_strength=settings.get("color_match_strength", 0.0)
                )
                generator_cache[cache_key] = generator
            else:
                generator = generator_cache[cache_key]
                # Reset the RNG with the new seed
                generator.rng = np.random.RandomState(settings["z_seed"])

            # Generate image
            image = generator.generate(
                width=width,
                height=height,
                num_layers=settings["num_layers"]
            )

            # Save image
            image_path = output_path / f"{base_filename}.jpg"
            generator.save_image(image, str(image_path), invert=settings["invert"])

            # Save settings JSON
            #json_path = output_path / f"{base_filename}.json"
            #with open(json_path, 'w') as f:
            #    json.dump(settings, f, indent=2)

            settings["status"] = "success"
            print(f"  ✓ Saved: {image_path.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            settings["status"] = "error"
            settings["error"] = str(e)

        all_settings.append(settings)

    # Save master index
    index_path = output_path / "index.json"
    with open(index_path, 'w') as f:
        json.dump({
            "num_images": num_images,
            "master_seed": master_seed,
            "generation_time": datetime.now().isoformat(),
            "settings": all_settings
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ Generated {num_images} images")
    print(f"✓ Saved master index to {index_path}")
    print("\nFiles saved with format:")
    print("  cppn_XXX_nNN_hNN_lN_sNN.N_rgb/gray_inv/norm_zSEED.jpg")
    print("  cppn_XXX_nNN_hNN_lN_sNN.N_rgb/gray_inv/norm_zSEED.json")


def generate_focused_exploration(
    output_dir="exploration_focused",
    width=1920,
    height=1080
):
    """
    Generate a more focused exploration of specific parameter ranges.
    Varies one parameter at a time while keeping others constant.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_settings = {
        "net_size": 32,
        "h_size": 32,
        "num_layers": 3,
        "scaling": 10.0,
        "rgb": True,
        "z_seed": 42,
        "invert": True
    }

    all_settings = []
    index = 0

    print("Generating focused exploration...")
    print("=" * 70)

    # Create a base generator for reuse when only num_layers or seed changes
    base_generator = CPPNGenerator(
        net_size=base_settings["net_size"],
        h_size=base_settings["h_size"],
        rgb=base_settings["rgb"],
        scaling=base_settings["scaling"],
        seed=base_settings["z_seed"]
    )

    # 1. Vary net_size
    print("\n[1] Varying net_size (16, 24, 32, 48, 64)...")
    for net_size in [16, 24, 32, 48, 64]:
        settings = base_settings.copy()
        settings["net_size"] = net_size
        settings["index"] = index

        generator = CPPNGenerator(
            net_size=settings["net_size"],
            h_size=settings["h_size"],
            rgb=settings["rgb"],
            scaling=settings["scaling"],
            seed=settings["z_seed"]
        )

        image = generator.generate(width=width, height=height, num_layers=settings["num_layers"])

        filename = f"netsize_{net_size:02d}"
        image_path = output_path / f"{filename}.jpg"
        json_path = output_path / f"{filename}.json"

        generator.save_image(image, str(image_path), invert=settings["invert"])
        with open(json_path, 'w') as f:
            json.dump(settings, f, indent=2)

        all_settings.append(settings)
        index += 1
        print(f"  ✓ net_size={net_size}")

    # 2. Vary h_size
    print("\n[2] Varying h_size (8, 16, 32, 64)...")
    for h_size in [8, 16, 32, 64]:
        settings = base_settings.copy()
        settings["h_size"] = h_size
        settings["index"] = index

        generator = CPPNGenerator(
            net_size=settings["net_size"],
            h_size=settings["h_size"],
            rgb=settings["rgb"],
            scaling=settings["scaling"],
            seed=settings["z_seed"]
        )

        image = generator.generate(width=width, height=height, num_layers=settings["num_layers"])

        filename = f"hsize_{h_size:02d}"
        image_path = output_path / f"{filename}.jpg"
        json_path = output_path / f"{filename}.json"

        generator.save_image(image, str(image_path), invert=settings["invert"])
        with open(json_path, 'w') as f:
            json.dump(settings, f, indent=2)

        all_settings.append(settings)
        index += 1
        print(f"  ✓ h_size={h_size}")

    # 3. Vary scaling
    print("\n[3] Varying scaling (3, 7, 10, 15, 20, 25)...")
    for scaling in [3.0, 7.0, 10.0, 15.0, 20.0, 25.0]:
        settings = base_settings.copy()
        settings["scaling"] = scaling
        settings["index"] = index

        generator = CPPNGenerator(
            net_size=settings["net_size"],
            h_size=settings["h_size"],
            rgb=settings["rgb"],
            scaling=settings["scaling"],
            seed=settings["z_seed"]
        )

        image = generator.generate(width=width, height=height, num_layers=settings["num_layers"])

        filename = f"scaling_{scaling:04.1f}".replace('.', '_')
        image_path = output_path / f"{filename}.jpg"
        json_path = output_path / f"{filename}.json"

        generator.save_image(image, str(image_path), invert=settings["invert"])
        with open(json_path, 'w') as f:
            json.dump(settings, f, indent=2)

        all_settings.append(settings)
        index += 1
        print(f"  ✓ scaling={scaling}")

    # 4. Vary num_layers - reuse base generator since only num_layers changes
    print("\n[4] Varying num_layers (2, 3, 4, 5, 6)...")
    for num_layers in [2, 3, 4, 5, 6]:
        settings = base_settings.copy()
        settings["num_layers"] = num_layers
        settings["index"] = index

        # Reuse base_generator
        image = base_generator.generate(width=width, height=height, num_layers=num_layers)

        filename = f"layers_{num_layers}"
        image_path = output_path / f"{filename}.jpg"
        json_path = output_path / f"{filename}.json"

        base_generator.save_image(image, str(image_path), invert=settings["invert"])
        with open(json_path, 'w') as f:
            json.dump(settings, f, indent=2)

        all_settings.append(settings)
        index += 1
        print(f"  ✓ num_layers={num_layers}")

    # 5. Different random seeds - reuse base generator, just reset RNG
    print("\n[5] Varying random seed (10 different seeds)...")
    for i, seed in enumerate([42, 123, 456, 789, 999, 111, 222, 333, 555, 777]):
        settings = base_settings.copy()
        settings["z_seed"] = seed
        settings["index"] = index

        # Reuse base generator but reset the RNG
        base_generator.rng = np.random.RandomState(seed)
        image = base_generator.generate(width=width, height=height, num_layers=settings["num_layers"])

        filename = f"seed_{seed:06d}"
        image_path = output_path / f"{filename}.jpg"
        json_path = output_path / f"{filename}.json"

        base_generator.save_image(image, str(image_path), invert=settings["invert"])
        with open(json_path, 'w') as f:
            json.dump(settings, f, indent=2)

        all_settings.append(settings)
        index += 1
        print(f"  ✓ seed={seed}")

    # Save master index
    index_path = output_path / "index.json"
    with open(index_path, 'w') as f:
        json.dump({
            "exploration_type": "focused",
            "num_images": len(all_settings),
            "generation_time": datetime.now().isoformat(),
            "settings": all_settings
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ Generated {len(all_settings)} images in focused exploration")
    print(f"✓ Saved to {output_dir}/")
    print("\nParameters varied:")
    print("  - net_size: 5 variations")
    print("  - h_size: 4 variations")
    print("  - scaling: 6 variations")
    print("  - num_layers: 5 variations")
    print("  - random seeds: 10 variations")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CPPN Parameter Space Explorer")
    print("=" * 70)
    print("\nGenerating images with configured settings:")
    print(f"  Number of images: {N}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"  Master seed: {MASTER_SEED if MASTER_SEED is not None else 'Random'}")
    print()

    # Generate batch using hardcoded configuration
    generate_batch(
        num_images=N,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        output_dir=OUTPUT_DIR,
        master_seed=MASTER_SEED
    )

    print("\n" + "=" * 70)
    print("Exploration complete!")
    print("\nTips:")
    print("- Check the JSON files to see exact settings for each image")
    print("- Use index.json for an overview of all generated images")
    print("- Filenames encode key parameters for quick identification")
    print("- Look for patterns in what settings produce interesting results")
    print("\nTo change settings, edit the configuration at the top of this script.")
