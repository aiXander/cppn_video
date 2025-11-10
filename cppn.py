"""
CPPN - Generate Art using Neural Networks

Compositional Pattern Producing Networks generate artistic images by
using neural networks with various activation functions.

Author: AntixK (Original), Modernized 2025
Inspired by: https://github.com/hardmaru/cppn-tensorflow
"""

from typing import Optional, Tuple, Callable
import numpy as np
from PIL import Image
from pathlib import Path

from color_matching import ColorMatcher


class CPPNGenerator:
    """
    Compositional Pattern Producing Network for generating artistic images.

    This class creates images by passing coordinate information through a
    neural network with various activation functions.
    """

    # Class-level cache for ColorMatcher instances
    # Key: (color_reference_path, color_match_mode)
    # Value: ColorMatcher instance
    _color_matcher_cache = {}

    def __init__(
        self,
        net_size: int = 32,
        h_size: int = 32,
        rgb: bool = True,
        scaling: float = 10.0,
        seed: Optional[int] = None,
        color_reference: Optional[str] = None,
        color_match_strength: float = 1.0,
        color_match_mode: str = 'mean_std'
    ):
        """
        Initialize the CPPN generator.

        Args:
            net_size: Number of neurons in hidden layers
            h_size: Size of the latent vector
            rgb: Whether to generate RGB (True) or grayscale (False) images
            scaling: Coordinate scaling factor
            seed: Random seed for reproducibility
            color_reference: Path to reference image for color matching (optional)
            color_match_strength: Strength of color matching from 0 to 1 (default: 1.0)
            color_match_mode: Mode for color matching - 'mean_std' (default, better color tone)
                              or 'percentile' (better contrast preservation)
        """
        self.net_size = net_size
        self.h_size = h_size
        self.c_dim = 3 if rgb else 1
        self.scaling = scaling
        self.color_match_strength = color_match_strength

        # Use a RandomState for reproducibility
        self.rng = np.random.RandomState(seed)

        # Store network weights for consistency
        self.weights = {}

        # Initialize color matcher if reference provided
        self.color_matcher = None
        if color_reference is not None:
            if not rgb:
                print("Warning: Color matching only works with RGB images. Ignoring color_reference.")
            else:
                self.color_matcher = self._get_color_matcher(color_reference, color_match_mode)

    @classmethod
    def _get_color_matcher(cls, color_reference: str, color_match_mode: str = 'mean_std') -> ColorMatcher:
        """
        Get or create a cached ColorMatcher instance.

        Args:
            color_reference: Path to reference image
            color_match_mode: Mode for color matching

        Returns:
            ColorMatcher instance (cached)
        """
        cache_key = (color_reference, color_match_mode)

        if cache_key not in cls._color_matcher_cache:
            cls._color_matcher_cache[cache_key] = ColorMatcher(color_reference, mode=color_match_mode)

        return cls._color_matcher_cache[cache_key]

    @classmethod
    def clear_color_matcher_cache(cls):
        """Clear the ColorMatcher cache. Useful if reference images change."""
        cls._color_matcher_cache.clear()

    def _create_grid(
        self,
        x_res: int,
        y_res: int,
        scaling: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create coordinate grids for the image.

        Args:
            x_res: Width of the image
            y_res: Height of the image
            scaling: Coordinate scaling factor

        Returns:
            Tuple of (x_coords, y_coords, radius) arrays
        """
        x_range = np.linspace(-scaling, scaling, num=x_res)
        y_range = np.linspace(-scaling, scaling, num=y_res)

        x_mat = np.outer(np.ones(y_res), x_range)
        y_mat = np.outer(y_range, np.ones(x_res))
        r_mat = np.sqrt(x_mat**2 + y_mat**2)

        x_flat = x_mat.flatten().reshape(-1, 1)
        y_flat = y_mat.flatten().reshape(-1, 1)
        r_flat = r_mat.flatten().reshape(-1, 1)

        return x_flat, y_flat, r_flat

    def _fully_connected(
        self,
        x: np.ndarray,
        out_dim: int,
        layer_name: str,
        with_bias: bool = True
    ) -> np.ndarray:
        """
        Apply a fully connected layer with stored weights.

        Args:
            x: Input array
            out_dim: Output dimension
            layer_name: Unique name for this layer (for weight storage)
            with_bias: Whether to include bias term

        Returns:
            Output of the layer
        """
        in_dim = x.shape[1]

        # Create or retrieve weights
        weight_key = f"{layer_name}_weight"
        bias_key = f"{layer_name}_bias"

        if weight_key not in self.weights:
            self.weights[weight_key] = self.rng.standard_normal(
                size=(in_dim, out_dim)
            ).astype(np.float32)

        if with_bias and bias_key not in self.weights:
            self.weights[bias_key] = self.rng.standard_normal(
                size=(1, out_dim)
            ).astype(np.float32)

        result = np.matmul(x, self.weights[weight_key])

        if with_bias:
            result += self.weights[bias_key]

        return result

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus activation function."""
        return np.log(1.0 + np.exp(np.clip(x, -20, 20)))

    def _build_network(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        r_coords: np.ndarray,
        z_vec: np.ndarray,
        num_layers: int = 3
    ) -> np.ndarray:
        """
        Build the CPPN network.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            r_coords: Radial coordinates
            z_vec: Latent vector
            num_layers: Number of hidden layers

        Returns:
            Network output
        """
        num_points = x_coords.shape[0]

        # Expand z_vec to match coordinate dimensions
        z_expanded = np.tile(z_vec, (num_points, 1)) * self.scaling

        # Initial layer combines all inputs
        h = (
            self._fully_connected(z_expanded, self.net_size, "z_input") +
            self._fully_connected(x_coords, self.net_size, "x_input", with_bias=False) +
            self._fully_connected(y_coords, self.net_size, "y_input", with_bias=False) +
            self._fully_connected(r_coords, self.net_size, "r_input", with_bias=False)
        )

        # Hidden layers with tanh activation
        h = np.tanh(h)
        for i in range(num_layers):
            h = np.tanh(self._fully_connected(h, self.net_size, f"hidden_{i}"))

        # Output layer with sigmoid for [0, 1] range
        output = self._sigmoid(self._fully_connected(h, self.c_dim, "output"))

        return output

    def generate(
        self,
        width: int = 768,
        height: int = 512,
        z: Optional[np.ndarray] = None,
        scaling: Optional[float] = None,
        num_layers: int = 3,
        color_match_strength: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate an image.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            z: Latent vector (will be randomly generated if None)
            scaling: Coordinate scaling (uses default if None)
            num_layers: Number of hidden layers in the network
            color_match_strength: Override color matching strength for this generation

        Returns:
            Generated image as numpy array with shape (height, width, channels)
        """
        if scaling is None:
            scaling = self.scaling

        if z is None:
            z = self.rng.uniform(-1.0, 1.0, size=(1, self.h_size)).astype(np.float32)

        # Ensure z is 2D
        if z.ndim == 1:
            z = z.reshape(1, -1)

        # Initialize weights only if they don't exist yet
        # This ensures consistent network across multiple generations
        if not self.weights:
            # Weights will be created on first forward pass
            pass

        # Create coordinate grids
        x_coords, y_coords, r_coords = self._create_grid(width, height, scaling)

        # Generate image
        output = self._build_network(x_coords, y_coords, r_coords, z, num_layers)

        # Reshape to image
        if self.c_dim == 1:
            image = output.reshape(height, width)
        else:
            image = output.reshape(height, width, self.c_dim)

        # Apply color matching if enabled
        if self.color_matcher is not None and self.c_dim == 3:
            strength = color_match_strength if color_match_strength is not None else self.color_match_strength
            image = self.color_matcher.apply_color_matching(image, strength=strength)

        return image

    def save_image(
        self,
        image: np.ndarray,
        path: str,
        invert: bool = False,
        bit_depth: int = 16
    ) -> None:
        """
        Save generated image to disk.

        Args:
            image: Image array from generate()
            path: Output file path
            invert: Whether to invert colors
            bit_depth: Bit depth for saving (8 or 16). 16-bit provides smoother gradients.
        """
        # Process image
        img_data = 1 - image if invert else image

        if bit_depth == 16:
            # Use 16-bit for much smoother gradients (65536 levels vs 256)
            img_data = np.clip(img_data * 65535, 0, 65535).astype(np.uint16)
            # Ensure we save as PNG for 16-bit support
            if not path.lower().endswith('.png'):
                path = path.rsplit('.', 1)[0] + '.png'
                print(f"Note: Using PNG format for 16-bit depth")
        else:
            # 8-bit standard
            img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)

        # Create PIL image and save
        im = Image.fromarray(img_data)
        im.save(path)
        print(f"Image saved to {path}")


class CPPNVideoGenerator:
    """
    Generate animated videos by interpolating CPPN parameters.
    """

    def __init__(
        self,
        generator: CPPNGenerator,
        width: int = 768,
        height: int = 512
    ):
        """
        Initialize video generator.

        Args:
            generator: CPPNGenerator instance to use
            width: Video frame width
            height: Video frame height
        """
        self.generator = generator
        self.width = width
        self.height = height

    def interpolate_latent(
        self,
        z_start: np.ndarray,
        z_end: np.ndarray,
        num_frames: int,
        interpolation: str = "linear"
    ) -> np.ndarray:
        """
        Interpolate between two latent vectors.

        Args:
            z_start: Starting latent vector
            z_end: Ending latent vector
            num_frames: Number of frames to generate
            interpolation: Interpolation method ("linear" or "spherical")

        Returns:
            Array of interpolated latent vectors
        """
        t = np.linspace(0, 1, num_frames)

        if interpolation == "linear":
            # Linear interpolation
            z_interp = np.array([
                (1 - t_i) * z_start + t_i * z_end
                for t_i in t
            ])
        elif interpolation == "spherical":
            # Spherical linear interpolation (slerp)
            z_start_norm = z_start / np.linalg.norm(z_start)
            z_end_norm = z_end / np.linalg.norm(z_end)

            omega = np.arccos(np.clip(np.dot(z_start_norm.flatten(), z_end_norm.flatten()), -1, 1))
            sin_omega = np.sin(omega)

            if sin_omega < 1e-6:
                z_interp = np.array([z_start] * num_frames)
            else:
                z_interp = np.array([
                    (np.sin((1 - t_i) * omega) / sin_omega) * z_start +
                    (np.sin(t_i * omega) / sin_omega) * z_end
                    for t_i in t
                ])
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        return z_interp

    def generate_frames(
        self,
        num_frames: int,
        z_start: Optional[np.ndarray] = None,
        z_end: Optional[np.ndarray] = None,
        scaling_start: Optional[float] = None,
        scaling_end: Optional[float] = None,
        interpolation: str = "linear"
    ) -> list:
        """
        Generate a sequence of frames with interpolated parameters.

        Args:
            num_frames: Number of frames to generate
            z_start: Starting latent vector (random if None)
            z_end: Ending latent vector (random if None)
            scaling_start: Starting scaling factor
            scaling_end: Ending scaling factor
            interpolation: Interpolation method

        Returns:
            List of generated image arrays
        """
        # Generate random latent vectors if not provided
        if z_start is None:
            z_start = self.generator.rng.uniform(-1.0, 1.0,
                                                 size=(1, self.generator.h_size)).astype(np.float32)
        if z_end is None:
            z_end = self.generator.rng.uniform(-1.0, 1.0,
                                               size=(1, self.generator.h_size)).astype(np.float32)

        # Interpolate latent vectors
        z_sequence = self.interpolate_latent(z_start, z_end, num_frames, interpolation)

        # Interpolate scaling if provided
        if scaling_start is not None and scaling_end is not None:
            t = np.linspace(0, 1, num_frames)
            scaling_sequence = (1 - t) * scaling_start + t * scaling_end
        else:
            scaling_sequence = [None] * num_frames

        # Generate frames
        frames = []
        for i, (z, scaling) in enumerate(zip(z_sequence, scaling_sequence)):
            print(f"Generating frame {i+1}/{num_frames}...", end="\r")
            frame = self.generator.generate(
                width=self.width,
                height=self.height,
                z=z,
                scaling=scaling
            )
            frames.append(frame)

        print(f"\nGenerated {num_frames} frames!")
        return frames

    def save_frames(
        self,
        frames: list,
        output_dir: str,
        prefix: str = "frame",
        invert: bool = False,
        bit_depth: int = 16
    ) -> None:
        """
        Save frames as individual images.

        Args:
            frames: List of frame arrays
            output_dir: Output directory
            prefix: Filename prefix
            invert: Whether to invert colors
            bit_depth: Bit depth for saving (8 or 16). 16-bit provides smoother gradients.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            # Use PNG for 16-bit, JPG for 8-bit
            ext = ".png" if bit_depth == 16 else ".jpg"
            filename = output_path / f"{prefix}_{i:04d}{ext}"
            self.generator.save_image(frame, str(filename), invert=invert, bit_depth=bit_depth)

    def save_video(
        self,
        frames: list,
        output_path: str,
        fps: int = 30,
        invert: bool = False,
        bit_depth: int = 8
    ) -> None:
        """
        Save frames as a video file (requires imageio with ffmpeg).

        Args:
            frames: List of frame arrays
            output_path: Output video file path
            fps: Frames per second
            invert: Whether to invert colors
            bit_depth: Bit depth for video encoding (8 or 16). Note: 16-bit requires H.265/HEVC codec.
                       Most video formats use 8-bit, but 16-bit is useful for high-quality intermediate files.
        """
        try:
            import imageio
        except ImportError:
            print("imageio not installed. Saving frames as images instead.")
            self.save_frames(frames, "output_frames", invert=invert, bit_depth=bit_depth)
            return

        # Process frames
        processed_frames = []
        for frame in frames:
            img_data = 1 - frame if invert else frame

            if bit_depth == 16:
                # 16-bit video output (requires appropriate codec support)
                img_data = np.clip(img_data * 65535, 0, 65535).astype(np.uint16)
            else:
                # Standard 8-bit video
                img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)

            processed_frames.append(img_data)

        # Save video
        imageio.mimsave(output_path, processed_frames, fps=fps)
        print(f"Video saved to {output_path}")


def generate_image(
    width: int = 768,
    height: int = 512,
    net_size: int = 32,
    h_size: int = 32,
    scaling: float = 10.0,
    rgb: bool = True,
    seed: Optional[int] = None,
    output_path: str = "art.png",
    invert: bool = True,
    color_reference: Optional[str] = None,
    color_match_strength: float = 1.0,
    color_match_mode: str = 'mean_std',
    bit_depth: int = 16
) -> np.ndarray:
    """
    Quick function to generate and save a single CPPN image.

    Args:
        width: Image width
        height: Image height
        net_size: Network size (more = more complex patterns)
        h_size: Latent vector size
        scaling: Coordinate scaling (higher = more zoomed out)
        rgb: Generate RGB or grayscale
        seed: Random seed for reproducibility
        output_path: Where to save the image (default: PNG for 16-bit support)
        invert: Invert colors
        color_reference: Path to reference image for color matching (optional)
        color_match_strength: Strength of color matching from 0 to 1
        color_match_mode: Mode for color matching - 'mean_std' (default, better color tone)
                          or 'percentile' (better contrast preservation)
        bit_depth: Bit depth for saving (8 or 16). 16-bit provides much smoother gradients.

    Returns:
        Generated image array

    Note:
        ColorMatcher instances are cached automatically when color_reference is provided,
        so calling this function multiple times with the same reference image is efficient.
    """
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

    image = generator.generate(width=width, height=height)
    generator.save_image(image, output_path, invert=invert, bit_depth=bit_depth)

    return image


if __name__ == '__main__':
    # Example 1: Generate a single image with 16-bit depth for smooth gradients
    print("Generating single image...")
    generate_image(
        width=512,
        height=512,
        net_size=32,
        h_size=32,
        scaling=10.0,
        rgb=True,
        seed=42,
        output_path="example_art.png",  # PNG format for 16-bit support
        bit_depth=16  # Use 16-bit for smooth gradients (default)
    )

    # Example 2: Generate with color reference matching
    print("\nGenerating image with color matching...")
    # First create a reference image or use an existing one
    # generate_image(
    #     width=512,
    #     height=512,
    #     net_size=32,
    #     h_size=32,
    #     seed=99,
    #     output_path="color_reference.jpg",
    #     color_reference="path/to/your/reference.jpg",
    #     color_match_strength=0.8  # 0.0 = no matching, 1.0 = full matching
    # )

    # Example 3: Generate a video sequence
    print("\nGenerating video frames...")
    generator = CPPNGenerator(net_size=32, h_size=32, rgb=True, seed=123)
    video_gen = CPPNVideoGenerator(generator, width=256, height=256)

    # Create frames with interpolating parameters
    frames = video_gen.generate_frames(
        num_frames=60,
        scaling_start=5.0,
        scaling_end=20.0,
        interpolation="spherical"
    )

    # Save as individual frames
    video_gen.save_frames(frames, "output_frames", invert=True)

    # Optionally save as video (requires imageio)
    # video_gen.save_video(frames, "cppn_animation.mp4", fps=30, invert=True)

    # Example 4: Generate video with color matching for smooth color consistency
    # print("\nGenerating color-matched video...")
    # generator_colored = CPPNGenerator(
    #     net_size=32,
    #     h_size=32,
    #     rgb=True,
    #     seed=456,
    #     color_reference="path/to/your/reference.jpg",
    #     color_match_strength=0.7
    # )
    # video_gen_colored = CPPNVideoGenerator(generator_colored, width=512, height=512)
    # frames_colored = video_gen_colored.generate_frames(num_frames=120, interpolation="spherical")
    # video_gen_colored.save_video(frames_colored, "cppn_color_matched.mp4", fps=30, invert=True)
