"""
Color Matching Module - Smooth color transfer for CPPN images

This module provides smooth, continuous color matching using LAB color space
transformations. Designed for flicker-free color transfer in video sequences.

Author: Extracted from CPPN project, 2025
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
from skimage import color as skcolor


class ColorMatcher:
    """
    Smooth color transfer using continuous non-linear transforms in LAB color space.

    This class provides continuous, flicker-free color mapping for video sequences.
    Uses smooth sigmoid-based transforms instead of hard clipping to ensure
    temporal smoothness emerges naturally from smooth input frames.
    """

    def __init__(self, reference_image_path: str, smoothness: float = 5.0, mode: str = 'mean_std', max_size: int = 512):
        """
        Initialize color matcher with a reference image.

        Args:
            reference_image_path: Path to the reference image
            smoothness: Controls the smoothness of the sigmoid limiting (higher = smoother)
            mode: Matching mode - 'mean_std' (stronger, better for color tone) or
                  'percentile' (gentler, better for contrast preservation)
            max_size: Maximum dimension for downscaled statistics computation (default: 512)
        """
        self.smoothness = smoothness
        self.mode = mode
        self.max_size = max_size

        # Load and process reference image
        ref_img = Image.open(reference_image_path).convert('RGB')
        ref_array = np.array(ref_img).astype(np.float32) / 255.0

        # Downscale reference image for statistics computation
        ref_array_downscaled = self._downscale_bicubic(ref_array, max_size)

        # Convert to LAB color space (on downscaled image)
        ref_lab_downscaled = skcolor.rgb2lab(ref_array_downscaled)

        # Compute reference statistics for each LAB channel (on downscaled image)
        self.ref_stats = []
        for channel in range(3):
            channel_data = ref_lab_downscaled[:, :, channel].flatten()

            # Use percentiles for robust statistics (less sensitive to outliers)
            p05, p25, p50, p75, p95 = np.percentile(channel_data, [5, 25, 50, 75, 95])

            stats = {
                'min': np.min(channel_data),
                'max': np.max(channel_data),
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'median': p50,
                'p05': p05,
                'p25': p25,
                'p75': p75,
                'p95': p95,
                'iqr': p75 - p25  # Interquartile range
            }
            self.ref_stats.append(stats)

    def _downscale_bicubic(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """
        Downscale image using bicubic interpolation if needed.

        Args:
            image: Input image array in range [0, 1] with shape (H, W, 3)
            max_size: Maximum dimension for downscaled image

        Returns:
            Downscaled image (or original if already small enough)
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim <= max_size:
            return image

        # Calculate new dimensions maintaining aspect ratio
        scale = max_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Convert to PIL for high-quality bicubic downscaling
        # PIL expects uint8 for best results
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(image_uint8)
        pil_img_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)

        # Convert back to float32 in [0, 1]
        downscaled = np.array(pil_img_resized).astype(np.float32) / 255.0

        return downscaled

    def _smooth_normalize(self, x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
        """
        Smooth normalization to [0, 1] using sigmoid-based soft limits.

        This ensures continuity even for values outside [x_min, x_max].
        """
        # Normalize to approximately [0, 1]
        x_range = x_max - x_min
        if x_range < 1e-10:
            return np.full_like(x, 0.5)

        # Linear mapping to [-1, 1] centered at midpoint
        normalized = 2 * (x - x_min) / x_range - 1

        # Apply smooth sigmoid to map to [0, 1] with soft boundaries
        # tanh provides smooth limiting without hard clipping
        return 0.5 * (np.tanh(normalized / self.smoothness) + 1)

    def _smooth_denormalize(self, y: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
        """
        Inverse of smooth normalization - maps [0, 1] to target range smoothly.
        """
        # Map [0, 1] to [-1, 1] via inverse tanh
        # Clamp y slightly away from boundaries to avoid numerical issues
        y_clamped = np.clip(y, 1e-7, 1 - 1e-7)
        normalized = np.arctanh(2 * y_clamped - 1) * self.smoothness

        # Map to target range
        target_range = target_max - target_min
        return target_min + target_range * (normalized + 1) / 2

    def _continuous_percentile_match(
        self,
        src_data: np.ndarray,
        src_stats: dict,
        ref_stats: dict
    ) -> np.ndarray:
        """
        Apply continuous percentile-based color matching.

        Uses smooth transformations throughout - no hard clipping.
        """
        # Step 1: Smoothly normalize source data to [0, 1]
        normalized = self._smooth_normalize(src_data, src_stats['p05'], src_stats['p95'])

        # Step 2: Apply non-linear transform based on distribution shapes
        # Match the interquartile ranges for better color distribution
        src_iqr_normalized = (src_stats['p75'] - src_stats['p25']) / (src_stats['p95'] - src_stats['p05'] + 1e-10)
        ref_iqr_normalized = (ref_stats['p75'] - ref_stats['p25']) / (ref_stats['p95'] - ref_stats['p05'] + 1e-10)

        # Adjust contrast by modulating around the median
        # This is a smooth, continuous operation
        if src_iqr_normalized > 1e-6:
            contrast_factor = ref_iqr_normalized / src_iqr_normalized
            # Apply power transform for smooth contrast adjustment
            power = 0.5 + 0.5 * contrast_factor  # Smooth power between 0.5 and inf
            normalized = normalized ** power

        # Step 3: Smoothly denormalize to target range
        matched = self._smooth_denormalize(normalized, ref_stats['p05'], ref_stats['p95'])

        return matched

    def _match_mean_std(
        self,
        src_data: np.ndarray,
        src_stats: dict,
        ref_stats: dict
    ) -> np.ndarray:
        """
        Simple mean and standard deviation matching.

        This directly shifts the distribution to match the reference,
        which is more effective for preserving overall color tone.
        """
        # Standardize source data (zero mean, unit variance)
        standardized = (src_data - src_stats['mean']) / (src_stats['std'] + 1e-10)

        # Scale and shift to match reference statistics
        matched = standardized * ref_stats['std'] + ref_stats['mean']

        return matched

    def _compute_transform_params(
        self,
        src_lab: np.ndarray
    ) -> list:
        """
        Compute transformation parameters from source LAB image to reference statistics.

        This computes the mapping parameters at downscaled resolution for efficiency.

        Args:
            src_lab: Source image in LAB color space (downscaled)

        Returns:
            List of transformation parameters for each channel
        """
        transform_params = []

        for channel in range(3):
            channel_data = src_lab[:, :, channel].flatten()

            # Compute source statistics
            p05, p25, p50, p75, p95 = np.percentile(channel_data, [5, 25, 50, 75, 95])
            src_stats = {
                'min': np.min(channel_data),
                'max': np.max(channel_data),
                'mean': np.mean(channel_data),
                'std': np.std(channel_data),
                'median': p50,
                'p05': p05,
                'p25': p25,
                'p75': p75,
                'p95': p95,
                'iqr': p75 - p25
            }

            # Store parameters needed for transformation
            transform_params.append({
                'src_stats': src_stats,
                'ref_stats': self.ref_stats[channel]
            })

        return transform_params

    def _apply_transform_fullres(
        self,
        lab_img: np.ndarray,
        transform_params: list,
        strength: float
    ) -> np.ndarray:
        """
        Apply pre-computed transformation parameters to full-resolution LAB image.

        Args:
            lab_img: Full-resolution image in LAB color space
            transform_params: Pre-computed transformation parameters
            strength: Blending strength

        Returns:
            Matched LAB image
        """
        matched_lab = np.zeros_like(lab_img)

        # LAB color space valid ranges (approximate, for smooth limiting)
        lab_bounds = [
            (0, 100),      # L channel
            (-128, 127),   # a channel
            (-128, 127)    # b channel
        ]

        for channel in range(3):
            channel_data = lab_img[:, :, channel]
            flat_data = channel_data.flatten()

            # Apply color matching using pre-computed parameters
            params = transform_params[channel]
            if self.mode == 'mean_std':
                matched_flat = self._match_mean_std(
                    flat_data,
                    params['src_stats'],
                    params['ref_stats']
                )
            else:  # percentile mode
                matched_flat = self._continuous_percentile_match(
                    flat_data,
                    params['src_stats'],
                    params['ref_stats']
                )

            # Smoothly constrain to valid LAB range
            low, high = lab_bounds[channel]
            matched_flat = self._sigmoid_limit(matched_flat, low, high)

            matched_channel = matched_flat.reshape(channel_data.shape)

            # Smooth blending between original and matched
            matched_lab[:, :, channel] = (
                (1 - strength) * channel_data + strength * matched_channel
            )

        return matched_lab

    def apply_color_matching(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply smooth color matching to an image.

        This method is optimized for speed by computing statistics on downscaled images
        but applying the transformations at full resolution.

        Args:
            image: Input image in range [0, 1] with shape (H, W, 3)
            strength: Blending strength from 0 (no change) to 1 (full matching)

        Returns:
            Color-matched image in range [0, 1] (soft-limited, never hard-clipped)
        """
        # Smooth strength clamping (though user should provide valid values)
        strength = np.clip(strength, 0.0, 1.0)

        if strength < 1e-6:
            return image

        # Step 1: Downscale image for fast statistics computation
        image_downscaled = self._downscale_bicubic(image, self.max_size)

        # Step 2: Convert downscaled image to LAB and compute transformation parameters
        lab_downscaled = skcolor.rgb2lab(image_downscaled)
        transform_params = self._compute_transform_params(lab_downscaled)

        # Step 3: Convert full-resolution image to LAB
        lab_img = skcolor.rgb2lab(image)

        # Step 4: Apply transformation at full resolution (fast - just linear operations)
        matched_lab = self._apply_transform_fullres(lab_img, transform_params, strength)

        # Step 5: Convert back to RGB
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            matched_rgb = skcolor.lab2rgb(matched_lab)

        # Apply smooth sigmoid limiting instead of hard clipping
        matched_rgb = self._sigmoid_limit(matched_rgb, 0.0, 1.0)

        return matched_rgb

    def _sigmoid_limit(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        """
        Apply smooth sigmoid-based limiting to range [low, high].

        Unlike np.clip(), this is continuous and differentiable everywhere.
        """
        # Map to [0, 1] with smooth boundaries
        x_range = high - low
        normalized = (x - low) / x_range

        # Apply smooth sigmoid limiting
        # Using a gentle slope to maintain most values while smoothly handling extremes
        limited = 1.0 / (1.0 + np.exp(-self.smoothness * (normalized - 0.5)))

        # Map back to [low, high]
        return low + x_range * limited
