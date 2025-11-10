"""
network_torch.py - CPPN Network Architecture using PyTorch

This module contains the PyTorch implementation of the CPPN network.
Automatically detects and uses the best available device:
- Apple Silicon: MPS (Metal Performance Shaders)
- NVIDIA GPU: CUDA
- Fallback: CPU

This is a drop-in replacement for network.py with identical interface.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


def get_best_device() -> torch.device:
    """
    Automatically detect the best available device.

    Returns:
        torch.device: Best available device (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders) backend")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Using CUDA backend: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU backend")
        return torch.device("cpu")


class CPPNNetwork:
    """
    CPPN network implementation using PyTorch with automatic device detection.

    This class handles the neural network forward pass that generates
    the image from coordinate inputs and latent vectors.

    Provides identical interface to the NumPy version for drop-in replacement.

    Optimized for video generation:
    - Caches coordinate grids for repeated renders at same resolution
    - Minimizes CPU/GPU data transfers
    - Supports batch inference for processing multiple frames at once
    """

    def __init__(self, net_size: int, h_size: int, c_dim: int, scaling: float, rng: np.random.RandomState):
        """
        Initialize the CPPN network.

        Args:
            net_size: Number of neurons in hidden layers
            h_size: Size of the latent vector
            c_dim: Number of output channels (1 for grayscale, 3 for RGB)
            scaling: Coordinate scaling factor
            rng: Random number generator for weight initialization (NumPy)
        """
        self.net_size = net_size
        self.h_size = h_size
        self.c_dim = c_dim
        self.scaling = scaling
        self.rng = rng

        # Automatically select best device
        self.device = get_best_device()

        # Storage for network weights (as PyTorch tensors)
        self.weights: Dict[str, torch.Tensor] = {}

        # Cache for coordinate tensors (key: (width, height, scaling))
        self._coord_cache: Dict[tuple, tuple] = {}

        # Track if weights have been initialized
        self._weights_initialized = False

    def _initialize_layer_weights(
        self,
        in_dim: int,
        out_dim: int,
        layer_name: str,
        with_bias: bool = True
    ) -> None:
        """
        Initialize weights for a layer (called once during network build).

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            layer_name: Unique name for this layer
            with_bias: Whether to include bias term
        """
        weight_key = f"{layer_name}_weight"
        bias_key = f"{layer_name}_bias"

        if weight_key not in self.weights:
            # Initialize weights using NumPy RNG for consistency
            weight_np = self.rng.standard_normal(size=(in_dim, out_dim)).astype(np.float32)
            self.weights[weight_key] = torch.from_numpy(weight_np).to(self.device)

        if with_bias and bias_key not in self.weights:
            bias_np = self.rng.standard_normal(size=(1, out_dim)).astype(np.float32)
            self.weights[bias_key] = torch.from_numpy(bias_np).to(self.device)

    def _fully_connected(
        self,
        x: torch.Tensor,
        layer_name: str,
        with_bias: bool = True
    ) -> torch.Tensor:
        """
        Apply a fully connected layer with pre-initialized weights.

        Args:
            x: Input tensor on device
            layer_name: Unique name for this layer (for weight retrieval)
            with_bias: Whether to include bias term

        Returns:
            Output of the layer (tensor on device)
        """
        weight_key = f"{layer_name}_weight"
        bias_key = f"{layer_name}_bias"

        # Weights must be initialized before calling this
        assert weight_key in self.weights, f"Weights not initialized for layer {layer_name}"

        # Matrix multiplication
        result = torch.matmul(x, self.weights[weight_key])

        if with_bias:
            result += self.weights[bias_key]

        return result

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid activation function."""
        return torch.sigmoid(x)

    def _prepare_coordinates(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        r_coords: np.ndarray,
        scaling: float
    ) -> tuple:
        """
        Convert and cache coordinate arrays to GPU tensors.

        Caches coordinates by their shape and scaling to avoid redundant
        transfers for video generation at fixed resolution.

        Args:
            x_coords: X coordinates, shape (num_points, 1)
            y_coords: Y coordinates, shape (num_points, 1)
            r_coords: Radial coordinates, shape (num_points, 1)
            scaling: Scaling factor used to generate these coords

        Returns:
            Tuple of (x_tensor, y_tensor, r_tensor) on device
        """
        # Create cache key based on shape and scaling
        cache_key = (x_coords.shape[0], scaling)

        if cache_key in self._coord_cache:
            return self._coord_cache[cache_key]

        # Convert to tensors and move to device
        x_tensor = torch.from_numpy(x_coords.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_coords.astype(np.float32)).to(self.device)
        r_tensor = torch.from_numpy(r_coords.astype(np.float32)).to(self.device)

        # Cache for future use
        self._coord_cache[cache_key] = (x_tensor, y_tensor, r_tensor)

        return x_tensor, y_tensor, r_tensor

    def build_network_architecture(self, num_layers: int = 3) -> None:
        """
        Initialize the network weights ONCE.

        This should be called once during setup. After this, use forward_pass()
        or forward_pass_batch() to generate images.

        Args:
            num_layers: Number of hidden layers in the network
        """
        if self._weights_initialized:
            print("Warning: Network weights already initialized. Skipping.")
            return

        # Initialize all layer weights with dummy dimensions
        # The actual dimensions will be determined by the input
        # Input layers (coordinates and latent vector)
        self._initialize_layer_weights(self.h_size, self.net_size, "z_input")
        self._initialize_layer_weights(1, self.net_size, "x_input", with_bias=False)
        self._initialize_layer_weights(1, self.net_size, "y_input", with_bias=False)
        self._initialize_layer_weights(1, self.net_size, "r_input", with_bias=False)

        # Hidden layers
        for i in range(num_layers):
            self._initialize_layer_weights(self.net_size, self.net_size, f"hidden_{i}")

        # Output layer
        self._initialize_layer_weights(self.net_size, self.c_dim, "output")

        self._weights_initialized = True
        print(f"✓ Network architecture built: {num_layers} hidden layers, {self.net_size} neurons per layer")

    def forward_pass_batch(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        r_coords: np.ndarray,
        z_vecs: np.ndarray,
        num_layers: int = 3
    ) -> np.ndarray:
        """
        Run forward pass for a BATCH of latent vectors (optimized for video generation).

        Args:
            x_coords: X coordinates, shape (num_points, 1)
            y_coords: Y coordinates, shape (num_points, 1)
            r_coords: Radial coordinates, shape (num_points, 1)
            z_vecs: Batch of latent vectors, shape (batch_size, h_size)
            num_layers: Number of hidden layers

        Returns:
            Network outputs, shape (batch_size, num_points, c_dim) in range [0, 1]
        """
        # Build network if not already built
        if not self._weights_initialized:
            self.build_network_architecture(num_layers)

        batch_size = z_vecs.shape[0]
        num_points = x_coords.shape[0]

        # Use cached coordinate tensors
        x_tensor, y_tensor, r_tensor = self._prepare_coordinates(
            x_coords, y_coords, r_coords, self.scaling
        )

        # Convert z vectors to tensor
        z_tensor = torch.from_numpy(z_vecs.astype(np.float32)).to(self.device)

        with torch.no_grad():
            # Expand coordinates to batch dimensions
            # Shape: (batch_size, num_points, 1)
            x_batch = x_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            y_batch = y_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            r_batch = r_tensor.unsqueeze(0).expand(batch_size, -1, -1)

            # Expand z_vec to match coordinate dimensions
            # Shape: (batch_size, num_points, h_size)
            z_expanded = z_tensor.unsqueeze(1).expand(-1, num_points, -1) * self.scaling

            # Flatten batch and point dimensions for processing
            x_flat = x_batch.reshape(-1, 1)
            y_flat = y_batch.reshape(-1, 1)
            r_flat = r_batch.reshape(-1, 1)
            z_flat = z_expanded.reshape(-1, self.h_size)

            # Initial layer combines all inputs
            h = (
                self._fully_connected(z_flat, "z_input") +
                self._fully_connected(x_flat, "x_input", with_bias=False) +
                self._fully_connected(y_flat, "y_input", with_bias=False) +
                self._fully_connected(r_flat, "r_input", with_bias=False)
            )

            # Hidden layers with tanh activation
            h = torch.tanh(h)
            for i in range(num_layers):
                h = torch.tanh(self._fully_connected(h, f"hidden_{i}"))

            # Output layer with sigmoid for [0, 1] range
            output = self._sigmoid(self._fully_connected(h, "output"))

            # Reshape back to (batch_size, num_points, c_dim)
            output = output.reshape(batch_size, num_points, self.c_dim)

        return output.cpu().numpy()

    def build_network(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        r_coords: np.ndarray,
        z_vec: np.ndarray,
        num_layers: int = 3
    ) -> np.ndarray:
        """
        BACKWARD COMPATIBILITY: Build and execute the CPPN network forward pass for single image.

        This maintains the old API but internally uses the optimized batch processing.
        For video generation, prefer using forward_pass_batch() directly.

        Args:
            x_coords: X coordinates, shape (num_points, 1)
            y_coords: Y coordinates, shape (num_points, 1)
            r_coords: Radial coordinates, shape (num_points, 1)
            z_vec: Latent vector, shape (1, h_size)
            num_layers: Number of hidden layers

        Returns:
            Network output as NumPy array, shape (num_points, c_dim) in range [0, 1]
        """
        # Ensure z_vec is 2D for batch processing
        if z_vec.ndim == 1:
            z_vec = z_vec.reshape(1, -1)

        # Use batch processing with batch_size=1
        output = self.forward_pass_batch(x_coords, y_coords, r_coords, z_vec, num_layers)

        # Return single image (squeeze batch dimension)
        return output[0]

    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get the current network weights as NumPy arrays.

        Returns:
            Dictionary of weight arrays
        """
        return {k: v.cpu().numpy() for k, v in self.weights.items()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """
        Set network weights from NumPy arrays.

        Args:
            weights: Dictionary of weight arrays
        """
        self.weights = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in weights.items()
        }
