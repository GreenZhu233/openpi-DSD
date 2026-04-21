"""GPU-accelerated utilities for tracking trajectory statistics."""

import jax.numpy as jnp
import torch


# ============== JAX 版本 ==============

class GPUWindowedTrajectoryDistance:
    """GPU-accelerated sliding window for trajectory distance tracking.
    
    This class keeps all data on GPU to avoid CPU-GPU data transfer overhead.
    All methods return JAX arrays when possible to minimize data transfer.
    """
    
    def __init__(self, window_size: int = 5, device: str = "cuda"):
        """Initialize the GPU-accelerated sliding window.
        
        Args:
            window_size: Size of the sliding window.
            device: Device to store data on ('cuda' or 'cpu').
        """
        self.window_size = window_size
        self._device = device
        # 使用 jax.Array 存储在 GPU 上
        self._window = jnp.zeros(window_size, dtype=jnp.float32)
        self._current_size = 0
        self._head = 0  # 环形缓冲区指针
    
    def add(self, value):
        """Add a new value to the window.
        
        Args:
            value: Can be a float or JAX array. If JAX array, stays on GPU.
        """
        idx = self._head
        self._window = self._window.at[idx].set(value)
        
        # 移动指针
        self._head = (self._head + 1) % self.window_size
        self._current_size = min(self._current_size + 1, self.window_size)
    
    def get_max(self):
        """Get the maximum value in the window. Returns float or JAX scalar."""
        if self._current_size == 0:
            return jnp.inf
        return jnp.max(self._window[:self._current_size])
    
    def get_min(self):
        """Get the minimum value in the window. Returns float or JAX scalar."""
        if self._current_size == 0:
            return jnp.array(0.0, dtype=jnp.float32)
        return jnp.min(self._window[:self._current_size])
    
    def get_last(self):
        """Get the last added value. Returns float or JAX scalar."""
        if self._current_size == 0:
            return jnp.array(0.0, dtype=jnp.float32)
        idx = (self._head - 1) % self.window_size
        return self._window[idx]
    
    def get_stats(self):
        """Get all statistics at once to minimize GPU-CPU transfers.
        
        Returns:
            tuple: (current_size, window_array) - window_array may contain padding
        """
        return self._current_size, self._window


# ============== PyTorch 版本 ==============

class GPUWindowedTrajectoryDistanceTorch:
    """GPU-accelerated sliding window for trajectory distance tracking.
    
    This class keeps all data on GPU to avoid CPU-GPU data transfer overhead.
    Uses PyTorch's register_buffer for proper device management.
    """
    
    def __init__(self, window_size: int = 5, device: str = "cuda"):
        """Initialize the GPU-accelerated sliding window.
        
        Args:
            window_size: Size of the sliding window.
            device: Device to store data on ('cuda' or 'cpu').
        """
        super().__init__()
        self.window_size = window_size
        self._device = torch.device(device)
        self._window = torch.zeros(window_size, dtype=torch.float32, device=device)
        self._current_size = 0
        self._head = 0
    
    def add(self, value: torch.Tensor):
        """Add a new value to the window (GPU operation)."""
        idx = self._head
        self._window[idx] = value
        
        self._head = (self._head + 1) % self.window_size
        self._current_size = min(self._current_size + 1, self.window_size)
    
    def get_max(self):
        """Get the maximum value in the window (GPU operation)."""
        if self._current_size == 0:
            return torch.tensor(float('inf'), device=self._device)
        return self._window[:self._current_size].max()
    
    def get_min(self):
        """Get the minimum value in the window (GPU operation)."""
        if self._current_size == 0:
            return torch.tensor(0.0, device=self._device)
        return self._window[:self._current_size].min()
    
    def get_last(self) -> float:
        """Get the last added value (GPU operation)."""
        if self._current_size == 0:
            return torch.tensor(0.0, device=self._device)
        idx = (self._head - 1) % self.window_size
        return self._window[idx]


# ============== NumPy 版本 (CPU fallback) ==============

class CPUWindowedTrajectoryDistance:
    """CPU-based sliding window for trajectory distance tracking.
    
    This is a fallback implementation for CPU-only environments.
    """
    
    def __init__(self, window_size: int = 5, device: str = "cpu"):
        """Initialize the sliding window.
        
        Args:
            window_size: Size of the sliding window.
            device: Device string (kept for API compatibility).
        """
        from collections import deque
        
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.max_q = deque(maxlen=window_size)
        self.min_q = deque(maxlen=window_size)
    
    def add(self, value: float):
        """Add a new value to the window."""
        self.window.append(value)
        
        while self.max_q and self.max_q[-1] < value:
            self.max_q.pop()
        self.max_q.append(value)
        
        while self.min_q and self.min_q[-1] > value:
            self.min_q.pop()
        self.min_q.append(value)
    
    def get_max(self) -> float:
        """Get the maximum value in the window."""
        return self.max_q[0] if self.max_q else float('inf')
    
    def get_min(self) -> float:
        """Get the minimum value in the window."""
        return self.min_q[0] if self.min_q else 0.0
    
    def get_last(self) -> float:
        """Get the last added value."""
        return self.window[-1] if self.window else 0.0


def create_windowed_trajectory_distance(
    window_size: int = 5,
    device: str = "cuda",
    use_gpu: bool = True,
    backend: str = "jax"
) -> any:
    """Factory function to create the appropriate windowed trajectory distance.
    
    Args:
        window_size: Size of the sliding window.
        device: Device to store data on ('cuda' or 'cpu').
        use_gpu: Whether to use GPU acceleration.
        backend: Backend to use ('jax' or 'torch').
    
    Returns:
        An instance of the appropriate windowed trajectory distance class.
    """
    if not use_gpu:
        return CPUWindowedTrajectoryDistance(window_size=window_size, device=device)
    
    if backend == "jax":
        return GPUWindowedTrajectoryDistance(window_size=window_size, device=device)
    elif backend == "torch":
        return GPUWindowedTrajectoryDistanceTorch(window_size=window_size, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")
