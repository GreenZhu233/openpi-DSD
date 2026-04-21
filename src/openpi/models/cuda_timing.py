"""CUDA kernel latency measurement utilities for PyTorch and JAX."""

import time
from typing import Callable, Optional
from contextlib import contextmanager

import torch
import jax
import jax.numpy as jnp


# ============== PyTorch 版本 ==============

class CudaTimerTorch:
    """PyTorch-based CUDA kernel timing using torch.cuda.Events.
    
    This provides accurate GPU-only timing that excludes CPU overhead.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize the timer.
        
        Args:
            device: CUDA device to use.
        """
        self._device = torch.device(device)
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None
    
    def start(self):
        """Start timing. Call this before the operation you want to measure."""
        torch.cuda.synchronize(self._device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds.
        """
        if self._end_event is not None:
            self._end_event.record()
            torch.cuda.synchronize(self._device)
        
        if self._start_event is not None and self._end_event is not None:
            return self._start_event.elapsed_time(self._end_event)
        return 0.0
    
    @contextmanager
    def timed(self):
        """Context manager for timing a block of code.
        
        Usage:
            timer = CudaTimerTorch()
            with timer.timed() as result:
                # code to time
                pass
            print(f"Elapsed: {result} ms")
        """
        self.start()
        yield lambda: self.stop()
        self.stop()


def measure_cuda_latency_torch(func: Callable, *args, device: str = "cuda", **kwargs) -> float:
    """Measure CUDA kernel latency for a PyTorch function.
    
    Args:
        func: Function to measure.
        *args: Arguments to pass to the function.
        device: CUDA device to use.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Elapsed time in milliseconds.
    """
    timer = CudaTimerTorch(device=device)
    timer.start()
    result = func(*args, **kwargs)
    if isinstance(result, torch.Tensor):
        torch.cuda.synchronize(device)
    return timer.stop()


# ============== JAX 版本 ==============

class CudaTimerJax:
    """JAX-based CUDA kernel timing.
    
    Uses JAX's internal timing capabilities and block_until_ready()
    for accurate GPU timing.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize the timer.
        
        Args:
            device: JAX device to use (only 'cuda' supported for GPU timing).
        """
        self._device = jax.devices(device)[0] if device in ['cuda', 'cpu'] else jax.devices()[0]
        self._elapsed_ms: Optional[float] = None
    
    def start(self):
        """Start timing. For JAX, this is a no-op as we measure on stop."""
        pass
    
    def stop(self, result: jnp.ndarray) -> float:
        """Stop timing and return elapsed time in milliseconds.
        
        Args:
            result: The result tensor from the operation. This will be
                   synchronized to measure actual GPU execution time.
        
        Returns:
            Elapsed time in milliseconds.
        """
        start = time.perf_counter()
        result.block_until_ready()
        end = time.perf_counter()
        return (end - start) * 1000.0
    
    @contextmanager
    def timed(self, result_placeholder=None):
        """Context manager for timing a block of code.
        
        Note: For JAX, this is less useful as we need the result tensor
        to properly synchronize. Consider using measure_cuda_latency_jax instead.
        """
        start = time.perf_counter()
        yield lambda end_result: self._measure_elapsed(start, end_result)
    
    def _measure_elapsed(self, start: float, result: jnp.ndarray) -> float:
        """Measure elapsed time from start to when result is ready."""
        result.block_until_ready()
        end = time.perf_counter()
        return (end - start) * 1000.0


def measure_cuda_latency_jax(func: Callable, *args, **kwargs) -> float:
    """Measure CUDA kernel latency for a JAX function.
    
    Args:
        func: Function to measure.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Elapsed time in milliseconds.
    """
    result = func(*args, **kwargs)
    
    # Handle pmap or other distributed operations
    if isinstance(result, tuple):
        # For multiple outputs, synchronize the first one
        result_to_sync = result[0] if len(result) > 0 else None
    else:
        result_to_sync = result
    
    if isinstance(result_to_sync, jnp.ndarray):
        start = time.perf_counter()
        result_to_sync.block_until_ready()
        end = time.perf_counter()
        return (end - start) * 1000.0
    
    # If result is not a JAX array, just return CPU time
    return 0.0


# ============== 统一的 API ==============

class CudaTimer:
    """Unified CUDA timer that works with both PyTorch and JAX.
    
    Automatically detects the backend based on the tensor type.
    """
    
    def __init__(self):
        self._torch_timer = CudaTimerTorch()
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start timing."""
        self._torch_timer.start()
        self._start_time = time.perf_counter()
    
    def stop(self, result=None) -> float:
        """Stop timing and return elapsed time.
        
        Args:
            result: Optional result tensor (PyTorch or JAX). If provided,
                   will use GPU-synchronized timing.
        
        Returns:
            Elapsed time in milliseconds.
        """
        if result is None:
            # Use PyTorch timer
            return self._torch_timer.stop()
        
        if isinstance(result, torch.Tensor):
            return self._torch_timer.stop()
        elif isinstance(result, jnp.ndarray):
            start = self._start_time or time.perf_counter()
            result.block_until_ready()
            end = time.perf_counter()
            return (end - start) * 1000.0
        else:
            return self._torch_timer.stop()
    
    @contextmanager
    def timed(self):
        """Context manager for timing."""
        self.start()
        yield self.stop
        # Call stop without result to use PyTorch timer as fallback
        self.stop()


def time_function(func: Callable, *args, backend: str = "auto", **kwargs) -> tuple[float, any]:
    """Time a function execution on GPU.
    
    Args:
        func: Function to time.
        *args: Positional arguments.
        backend: 'torch', 'jax', or 'auto' to detect.
        **kwargs: Keyword arguments.
    
    Returns:
        Tuple of (elapsed_time_ms, result).
    """
    if backend == "auto":
        # Try to detect from function name or global state
        import torch
        import jax
        # Default to auto-detection based on tensor operations
        backend = "torch"  # Safe default
    
    if backend == "torch":
        timer = CudaTimerTorch()
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        return elapsed, result
    else:
        result = func(*args, **kwargs)
        if isinstance(result, jnp.ndarray):
            start = time.perf_counter()
            result.block_until_ready()
            elapsed = (time.perf_counter() - start) * 1000.0
            return elapsed, result
        return 0.0, result