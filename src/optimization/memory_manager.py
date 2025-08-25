"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""

import torch
import gc
import sys
import time
import psutil
from typing import Tuple, Dict, Any, Optional, List, Union
from src.common.cache import Cache
from src.common.distributed import get_device
    

def get_device_list():
  devs = ["none"]
  try:
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
      devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
  except Exception:
    pass
  try:
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
      devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
  except Exception:
    pass
  if len(devs) > 1:
    return devs[1:]
  return devs
    

def get_basic_vram_info() -> Dict[str, Any]:
    """
    Get basic VRAM availability info (free and total memory).
    Used for capacity planning and initial checks.
    
    Returns:
        dict: {"free_gb": float, "total_gb": float} or {"error": str}
    """
    try:
        if torch.cuda.is_available():
            device = get_device()
            free_memory, total_memory = torch.cuda.mem_get_info(device)
        elif torch.mps.is_available():
            mem = psutil.virtual_memory()
            free_memory = mem.total - mem.used
            total_memory = mem.total
        else:
            return {"error": "No GPU backend available (CUDA/MPS)"}
        
        return {
            "free_gb": free_memory / (1024**3),
            "total_gb": total_memory / (1024**3)
        }
    except Exception as e:
        return {"error": f"Failed to get memory info: {str(e)}"}


# Initial VRAM check at module load
vram_info = get_basic_vram_info()
if "error" not in vram_info:
    backend = "MPS" if torch.mps.is_available() else "CUDA"
    print(f"📊 Initial {backend} memory: {vram_info['free_gb']:.2f}GB free / {vram_info['total_gb']:.2f}GB total")
else:
    print(f"⚠️ Memory check failed: {vram_info['error']} - No available backend!")


def get_vram_usage(debug: Optional[Any] = None) -> Tuple[float, float, float]:
    """
    Get current VRAM usage metrics for monitoring.
    Used for tracking memory consumption during processing.

    Args:
        debug: Optional debug instance for logging
    
    Returns:
        tuple: (allocated_gb, reserved_gb, max_allocated_gb)
               Returns (0, 0, 0) if no GPU available
    """
    try:
        if torch.cuda.is_available():
            device = get_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            return allocated, reserved, max_allocated
        elif torch.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            reserved = torch.mps.driver_allocated_memory() / (1024**3)
            max_allocated = allocated  # MPS doesn't track peak separately
            return allocated, reserved, max_allocated
    except Exception as e:
        if debug:
            debug.log(f"Failed to get VRAM usage: {e}", level="WARNING", category="memory", force=True)
    return 0.0, 0.0, 0.0


def get_ram_usage(debug: Optional[Any] = None) -> Tuple[float, float, float, float]:
    """
    Get current RAM usage metrics for the current process.
    Provides accurate tracking of process-specific memory consumption.
    
    Args:
        debug: Optional debug instance for logging
    
    Returns:
        tuple: (process_gb, available_gb, total_gb, used_by_others_gb)
               Returns (0, 0, 0, 0) if psutil not available or on error
    """
    try:
        if not psutil:
            return 0.0, 0.0, 0.0, 0.0
            
        # Get current process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        process_gb = process_memory.rss / (1024**3)
        
        # Get system memory
        sys_memory = psutil.virtual_memory()
        total_gb = sys_memory.total / (1024**3)
        available_gb = sys_memory.available / (1024**3)
        
        # Calculate memory used by other processes
        # This is the CORRECT calculation:
        total_used_gb = total_gb - available_gb  # Total memory used by ALL processes
        used_by_others_gb = max(0, total_used_gb - process_gb)  # Subtract current process
        
        return process_gb, available_gb, total_gb, used_by_others_gb
        
    except Exception as e:
        if debug:
            debug.log(f"Failed to get RAM usage: {e}", level="WARNING", category="memory", force=True)
        return 0.0, 0.0, 0.0, 0.0
    
    
# Global cache for OS libraries (initialized once)
_os_memory_lib = None


def clear_memory(debug: Optional[Any] = None, deep: bool = False, force: bool = True) -> None:
    """
    Clear memory caches with two-tier approach for optimal performance.
    
    Args:
        debug: Debug instance for logging (optional)
        force: If True, always clear. If False, only clear when <15% free
        deep: If True, perform deep cleanup including GC and OS operations.
              If False (default), only perform minimal GPU cache clearing.
    
    Two-tier approach:
        - Minimal mode (deep=False): GPU cache operations (~1-5ms)
          Used for frequent calls during batch processing
        - Deep mode (deep=True): Complete cleanup with GC and OS operations (~10-50ms)
          Used at key points like model switches or final cleanup
    """
    global _os_memory_lib
    
    # Start timer for entire operation
    if debug:
        debug.start_timer("memory_clear")

    # Check if we should clear based on memory pressure
    if not force:
        should_clear = False
        
        # Use existing function for memory info
        mem_info = get_basic_vram_info()
        
        if "error" not in mem_info:
            # Check VRAM/MPS memory pressure (15% free threshold)
            free_ratio = mem_info["free_gb"] / mem_info["total_gb"]
            if free_ratio < 0.15:
                should_clear = True
                if debug:
                    backend = "MPS" if torch.mps.is_available() else "VRAM"
                    debug.log(f"{backend} pressure: {mem_info['free_gb']:.1f}GB free of {mem_info['total_gb']:.1f}GB", category="memory")
        
        # For non-MPS systems, also check system RAM separately
        if not should_clear and not torch.mps.is_available():
            mem = psutil.virtual_memory()
            if mem.available < mem.total * 0.15:
                should_clear = True
                if debug:
                    debug.log(f"RAM pressure: {mem.available/(1024**3):.1f}GB free of {mem.total/(1024**3):.1f}GB", category="memory")
        
        if not should_clear:
            return
    
    # Determine cleanup level
    cleanup_mode = "deep" if deep else "minimal"
    if debug:
        debug.log(f"Clearing memory caches ({cleanup_mode})...", category="cleanup")
    
    # ===== MINIMAL OPERATIONS (Always performed) =====
    # Step 1: Clear GPU caches - Fast operations (~1-5ms)
    if debug:
        debug.start_timer("gpu_cache_clear")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.mps.is_available():
        torch.mps.empty_cache()
    
    if debug:
        debug.end_timer("gpu_cache_clear", "GPU cache clearing")

    # ===== DEEP OPERATIONS (Only when deep=True) =====
    if deep:
        # Step 2: Deep garbage collection (expensive ~5-20ms)
        if debug:
            debug.start_timer("garbage_collection")

        gc.collect(2)

        if debug:
            debug.end_timer("garbage_collection", "Garbage collection")

        # Step 3: Return memory to OS (platform-specific, ~5-30ms)
        if debug:
            debug.start_timer("os_memory_release")

        try:
            if sys.platform == 'linux':
                # Linux: malloc_trim
                import ctypes  # Import only when needed
                if _os_memory_lib is None:
                    _os_memory_lib = ctypes.CDLL("libc.so.6")
                _os_memory_lib.malloc_trim(0)
                
            elif sys.platform == 'win32':
                # Windows: Trim working set
                import ctypes  # Import only when needed
                if _os_memory_lib is None:
                    _os_memory_lib = ctypes.windll.kernel32
                handle = _os_memory_lib.GetCurrentProcess()
                _os_memory_lib.SetProcessWorkingSetSize(handle, -1, -1)
                
            elif torch.mps.is_available():
                # macOS with MPS
                import ctypes  # Import only when needed
                import ctypes.util
                if _os_memory_lib is None:
                    libc_path = ctypes.util.find_library('c')
                    if libc_path:
                        _os_memory_lib = ctypes.CDLL(libc_path)
                
                if _os_memory_lib:
                    _os_memory_lib.sync()
        except Exception as e:
            if debug:
                debug.log(f"Failed to perform OS memory operations: {e}", level="WARNING", category="memory", force=True)

        if debug:
            debug.end_timer("os_memory_release", "OS memory release")
    
    # End overall timer
    if debug:
        debug.end_timer("memory_clear", "clear_memory() completion")


def reset_vram_peak(debug: Optional[Any]) -> None:
    """
    Reset VRAM peak memory statistics for fresh tracking.
    """
    debug.log("Resetting VRAM peak memory statistics", category="memory")
    try:
        if torch.cuda.is_available():
            device = get_device()
            torch.cuda.reset_peak_memory_stats(device)
        # MPS doesn't support peak memory reset
    except Exception as e:
        debug.log(f"Failed to reset peak memory stats: {e}", level="WARNING", category="memory", force=True)

def preinitialize_rope_cache(runner: Any, debug: Optional[Any]) -> None:
    """
    🚀 Pre-initialize RoPE cache to avoid OOM at first launch
    
    Args:
        runner: The model runner containing DiT and VAE models
        debug: Optional Debug instance
    """
    
    debug.log("Pre-initializing RoPE cache to avoid OOM...", category="setup")

    try:
        # Create dummy tensors to simulate common shapes
        # Format: [batch, channels, frames, height, width] for vid_shape
        # Format: [batch, seq_len] for txt_shape
        common_shapes = [
            # Common video resolutions
            (torch.tensor([[1, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 1 frame, 77 tokens
            (torch.tensor([[4, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 4 frames
            (torch.tensor([[5, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 5 frames (4n+1 format)
            (torch.tensor([[1, 4, 4]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # Higher resolution
        ]
        
        # Create mock cache for pre-initialization
            
        temp_cache = Cache()
        
        # Access RoPE modules in DiT (recursive search)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(getattr(child, 'get_freqs')):
                    rope_modules.append((name, child))
            return rope_modules
        
        rope_modules = find_rope_modules(runner.dit)
        
        # Pre-calculate for each RoPE module found
        for name, rope_module in rope_modules:
            # Temporarily move module to CPU if necessary
            original_device = next(rope_module.parameters()).device if list(rope_module.parameters()) else torch.device('cpu')
            rope_module.to('cpu')
            
            try:
                for vid_shape, txt_shape in common_shapes:
                    cache_key = f"720pswin_by_size_bysize_{tuple(vid_shape[0].tolist())}_sd3.mmrope_freqs_3d"
                    
                    def compute_freqs():
                        try:
                            # Calculate with reduced dimensions to avoid OOM
                            with torch.no_grad():
                                # Detect RoPE module type
                                module_type = type(rope_module).__name__
                                
                                if module_type == 'NaRotaryEmbedding3d':
                                    # NaRotaryEmbedding3d: only takes shape (vid_shape)
                                    return rope_module.get_freqs(vid_shape.cpu())
                                else:
                                    # Standard RoPE: takes vid_shape and txt_shape
                                    return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())
                                    
                        except Exception as e:
                            debug.log(f"Failed pre-initializing RoPE cache for {cache_key}: {e}", level="ERROR", category="setup", force=True)
                            # Return empty tensors as fallback
                            clear_memory(debug=debug, deep=True, force=True)
                            time.sleep(1)

                            return torch.zeros(1, 64)
                    
                    # Store in cache
                    temp_cache(cache_key, compute_freqs)
                
            except Exception as e:
                debug.log(f"Error in module {name}: {e}", level="ERROR", category="setup", force="True")
            finally:
                # Restore to original device
                rope_module.to(original_device)
        
        # Copy temporary cache to runner cache
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache
        
    except Exception as e:
        debug.log(f"Error during RoPE pre-init: {e}", level="ERROR", category="setup", force=True)
        debug.log("Model will work but could have OOM at first launch", level="WARNING", category="info", force=True)


def clear_rope_lru_caches(model: Optional[torch.nn.Module], debug: Optional[Any] = None) -> int:
    """
    Clear ALL LRU caches from RoPE modules.
    
    Args:
        model: PyTorch model to clear caches from
        debug: Optional debug instance for logging
        
    Returns:
        Number of caches cleared
    """
    if model is None:
        return 0
    
    cleared_count = 0
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
                try:
                    module.get_axial_freqs.cache_clear()
                    cleared_count += 1
                except Exception as e:
                    if debug:
                        debug.log(f"Failed to clear RoPE LRU cache for module {name}: {e}", level="WARNING", category="memory", force=True)
    except (AttributeError, RuntimeError) as e:
        if debug:
            debug.log(f"Failed to iterate model modules for RoPE LRU cache clearing: {e}", level="WARNING", category="memory", force=True)
    
    return cleared_count


def release_tensor_memory(tensor: Optional[torch.Tensor]) -> None:
    """Release tensor memory properly without CPU allocation"""
    if tensor is not None and torch.is_tensor(tensor):
        if tensor.is_cuda or tensor.is_mps:
            # Release GPU memory directly without CPU transfer
            if tensor.numel() > 0:
                tensor.data.set_()
        tensor.grad = None


def release_text_embeddings(*embeddings: torch.Tensor, debug: Optional[Any] = None, names: Optional[List[str]] = None) -> None:
    """
    Release memory for text embeddings
    
    Args:
        *embeddings: Variable number of embedding tensors to release
        debug: Optional debug instance for logging
        names: Optional list of names for logging
    """
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            release_tensor_memory(embedding)
            if debug and names and i < len(names):
                debug.log(f"Cleaned up {names[i]}", category="cleanup")


def release_model_memory(model: Optional[torch.nn.Module], debug: Optional[Any] = None) -> None:
    """
    Release all GPU/MPS memory from model in-place without CPU transfer.
    
    Args:
        model: PyTorch model to release memory from
        debug: Optional debug instance for logging
    """
    if model is None:
        return
    
    try:
        # Clear gradients first
        model.zero_grad(set_to_none=True)
        
        # Release GPU memory directly without CPU transfer
        released_params = 0
        released_buffers = 0
        
        for param in model.parameters():
            if param.is_cuda or param.is_mps:
                if param.numel() > 0:
                    param.data.set_()
                    released_params += 1
                param.grad = None
                
        for buffer in model.buffers():
            if buffer.is_cuda or buffer.is_mps:
                if buffer.numel() > 0:
                    buffer.data.set_()
                    released_buffers += 1
        
        if debug and (released_params > 0 or released_buffers > 0):
            debug.log(f"Released memory from {released_params} params and {released_buffers} buffers", category="success")
                
    except (AttributeError, RuntimeError) as e:
        if debug:
            debug.log(f"Failed to release model memory: {e}", level="WARNING", category="memory", force=True)


def manage_model_device(model, target_device: str, model_name: str = "model",
                       preserve_vram: bool = False, debug=None, reason: str = None) -> bool:
    """
    Unified model device management with intelligent movement and logging.
    
    Args:
        model: The model to move
        target_device: Target device ('cuda:0', 'cpu', etc.)
        model_name: Name for logging (e.g., "VAE", "DiT")
        preserve_vram: Whether preserve_vram mode is active
        debug: Debug instance for logging
        reason: Optional custom reason for the movement
        
    Returns:
        bool: True if model was moved, False if already on target device
    """
    if model is None:
        return False
    
    # Get current device
    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        return False
    
    # Normalize device strings for comparison
    target_type = target_device.split(':')[0] if ':' in target_device else target_device
    current_type = str(current_device.type)
    
    # Skip if already on target device
    if current_type == target_type:
        return False
    
    # Determine reason for movement
    if not reason:
        reason = "preserve_vram" if preserve_vram else "inference requirement"
    
    # Start timer based on direction
    timer_name = f"{model_name.lower()}_to_{'gpu' if target_type != 'cpu' else 'cpu'}"
    if debug:
        debug.start_timer(timer_name)
    
    # Log the movement
    if debug:
        if target_type == 'cpu':
            debug.log(f"Moving {model_name} to CPU ({reason})", category="general")
        else:
            debug.log(f"Moving {model_name} from {current_type} to {target_device} ({reason})", category="memory")
    
    # Move model and clear gradients
    model.to(target_device)
    model.zero_grad(set_to_none=True)
    
    # Clear VAE memory buffers when moving to CPU
    if target_type == 'cpu' and model_name == "VAE":
        cleared_count = 0
        for module in model.modules():
            if hasattr(module, 'memory') and module.memory is not None:
                if torch.is_tensor(module.memory) and (module.memory.is_cuda or module.memory.is_mps):
                    module.memory = None
                    cleared_count += 1
        if cleared_count > 0 and debug:
            debug.log(f"Cleared {cleared_count} VAE memory buffers", category="success")
    
    # End timer
    if debug:
        if target_type == 'cpu':
            debug.end_timer(timer_name, f"{model_name} moved to CPU")
        else:
            debug.end_timer(timer_name, f"{model_name} moved to GPU")
    
    return True


def clear_runtime_caches(runner: Any, debug: Optional[Any]) -> int:
    """
    Clear all runtime caches and temporary attributes.
    """
    if not runner:
        return 0
    
    if debug:
        debug.start_timer("runtime_cache_clear")
    
    cleaned_items = 0
    
    # 1. Clear main runner cache
    if hasattr(runner, 'cache') and hasattr(runner.cache, 'cache'):
        if debug:
            debug.start_timer("runner_cache_clear")

        cache_entries = len(runner.cache.cache)
        
        # Properly release tensor memory
        for key, value in list(runner.cache.cache.items()):
            if torch.is_tensor(value):
                release_tensor_memory(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        release_tensor_memory(item)
        
        runner.cache.cache.clear()
        cleaned_items += cache_entries

        if debug:
            debug.end_timer("runner_cache_clear", f"Clearing main runner cache entries")

        if cache_entries > 0:
            debug.log(f"Cleared {cache_entries} runtime cache entries", category="success")
    
    # 2. Clear RoPE caches
    if hasattr(runner, 'dit'):
        if debug:
            debug.start_timer("rope_cache_clear")

        model = runner.dit
        if hasattr(model, 'dit_model'):  # Handle wrapper
            model = model.dit_model
        
        rope_cleared = clear_rope_lru_caches(model=model, debug=debug)
        cleaned_items += rope_cleared
        if debug:
            debug.end_timer("rope_cache_clear", "Clearing RoPE LRU caches")

        if rope_cleared > 0:
            debug.log(f"Cleared {rope_cleared} RoPE LRU caches", category="success")
    
    # 3. Clear temporary attributes
    temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                  '_rope_cache', '_intermediate_cache', '_backward_cache']
    
    for obj in [runner, getattr(runner, 'dit', None), getattr(runner, 'vae', None)]:
        if obj is None:
            continue
            
        actual_obj = obj.dit_model if hasattr(obj, 'dit_model') else obj
        
        for attr in temp_attrs:
            if hasattr(actual_obj, attr):
                delattr(actual_obj, attr)
                cleaned_items += 1

    if debug:
        debug.end_timer("runtime_cache_clear", f"clear_runtime_caches() completion")

    return cleaned_items


def complete_cleanup(runner: Any, debug: Optional[Any], keep_models_in_ram: bool = False) -> None:
    """
    Complete cleanup of runner and all components.
    """
    if not runner:
        return
    
    cleanup_type = "partial cleanup (keeping models in RAM)" if keep_models_in_ram else "full cleanup"
    if debug:
        debug.log(f"Starting {cleanup_type}", category="cleanup")
    
    # 1. Clean BlockSwap if active
    if hasattr(runner, "_blockswap_active") and runner._blockswap_active:
        # Import here to avoid circular dependency
        from src.optimization.blockswap import cleanup_blockswap
        cleanup_blockswap(runner, keep_state_for_cache=keep_models_in_ram)
    
    # 2. Clear all runtime caches
    clear_runtime_caches(runner=runner, debug=debug)
    
    if keep_models_in_ram:
        # 3a. Partial cleanup - move models to CPU but keep structure
        blockswap_configured = hasattr(runner, '_block_swap_config') and runner._block_swap_config
        
        if hasattr(runner, 'dit') and not blockswap_configured:
            manage_model_device(model=runner.dit, target_device='cpu', model_name="DiT", preserve_vram=True, debug=debug, reason="model caching")
        elif blockswap_configured and debug:
            debug.log("Skipping DiT movement - BlockSwap configuration preserved", category="general")
        
        if hasattr(runner, 'vae'):
            manage_model_device(model=runner.vae, target_device='cpu', model_name="VAE", preserve_vram=True, debug=debug, reason="model caching")
    else:
        # 3b. Full cleanup - release memory and delete
        if hasattr(runner, 'dit'):
            release_model_memory(model=runner.dit, debug=debug)
            runner.dit = None
            debug.log("DiT model deleted", category="cleanup")
        
        if hasattr(runner, 'vae'):
            release_model_memory(model=runner.vae, debug=debug)
            runner.vae = None
            debug.log("VAE model deleted", category="cleanup")
        
        # Clear other components
        for component in ['sampler', 'sampling_timesteps', 'schedule', 'config']:
            if hasattr(runner, component):
                setattr(runner, component, None)
    
    # 4. Final memory cleanup
    clear_memory(debug=debug, deep=True, force=True)
    debug.log(f"Completed {cleanup_type}", category="success")