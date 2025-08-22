"""
Memory management module for SeedVR2
Handles VRAM usage, cache management, and memory optimization

Extracted from: seedvr2.py (lines 373-405, 607-626, 1016-1044)
"""
\
import torch
import gc
import sys
import time
import psutil
from typing import Tuple, Dict, Any
from src.common.cache import Cache
from src.common.distributed import get_device

try:
    from comfy import model_management as mm
    COMFYUI_AVAILABLE = True
except:
    COMFYUI_AVAILABLE = False
    pass
    
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

def get_vram_usage() -> Tuple[float, float, float]:
    """
    Get current VRAM usage metrics for monitoring.
    Used for tracking memory consumption during processing.
    
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
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def get_ram_usage() -> Tuple[float, float, float, float]:
    """
    Get current RAM usage metrics for the current process.
    Provides accurate tracking of process-specific memory consumption.
    
    Returns:
        tuple: (process_gb, available_gb, total_gb, used_by_others_gb)
               Returns (0, 0, 0, 0) if psutil not available
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
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    
    
# Global cache for OS libraries (initialized once)
_os_memory_lib = None


def clear_memory(debug=None, full=False, force=True) -> None:
    """
    Clear memory caches with two-tier approach for optimal performance.
    
    Args:
        debug: Debug instance for logging (optional)
        force: If True, always clear. If False, only clear when <15% free
        full: If True, perform full cleanup including GC and OS operations.
              If False (default), only perform minimal GPU cache clearing.
    
    Two-tier approach:
        - Minimal mode (full=False): GPU cache operations (~1-5ms)
          Used for frequent calls during batch processing
        - Full mode (full=True): Complete cleanup with GC and OS operations (~10-50ms)
          Used at key points like model switches or final cleanup
    """
    global _os_memory_lib
    
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
    cleanup_mode = "full" if full else "minimal"
    if debug:
        debug.log(f"Clearing memory caches ({cleanup_mode})...", category="cleanup")
    
    # ===== MINIMAL OPERATIONS (Always performed) =====
    # Step 1: Clear GPU caches - Fast operations (~1-5ms)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.mps.is_available():
        torch.mps.empty_cache()
    
    # ===== FULL OPERATIONS (Only when full=True) =====
    if full:
        # Step 2: Clear PyTorch internal caches
        if hasattr(torch, '_C'):
            try:
                torch._C._clear_cache()
            except:
                pass
        
        # Step 3: Full garbage collection (expensive ~5-20ms)
        gc.collect(2)
        
        # Step 4: Return memory to OS (platform-specific, ~5-30ms)
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
        except:
            # OS-specific memory operations are optional
            pass
            

def manage_vae_device(runner, target_device: str, preserve_vram: bool = False, 
                     debug=None, reason: str = None) -> bool:
    """
    Manage VAE device placement with intelligent movement and logging.
    
    Args:
        runner: Runner instance containing the VAE
        target_device: Target device ('cuda:0', 'cpu', etc.)
        preserve_vram: Whether preserve_vram mode is active
        debug: Debug instance for logging
        reason: Optional custom reason for the movement
        
    Returns:
        bool: True if VAE was moved, False if already on target device
    """
    if not hasattr(runner, 'vae') or runner.vae is None:
        return False
    
    # Get current VAE device
    current_device = next(runner.vae.parameters()).device if hasattr(runner.vae, 'parameters') else None
    if current_device is None:
        return False
    
    # Normalize device strings for comparison
    target_type = target_device.split(':')[0] if ':' in target_device else target_device
    current_type = str(current_device.type)
    
    # Skip if already on target device
    if current_type == target_type:
        return False
    
    # Determine reason for movement
    if reason:
        reason = reason
    elif preserve_vram:
        reason = "preserve_vram"
    else:
        reason = "inference requirement"
    
    # Start timer based on direction
    timer_name = "vae_to_gpu" if target_type != 'cpu' else "vae_to_cpu"
    if debug:
        debug.start_timer(timer_name)
    
    # Log the movement
    if debug:
        if target_type == 'cpu':
            debug.log(f"Moving VAE to CPU ({reason})", category="general")
        else:
            debug.log(f"Moving VAE from {current_type} to {target_device} ({reason})", category="memory")
    
    # Move VAE
    runner.vae = runner.vae.to(target_device)
    
    # End timer
    if debug:
        if target_type == 'cpu':
            debug.end_timer(timer_name, "VAE moved to CPU")
        else:
            debug.end_timer(timer_name, "VAE moved to GPU")
    
    return True


def reset_vram_peak(debug) -> None:
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

def preinitialize_rope_cache(runner, debug) -> None:
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
                            clear_memory(debug=debug, full=True, force=True)
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


def clear_rope_lru_caches(model) -> int:
    """Clear ALL LRU caches from RoPE modules"""
    cleared_count = 0
    
    if model is None:
        return 0
    
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'get_axial_freqs') and hasattr(module.get_axial_freqs, 'cache_clear'):
                module.get_axial_freqs.cache_clear()
                cleared_count += 1
    except AttributeError:
        # Model structure already damaged, skip
        pass
    
    return cleared_count


def complete_model_deletion(model):
    """Completely delete a model and free all its memory"""
    if model is None:
        return
    
    try:
        # Move to CPU first
        model.to("cpu")
        
        # Clear parameters and buffers recursively and release storage
        def clear_recursive(m):
            # Process children first
            for child in m.children():
                clear_recursive(child)
            
            # Clear parameters and release storage
            if hasattr(m, '_parameters'):
                for param_name, param in list(m._parameters.items()):
                    if param is not None:
                        param.data = param.data.cpu()
                        param.grad = None
                        # Release underlying storage
                        if param.data.numel() > 0:
                            param.data.set_()
            
            # Clear buffers and release storage
            if hasattr(m, '_buffers'):
                for buffer_name, buffer in list(m._buffers.items()):
                    if buffer is not None:
                        buffer.data = buffer.data.cpu()
                        # Release underlying storage
                        if buffer.data.numel() > 0:
                            buffer.data.set_()
        
        clear_recursive(model)
        
        # Clear all module dicts but keep the structure
        if hasattr(model, 'modules'):
            for module in model.modules():
                # Clear custom attributes but keep PyTorch internals
                if hasattr(module, '__dict__'):
                    keys_to_delete = []
                    for key in module.__dict__.keys():
                        # Keep PyTorch internal attributes
                        if not key.startswith('_') or key.startswith('_original_'):
                            keys_to_delete.append(key)
                    for key in keys_to_delete:
                        try:
                            delattr(module, key)
                        except:
                            pass
        
        # Now clear the model's dict
        if hasattr(model, '__dict__'):
            # Clear everything except PyTorch internals
            keys_to_delete = []
            for key in model.__dict__.keys():
                if not key in ['_modules', '_parameters', '_buffers', 'training']:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                try:
                    delattr(model, key)
                except:
                    pass
    except AttributeError:
        # Model already partially cleaned, that's OK
        pass
    
    # Final cleanup - now we can clear everything
    if hasattr(model, '__dict__'):
        model.__dict__.clear()

def clear_all_caches(runner, debug, offload_vae=False) -> int:
    """
    Aggressively clear all caches from runner and model.
    Optimized to only process what's necessary.
    
    Args:
        runner: The runner instance to clear caches from
        debug: Optional Debug instance for logging
        offload_vae: If True, also moves VAE to CPU and clears its caches
    """
    if not runner:
        return 0
    
    cleaned_items = 0
    
    # Early exit if no caches to clear
    has_cache = hasattr(runner, 'cache') and hasattr(runner.cache, 'cache')
    if not has_cache and not hasattr(runner, 'dit') and not (offload_vae and hasattr(runner, 'vae')):
        return 0
    
    # Clear main runner cache efficiently
    if has_cache and runner.cache.cache:
        cache_entries = len(runner.cache.cache)
        
        # Process all cache items to properly free memory
        for key, value in list(runner.cache.cache.items()):
            if torch.is_tensor(value):
                # Force deallocation of tensor storage
                if value.is_cuda or value.is_mps:
                    value.data = value.data.cpu()
                value.grad = None
                if value.numel() > 0:
                    value.data.set_()  # Release underlying storage
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if torch.is_tensor(item):
                        if item.is_cuda or item.is_mps:
                            item.data = item.data.cpu()
                        item.grad = None
                        if item.numel() > 0:
                            item.data.set_()
        
        # Clear the cache after processing
        runner.cache.cache.clear()
        cleaned_items += cache_entries
        debug.log(f"Cleared {cache_entries} cache entries", category="success")
    
    # Clear any accumulated state in blocks
    if hasattr(runner, 'dit'):
        model = runner.dit
        if hasattr(model, 'dit_model'):
            model = model.dit_model

        # Clear RoPE LRU caches
        rope_caches_cleared = clear_rope_lru_caches(model)
        cleaned_items += rope_caches_cleared
        if rope_caches_cleared > 0:
            debug.log(f"Cleared {rope_caches_cleared} RoPE LRU caches", category="success")
            
        # Clear block attributes if needed
        if hasattr(model, 'blocks'):
            block_attrs_cleared = 0
            
            # Define PyTorch's essential attributes that must NOT be deleted
            essential_attrs = {
                '_modules', '_parameters', '_buffers', 
                '_forward_hooks', '_forward_pre_hooks', 
                '_backward_hooks', '_backward_pre_hooks',
                '_state_dict_hooks', '_state_dict_pre_hooks',
                '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks',
                '_non_persistent_buffers_set', '_version',
                '_is_full_backward_hook', 'training',
                '_original_forward',  # BlockSwap attribute
                '_is_io_wrapped',     # BlockSwap attribute
                '_block_idx',         # BlockSwap attribute
            }
            
            for idx, block in enumerate(model.blocks):
                # Get all attributes that look like caches
                attrs_to_remove = []
                for attr_name in list(block.__dict__.keys()):
                    # Only remove cache-like attributes, not essential PyTorch attributes
                    if (attr_name not in essential_attrs and 
                        ('cache' in attr_name or 
                         'temp' in attr_name or 
                         (attr_name.startswith('_') and 
                          not attr_name.startswith('__') and 
                          attr_name not in essential_attrs))):
                        attrs_to_remove.append(attr_name)
                
                # Remove the identified attributes
                for attr_name in attrs_to_remove:
                    try:
                        delattr(block, attr_name)
                        block_attrs_cleared += 1
                    except AttributeError:
                        pass  # Already deleted or doesn't exist
                            
            if block_attrs_cleared > 0:
                debug.log(f"Cleared {block_attrs_cleared} temporary attributes from blocks", category="success")
    
    # Clear any temporary attributes that might accumulate
    temp_attrs = ['_temp_cache', '_block_cache', '_swap_cache', '_generation_cache',
                  '_rope_cache', '_intermediate_cache', '_backward_cache']
    
    # Check both runner and model for these attributes
    for obj in [runner, getattr(runner, 'dit', None)]:
        if obj is None:
            continue
            
        # Handle wrapped models
        if hasattr(obj, 'dit_model'):
            obj = obj.dit_model
            
        for attr in temp_attrs:
            if hasattr(obj, attr):
                delattr(obj, attr)
                cleaned_items += 1
                debug.log(f"Cleared {attr} from {type(obj).__name__}", category="success")
    
    # Handle VAE offloading if requested
    if offload_vae and hasattr(runner, 'vae') and runner.vae is not None:
        # Clear intermediate tensors BEFORE moving to CPU (more efficient)
        vae_caches_cleared = 0
        for module in runner.vae.modules():
            # Clear module-specific caches
            for cache_attr in ['_temp_cache', '_intermediate_cache']:
                if hasattr(module, cache_attr):
                    delattr(module, cache_attr)
                    vae_caches_cleared += 1
            
            # Clear any CUDA tensors in module attributes
            for attr_name in list(vars(module).keys()):
                attr = getattr(module, attr_name, None)
                if torch.is_tensor(attr) and (attr.is_cuda or attr.is_mps):
                    # Move tensor to CPU if it's not a parameter/buffer
                    if attr_name not in module._parameters and attr_name not in module._buffers:
                        setattr(module, attr_name, attr.cpu())
                        vae_caches_cleared += 1
        
        if vae_caches_cleared > 0:
            debug.log(f"Cleared {vae_caches_cleared} VAE caches", category="success")
        
        # Now move VAE to CPU using helper
        manage_vae_device(runner, 'cpu', preserve_vram=True, debug=debug)
        cleaned_items += vae_caches_cleared
                
    # Final memory cleanup
    clear_memory(debug=debug, full=True, force=True)

    return cleaned_items
    