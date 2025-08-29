# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

import os
import time
import torch
from typing import Tuple, Dict, Any

from src.utils.constants import get_base_cache_dir
from src.utils.downloads import download_weight
from src.utils.model_registry import get_available_models, DEFAULT_MODEL
from src.utils.constants import get_script_directory
from src.utils.debug import Debug
from src.core.model_manager import configure_runner
from src.core.generation import generation_loop
from src.optimization.memory_manager import (
    release_text_embeddings,
    complete_cleanup, 
    get_device_list
)

# Import ComfyUI progress reporting
from server import PromptServer

script_directory = get_script_directory()

class SeedVR2:
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    - Real-time progress reporting
    """
    
    def __init__(self):
        """Initialize SeedVR2 node"""
        self.runner = None
        self.text_pos_embeds = None
        self.text_neg_embeds = None
        self.current_model_name = ""
        self.debug = None
        self.last_batch_time = None


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define ComfyUI input parameter types and constraints
        
        Returns:
            Dictionary defining input parameters, types, and validation
        """
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": (get_available_models(), {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Model variants with different sizes and precisions. Models will automatically download on first use. Additional models can be added to the ComfyUI models folder."
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 2**32 - 1, 
                    "step": 1,
                    "tooltip": "Random seed for generation. Same seed = same output."
                }),
                "new_resolution": ("INT", {
                    "default": 1072, 
                    "min": 16, 
                    "max": 4320, 
                    "step": 16,
                    "tooltip": "Target resolution for the shortest edge. Maintains aspect ratio."
                }),
                "batch_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 4,
                    "tooltip": "Number of frames to process per batch (recommend 4n+1 format)"
                }),
            },
            "optional": {
                "block_swap_config": ("block_swap_config", {
                    "tooltip": "Optional BlockSwap configuration for additional VRAM savings"
                }),
                "extra_args": ("extra_args", {
                    "tooltip": "Configure extra args"
                }),
            }
        }
    
    # Define return types for ComfyUI
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images: torch.Tensor, model: str, seed: int, new_resolution: int, 
        batch_size: int, block_swap_config=None, extra_args=None) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling with progress reporting"""
        
        temporal_overlap = 0 
        
        if extra_args is None:
            tiled_vae = True
            vae_tile_size = 512
            vae_tile_overlap = 64
            preserve_vram = False
            cache_model = False
            enable_debug = False
            devices = get_device_list()
            device = devices[0]
        else:
            tiled_vae = extra_args["tiled_vae"]
            vae_tile_size = extra_args["vae_tile_size"]
            vae_tile_overlap = extra_args["vae_tile_overlap"]
            preserve_vram = extra_args["preserve_vram"]
            cache_model = extra_args["cache_model"]
            enable_debug = extra_args["enable_debug"]
            device = extra_args["device"]
        
        # Validate tiling parameters
        if vae_tile_overlap >= vae_tile_size:
            raise ValueError(f"VAE tile overlap ({vae_tile_overlap}) must be less than tile size ({vae_tile_size})")

        # Initialize or reuse debug instance based on enable_debug parameter with timestamps
        if self.debug is None:
            self.debug = Debug(enabled=enable_debug, show_timestamps=enable_debug)
        else:
            self.debug.enabled = enable_debug
        
        # Check if download succeeded
        if not download_weight(model, debug=self.debug):
            raise RuntimeError(
                f"Required files for {model} are not available. "
                "Please check the console output for manual download instructions."
            )

        cfg_scale = 1.0
        try:
            return self._internal_execute(images, model, seed, new_resolution, cfg_scale, 
                                        batch_size, tiled_vae, vae_tile_size, vae_tile_overlap, 
                                        preserve_vram, temporal_overlap, 
                                        cache_model, device, block_swap_config)
        except Exception as e:
            self.cleanup(cache_model=cache_model, debug=self.debug)
            raise e
        

    def cleanup(self, cache_model: bool = False, debug=None):
        """
        Cleanup runner and free memory
        """
        # Get debug from runner if not provided
        if debug is None and self.runner and hasattr(self.runner, 'debug'):
            debug = self.runner.debug
        
        # Consolidated cleanup function
        if self.runner:
            complete_cleanup(runner=self.runner, debug=debug, keep_models_in_ram=cache_model)
            
            if not cache_model:
                # Delete the runner completely
                del self.runner
                self.runner = None
        
        # Clear instance embeddings
        release_text_embeddings(
            self.text_pos_embeds, 
            self.text_neg_embeds,
            debug=debug,
            names=["text_pos_embeds", "text_neg_embeds"] if debug else None
        )
        
        self.text_pos_embeds = None
        self.text_neg_embeds = None
        
        if not cache_model:
            self.current_model_name = ""


    def _internal_execute(self, images, model, seed, new_resolution, cfg_scale, batch_size, 
                 tiled_vae, vae_tile_size, vae_tile_overlap,
                 preserve_vram, temporal_overlap, cache_model, device, block_swap_config):
        """Internal execution logic with progress tracking"""
        
        debug = self.debug
        
        debug.start_timer("total_execution")
        debug.log("━━━━━━━━━ Model Preparation ━━━━━━━━━", category="none")

        # Initial memory state
        debug.log_memory_state("Before model preparation", show_tensors=True, detailed_tensors=False)
        debug.start_timer("model_preparation")

        os.environ["LOCAL_RANK"] = 0 if device == "none" else device.split(":")[1]
        
        if self.runner is not None:
            current_model = getattr(self.runner, '_model_name', None)
            model_changed = current_model != model

            if model_changed and self.runner is not None:
                debug.log(f"Model changed from {current_model} to {model}, clearing cache...", category="cache")
                self.cleanup(
                    cache_model=False,
                    debug=debug,
                )

        # Configure runner
        debug.log("Configuring inference runner...", category="runner")
        
        self.runner = configure_runner(
            model, get_base_cache_dir(), preserve_vram, debug,
            cache_model=cache_model,
            block_swap_config=block_swap_config,
            vae_tiling_enabled=tiled_vae,
            vae_tile_size=(vae_tile_size, vae_tile_size),
            vae_tile_overlap=(vae_tile_overlap, vae_tile_overlap),
            cached_runner=self.runner if cache_model else None
        )

        self.current_model_name = model

        debug.end_timer("model_preparation", "Model preparation", force=True, show_breakdown=True)

        debug.log("", category="none", force=True)
        debug.log("Starting video upscaling generation...", category="generation", force=True)
        debug.start_timer("generation_loop")

        # Execute generation with debug
        sample = generation_loop(
            self.runner, images, cfg_scale, seed, new_resolution, 
            batch_size, preserve_vram, temporal_overlap, debug,
            progress_callback=self._progress_callback
        )
        
        debug.log("", category="none", force=True)
        debug.log("Video upscaling completed successfully!", category="generation", force=True)
        # Log performance summary before clearing
        if debug.enabled:            
            # Log BlockSwap summary if it was used
            if hasattr(self.runner, '_blockswap_active') and self.runner._blockswap_active:
                swap_summary = debug.get_swap_summary()
                if swap_summary and swap_summary.get('total_swaps', 0) > 0:
                    total_time = swap_summary.get('block_total_ms', 0) + swap_summary.get('io_total_ms', 0)
                    debug.log(f"BlockSwap overhead: {total_time:.2f}ms", category="blockswap")
                    debug.log(f"  Total swaps: {swap_summary['total_swaps']}", category="blockswap")
                    
                    # Show block swap details
                    if 'block_swaps' in swap_summary and swap_summary['block_swaps'] > 0:
                        avg_ms = swap_summary.get('block_avg_ms', 0)
                        total_ms = swap_summary.get('block_total_ms', 0)
                        min_ms = swap_summary.get('block_min_ms', 0)
                        max_ms = swap_summary.get('block_max_ms', 0)
                        
                        debug.log(f"  Block swaps: {swap_summary['block_swaps']} "
                                f"(avg: {avg_ms:.2f}ms, min: {min_ms:.2f}ms, max: {max_ms:.2f}ms, total: {total_ms:.2f}ms)", 
                                category="blockswap")
                        
                        # Show most frequently swapped block
                        if 'most_swapped_block' in swap_summary:
                            debug.log(f"  Most swapped: Block {swap_summary['most_swapped_block']} "
                                    f"({swap_summary['most_swapped_count']} times)", category="blockswap")

        debug.end_timer("generation_loop", "Video generation", show_breakdown=True)
        debug.log_memory_state("After video generation", detailed_tensors=False)
       
        debug.log("", category="none")
        debug.log("━━━━━━━━━ Final Cleanup ━━━━━━━━━", category="none")
        debug.start_timer("final_cleanup")
        
        # Perform cleanup (this already calls clear_memory internally)
        self.cleanup(cache_model=cache_model, debug=debug)
        
        # Ensure sample is on CPU (ComfyUI expects CPU tensors)
        if torch.is_tensor(sample) and sample.is_cuda:
            sample = sample.cpu()
        
        # Log final memory state after ALL cleanup is done
        debug.end_timer("final_cleanup", "Final cleanup", show_breakdown=True)
        debug.log_memory_state("After final cleanup", show_tensors=True, detailed_tensors=False)
        
        # Final timing summary
        debug.log("", category="none")
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")
        child_times = {
            "Model preparation": debug.timer_durations.get("model_preparation", 0),
            "Video generation": debug.timer_durations.get("generation_loop", 0),
            "Final cleanup": debug.timer_durations.get("final_cleanup", 0)
        }
        debug.end_timer("total_execution", "Total execution", show_breakdown=True, custom_children=child_times)
        debug.log("━━━━━━━━━━━━━━━━━━", category="none")
        
        # Clear history for next run (do this last, after all logging)
        debug.clear_history()
        
        return (sample,)

    def _progress_callback(self, batch_idx, total_batches, current_batch_frames, message=""):
        """Progress callback for generation loop"""
        
        # Calculate batch FPS
        batch_time = 0
        if self.last_batch_time is not None:
            batch_time = time.time() - self.last_batch_time
        elif "generation_loop" in self.debug.timers:
            batch_time = time.time() - self.debug.timers["generation_loop"]
        else:
            batch_time = self.debug.timer_durations.get("generation_loop", 0)
        batch_fps = current_batch_frames / batch_time if batch_time > 0 else 0.0
        self.debug.log(f"Batch {batch_idx} - FPS: {batch_fps:.2f} frames/sec", category="timing")
        self.last_batch_time = time.time()
        
        # Send numerical progress
        progress_value = int((batch_idx / total_batches) * 100)
        progress_data = {
            "value": progress_value,
            "max": 100,
            "node": "seedvr2_node"
        }
        PromptServer.instance.send_sync("progress", progress_data, None)

    def __del__(self):
        """Destructor"""
        try:
            # Store debug reference
            debug = self.debug if hasattr(self, 'debug') else None
            
            # Full cleanup
            if hasattr(self, 'cleanup'):
                self.cleanup(cache_model=False, debug=debug)
            
            # Clear all remaining references
            for attr in ['runner', 'text_pos_embeds', 'text_neg_embeds', 
                        'current_model_name', 'debug', 'last_batch_time']:
                if hasattr(self, attr):
                    delattr(self, attr)
        except:
            pass

class SeedVR2BlockSwap:
    """Configure block swapping to reduce VRAM usage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 36,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. Start with 16 and increase until OOM errors stop. 0=disabled",
                    },
                ),
                "offload_io_components": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload embeddings and I/O layers to CPU. Enable if you need additional VRAM savings beyond block swapping",
                    },
                ),
            }
        }

    RETURN_TYPES = ("block_swap_config",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = """Configure block swapping to reduce VRAM usage during video upscaling.

BlockSwap dynamically moves transformer blocks between GPU and CPU/RAM during inference, enabling large models to run on limited VRAM systems with minimal performance impact.

Configuration Guidelines:
    - blocks_to_swap=0: Disabled (fastest, highest VRAM usage)
    - blocks_to_swap=16: Balanced mode (moderate speed/VRAM trade-off)
    - blocks_to_swap=32-36: Maximum savings (slowest, lowest VRAM)

Advanced Options:
    - offload_io_components: Moves embeddings and I/O layers to CPU for additional VRAM savings (slower)

Performance Tips:
    - Start with blocks_to_swap=16 and increase until you no longer get OOM errors or decrease if you have spare VRAM
    - Enable offload_io_components if you still need additional VRAM savings
    - Note: Even if inference succeeds, you may still OOM during VAE decoding - combine BlockSwap with VAE tiling if needed (feature in development)
    - Combine with smaller batch_size for maximum VRAM savings

The actual memory savings depend on your specific model architecture and will be shown in the debug output when enabled.
    """

    def create_config(self, blocks_to_swap, offload_io_components):
        """Create BlockSwap configuration"""
        if blocks_to_swap == 0:
            return (None,)
        config = {
            "blocks_to_swap": blocks_to_swap,
            "offload_io_components": offload_io_components,
        }
        
        return (config,)
    
class SeedVR2ExtraArgs:
    """Configure extra args"""
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = get_device_list()
        return {
            "required": {
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Process VAE in tiles to reduce VRAM usage but slower with potential artifacts. Only enable if running out of memory."
                }),
                "vae_tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "step": 32,
                    "tooltip": "VAE tile size in pixels. Smaller = less VRAM but more seams/artifacts and slower. Larger = more VRAM but better quality and faster."
                }),
                "vae_tile_overlap": ("INT", {
                    "default": 64,
                    "min": 0,
                    "step": 32,
                    "tooltip": "Pixel overlap between tiles to reduce visible seams. Higher = better blending but slower processing."
                }),
                "preserve_vram": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models between steps to save VRAM. Slower but uses less memory."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model and VAE in RAM between runs. Speeds up batch processing."
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show detailed memory usage and timing information during generation."
                }),
                "device": (devices, {
                    "default": devices[0]
                }),
            }
        }
    
    RETURN_TYPES = ("extra_args",)
    FUNCTION = "create_config"
    CATEGORY = "SEEDVR2"
    DESCRIPTION = "Configure extra args."
    
    def create_config(self, tiled_vae, vae_tile_size, vae_tile_overlap, preserve_vram, cache_model, enable_debug, device):
        config = {
            "tiled_vae": tiled_vae,
            "vae_tile_size": vae_tile_size,
            "vae_tile_overlap": vae_tile_overlap,
            "preserve_vram": preserve_vram,
            "cache_model": cache_model,
            "enable_debug": enable_debug,
            "device": device,
        }
        
        return (config,)


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
    "SeedVR2BlockSwap": SeedVR2BlockSwap,
    "SeedVR2ExtraArgs": SeedVR2ExtraArgs,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
    "SeedVR2BlockSwap": "SeedVR2 BlockSwap Config",
    "SeedVR2ExtraArgs": "SeedVR2 Extra Args",
}

# Export version and metadata
__version__ = "2.0.0-modular"
__author__ = "SeedVR2 Team"
__description__ = "High-quality video upscaling using advanced diffusion models"

# Additional exports for introspection
__all__ = [
    'SeedVR2',
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS'
] 
