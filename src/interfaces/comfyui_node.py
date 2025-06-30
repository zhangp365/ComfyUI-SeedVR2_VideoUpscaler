# ComfyUI Node Interface
# Clean interface for SeedVR2 VideoUpscaler integration with ComfyUI
# Extracted from original seedvr2.py lines 1731-1812

from datetime import datetime
import os
import gc
import time
import torch
from typing import Tuple, Dict, Any

from src.utils.downloads import download_weight, get_base_cache_dir
from src.core.model_manager import configure_runner
from src.core.generation import generation_loop
from src.optimization.memory_manager import clear_rope_lru_caches, fast_model_cleanup
script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SeedVR2:
    """
    SeedVR2 Video Upscaler ComfyUI Node
    
    High-quality video upscaling using diffusion models with support for:
    - Multiple model variants (3B/7B, FP16/FP8)
    - Adaptive VRAM management
    - Advanced dtype compatibility
    - Optimized inference pipeline
    """
    
    def __init__(self):
        """Initialize SeedVR2 node"""
        self.runner = None
        self.text_pos_embeds = None
        self.text_neg_embeds = None
        self.current_model_name = ""


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
                "model": ([
                    "seedvr2_ema_3b_fp16.safetensors", 
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_fp16.safetensors",
                    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                ], {
                    "default": "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
                }),
                "seed": ("INT", {
                    "default": 100, 
                    "min": 0, 
                    "max": 5000, 
                    "step": 1,
                    "tooltip": "Random seed for generation reproducibility"
                }),
                "new_resolution": ("INT", {
                    "default": 1280, 
                    "min": 1, 
                    "max": 4320, 
                    "step": 1,
                    "tooltip": "Target width for upscaled video"
                }),
                "batch_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 2048, 
                    "step": 4,
                    "tooltip": "Number of frames to process per batch (recommend 4n+1 format)"
                }),
                "preserve_vram": ("BOOLEAN", {"default": False}),
            },
        }
    
    # Define return types for ComfyUI
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images: torch.Tensor, model: str, seed: int, new_resolution: int, 
        batch_size: int, preserve_vram: bool) -> Tuple[torch.Tensor]:
        """Execute SeedVR2 video upscaling"""
        
        temporal_overlap = 0 
        print(f"🔄 Preparing model: {model}")
        download_weight(model)
        debug = False
        cfg_scale = 1.0
        try:
            return self._internal_execute(images, model, seed, new_resolution, cfg_scale, batch_size, preserve_vram, temporal_overlap, debug)
        except Exception as e:
            self.cleanup(force_ram_cleanup=True)
            raise e



    def cleanup(self, force_ram_cleanup: bool = True):
        """Fast cleanup with minimal logging"""
        
        if self.runner:
            # Clear cache
            if hasattr(self.runner, 'cache') and hasattr(self.runner.cache, 'cache'):
                for key, value in list(self.runner.cache.cache.items()):
                    if hasattr(value, 'cpu'):
                        value.cpu()
                    if hasattr(value, 'detach'):
                        value.detach()
                    del value
                self.runner.cache.cache.clear()
            
            # Clear DiT model
            if hasattr(self.runner, 'dit') and self.runner.dit is not None:
                
                clear_rope_lru_caches(self.runner.dit)
                fast_model_cleanup(self.runner.dit)
                del self.runner.dit
                self.runner.dit = None
            
            # Clear VAE model  
            if hasattr(self.runner, 'vae') and self.runner.vae is not None:
                #from src.optimization.memory_manager import fast_model_cleanup
                fast_model_cleanup(self.runner.vae)
                del self.runner.vae
                self.runner.vae = None
            
            # Clear other components
            for component in ['sampler', 'sampling_timesteps', 'schedule', 'config']:
                if hasattr(self.runner, component):
                    setattr(self.runner, component, None)
            
            del self.runner
            self.runner = None
        
        # Clear embeddings
        if self.text_pos_embeds is not None:
            if hasattr(self.text_pos_embeds, 'cpu'):
                self.text_pos_embeds.cpu()
            del self.text_pos_embeds
            self.text_pos_embeds = None
        
        if self.text_neg_embeds is not None:
            if hasattr(self.text_neg_embeds, 'cpu'):
                self.text_neg_embeds.cpu()
            del self.text_neg_embeds
            self.text_neg_embeds = None
        
        self.current_model_name = ""
        
        # Fast RAM cleanup
        if force_ram_cleanup:
            from src.optimization.memory_manager import fast_ram_cleanup
            fast_ram_cleanup()


    def _internal_execute(self, images, model, seed, new_resolution, cfg_scale, batch_size, preserve_vram, temporal_overlap, debug):
        """Internal execution logic"""
        total_start_time = time.time()
        # Configure runner
        if debug:
            print("🔄 Configuring inference runner...")
        runner_start = time.time()
        self.runner = configure_runner(model, get_base_cache_dir(), preserve_vram, debug)
        if debug:
            print(f"🔄 Runner configuration time: {time.time() - runner_start:.2f}s")
        
        if debug:
            print("🚀 Starting video upscaling generation...")
        
        # Execute generation
        sample = generation_loop(
            self.runner, images, cfg_scale, seed, new_resolution, 
            batch_size, preserve_vram, temporal_overlap, debug
        )
        print(f"✅ Video upscaling completed successfully!")
        # Cleanup
        print(f"🔄 Total execution time: {time.time() - total_start_time:.2f}s")           
        self.cleanup(force_ram_cleanup=True)
        return (sample,)


    def __del__(self):
        """Destructor"""
        try:
            self.cleanup(force_ram_cleanup=True)
        except:
            pass


# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
}

# Human-readable node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
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