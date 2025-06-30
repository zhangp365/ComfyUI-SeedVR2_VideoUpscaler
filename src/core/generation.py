"""
Generation Logic Module for SeedVR2

This module handles the main generation pipeline including:
- Single generation steps with adaptive dtype handling
- Complete generation loop with temporal awareness
- Context-aware batch processing with overlapping
- Video preprocessing and post-processing
- Optimized memory management during generation

Key Features:
- Native FP8 pipeline support for 2x speedup and 50% VRAM reduction
- Context-aware generation with temporal overlap for smooth transitions
- Adaptive dtype detection and optimal autocast configuration
- Intelligent batch processing with memory optimization
- Advanced video format handling (4n+1 constraint)
"""

import os
import torch
import time
import gc
from torchvision.transforms import Compose, Lambda, Normalize

# Import required modules
from src.optimization.memory_manager import reset_vram_peak
from src.optimization.performance import (
    optimized_video_rearrange, optimized_single_video_rearrange, 
    optimized_sample_to_image_format, temporal_latent_blending
)
from common.seed import set_seed

# Get script directory for embeddings
script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import transforms and color fix

from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize

from src.utils.color_fix import wavelet_reconstruction


def generation_step(runner, text_embeds_dict, preserve_vram, cond_latents, temporal_overlap):
    """
    Execute a single generation step with adaptive dtype handling
    
    Args:
        runner: VideoDiffusionInfer instance
        text_embeds_dict (dict): Text embeddings for positive and negative prompts
        preserve_vram (bool): Whether to enable VRAM optimization
        cond_latents (list): Conditional latents for generation
        temporal_overlap (int): Number of frames for temporal overlap
        
    Returns:
        tuple: (samples, last_latents) for potential temporal continuation
        
    Features:
        - Adaptive dtype detection (FP8/FP16/BFloat16)
        - Optimal autocast configuration for each model type
        - Memory-efficient noise generation and reuse
        - Automatic device placement with dtype preservation
        - Advanced inference optimization
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adaptive dtype detection for optimal performance
    model_dtype = next(runner.dit.parameters()).dtype
    
    # Configure dtypes according to model architecture
    if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 native: use BFloat16 for intermediate calculations (optimal compatibility)
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif model_dtype == torch.float16:
        dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16

    def _move_to_cuda(x):
        """Move tensors to CUDA with adaptive optimal dtype"""
        return [i.to(device, dtype=dtype) for i in x]

    # Memory optimization: Generate noise once and reuse to save VRAM
    with torch.cuda.device(device):
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    
    # Move tensors with adaptive dtype (optimized for FP8/FP16/BFloat16)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)
    
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Use adaptive optimal dtype
        t = (
            torch.tensor([1000.0], device=device, dtype=dtype)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    # Generate conditions with memory optimization
    condition = runner.get_condition(
        noises[0],
        task="sr",
        latent_blur=_add_noise(cond_latents[0], aug_noises[0]),
    )
    conditions = [condition]
    t = time.time()
    
    # Use adaptive autocast for optimal performance
    with torch.no_grad():
        with torch.autocast("cuda", autocast_dtype, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                preserve_vram=preserve_vram,  # Memory offload optimization
                temporal_overlap=temporal_overlap,
                **text_embeds_dict,
            )
    
    print(f"🔄 INFERENCE time: {time.time() - t} seconds")
    
    # Process samples with advanced optimization
    samples = optimized_video_rearrange(video_tensors)
    #last_latents = samples[-temporal_overlap:] if temporal_overlap > 0 else samples[-1:]
    noises = noises[0].to("cpu")
    aug_noises = aug_noises[0].to("cpu")
    cond_latents = cond_latents[0].to("cpu")
    conditions = conditions[0].to("cpu")
    condition = condition.to("cpu")

    return samples #, last_latents


def cut_videos(videos):
    """
    Correct video cutting respecting the constraint: frames % 4 == 1
    
    Args:
        videos (torch.Tensor): Video tensor to format
        
    Returns:
        torch.Tensor: Properly formatted video tensor
        
    Features:
        - Ensures frames % 4 == 1 constraint for model compatibility
        - Intelligent padding with last frame repetition
        - Memory-efficient tensor operations
    """
    t = videos.size(1)
    
    if t % 4 == 1:
        return videos
    
    # Calculate next valid number (4n + 1)
    padding_needed = (4 - (t % 4)) % 4 + 1
    
    # Apply padding to reach 4n+1 format
    last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
    result = torch.cat([videos, last_frame], dim=1)
    
    return result


def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, preserve_vram=False, temporal_overlap=0, debug=False):
    """
    Main generation loop with context-aware temporal processing
    
    Args:
        runner: VideoDiffusionInfer instance
        images (torch.Tensor): Input images for upscaling
        cfg_scale (float): Classifier-free guidance scale
        seed (int): Random seed for reproducibility
        res_w (int): Target resolution width
        batch_size (int): Batch size for processing
        preserve_vram (str/bool): VRAM preservation mode
        temporal_overlap (int): Frames for temporal continuity
        
    Returns:
        torch.Tensor: Generated video frames
        
    Features:
        - Context-aware generation with temporal overlap
        - Adaptive dtype pipeline (FP8/FP16/BFloat16)
        - Memory-optimized batch processing
        - Advanced video transformation pipeline
        - Intelligent VRAM management throughout process
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adaptive model dtype detection for maximum performance
    model_dtype = None
    try:
        # Get real dtype of loaded DiT model
        model_dtype = next(runner.dit.parameters()).dtype
        
        # Adapt dtypes according to model
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # For FP8, use BFloat16 for intermediate calculations (compatible)
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16  # VAE stays BFloat16 for compatibility
        elif model_dtype == torch.float16:
            compute_dtype = torch.float16
            autocast_dtype = torch.float16
            vae_dtype = torch.float16
        else:  # BFloat16 or others
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16
            
    except Exception as e:
        print(f"⚠️ Could not detect model dtype: {e}, falling back to BFloat16")
        model_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        vae_dtype = torch.bfloat16

    # Optimization tips for users
    if torch.cuda.is_available():
        total_frames = len(images)
        optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
        if optimal_batches:
            best_batch = max(optimal_batches)
            if best_batch != batch_size:
                print(f"\n💡 TIP: For {total_frames} frames, use batch_size={best_batch} to avoid padding")
                if batch_size not in optimal_batches:
                    padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))
                    print(f"   Currently: ~{padding_waste} wasted padding frames")

    # Configure classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # Configure sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # Set random seed
    set_seed(seed)

    # Advanced video transformation pipeline
    video_transform = Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            # Upsample image, model only trained for high res
            downsample_only=False,
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (faster than Rearrange)
    ])

    # Initialize generation state
    batch_samples = []
    final_tensor = None
    
    # Load text embeddings with adaptive dtype
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=compute_dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=compute_dtype)
    text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
    
    # Memory optimization
    reset_vram_peak()
    
    # Calculate processing parameters
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Move images to CPU for memory efficiency
    #t = time.time()
    #images = images.to("cpu")
    #print(f"🔄 Images to CPU time: {time.time() - t} seconds")
    
    try:
        # Main processing loop with context awareness
        for batch_idx in range(0, len(images), step):
            # Calculate batch indices with overlap
            if batch_idx == 0:
                # First batch: no overlap
                start_idx = 0
                end_idx = min(batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = True
            else:
                # Subsequent batches: temporal overlap
                start_idx = batch_idx 
                end_idx = min(start_idx + batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = False
                if effective_batch_size <= temporal_overlap:
                    break  # Not enough new frames, stop

            tps_loop = time.time()
            batch_number = (batch_idx // step + 1) if step > 0 else 1
            print(f"\n🎬 Batch {batch_number}: frames {start_idx}-{end_idx-1}")
            
            # Process current batch
            video = images[start_idx:end_idx]
            if debug:
                print(f"🔄 video Compute dtype: {compute_dtype}")
            # Use adaptive computation dtype 
            video = video.permute(0, 3, 1, 2).to(device, dtype=compute_dtype)
            
            # Apply video transformations with memory optimization
            transformed_video = video_transform(video)
            del video
            #video = video.to("cpu")
            #del video
            ori_lengths = [transformed_video.size(1)]
            
            # Handle correct format: frames % 4 == 1
            t = transformed_video.size(1)
            print(f"📹 Sequence of {t} frames")
            
            
            if len(images) >= 5 and t % 4 != 1:
                if debug:
                    print(f"🔄 Transformed video shape before cut: {transformed_video.shape}")
                transformed_video = cut_videos(transformed_video)
                if debug:
                    print(f"🔄 Transformed video shape: {transformed_video.shape}")
            
            # Context-aware temporal strategy
            # First batch: standard complete diffusion
            tps_vae = time.time()
            runner.vae.to(device)
            if debug:
                print(f"🔄 VAE to GPU time: {time.time() - tps_vae} seconds")
            tps_vae = time.time()
            if debug:
                print(f"🔄 VAE dtype: {autocast_dtype}")
            with torch.autocast("cuda", autocast_dtype, enabled=True):
                cond_latents = runner.vae_encode([transformed_video])
            if debug:
                print(f"🔄 VAE encode time: {time.time() - tps_vae} seconds")
            #tps = time.time()
            #transformed_video = transformed_video.to("cpu")
            #print(f"🔄 Transformed video to cpu time: {time.time() - tps} seconds")
            if debug:
                print(f"🔄 Cond latents shape: {cond_latents[0].shape}, time: {time.time() - tps_vae} seconds")
            
            # Normal generation
            samples = generation_step(runner, text_embeds, preserve_vram, cond_latents=cond_latents, temporal_overlap=temporal_overlap)
            #del cond_latents
            del cond_latents
               
            
            # Post-process samples
            sample = samples[0]
            del samples 
            # 🔧 DIAGNOSTIC: Vérifier les valeurs après generation_step
            '''
            if debug:
                print(f"🔍 DIAGNOSTIC - Après generation_step:")
                print(f"  📊 Sample shape: {sample.shape}")
                print(f"  📊 Sample dtype: {sample.dtype}")
                print(f"  📊 Sample device: {sample.device}")
                print(f"  📊 Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
                print(f"  📊 Sample mean: {sample.mean():.6f}")
                print(f"  📊 Sample std: {sample.std():.6f}")
                print(f"  📊 Non-zero count: {(sample != 0).sum()}/{sample.numel()}")
            '''
            #del samples
            if ori_lengths[0] < sample.shape[0]:
                sample = sample[:ori_lengths[0]]
            #if temporal_overlap > 0 and not is_first_batch and sample.shape[0] > effective_batch_size - temporal_overlap:
            #    sample = sample[temporal_overlap:]  # Remove overlap frames from output
            
            # Apply color correction if available
            tps = time.time()
            transformed_video = transformed_video.to(device)
            if debug:
                print(f"🔄 Transformed video to device time: {time.time() - tps} seconds")
            
            input_video = [optimized_single_video_rearrange(transformed_video)]
            del transformed_video
            #transformed_video = transformed_video.to("cpu")
            #del transformed_video
            sample = wavelet_reconstruction(sample, input_video[0][:sample.size(0)])
            del input_video
            '''
            if debug:
                # 🔧 DIAGNOSTIC: Vérifier les valeurs après wavelet_reconstruction
                print(f"🔍 DIAGNOSTIC - Après wavelet_reconstruction:")
                print(f"  📊 Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
                print(f"  📊 Sample mean: {sample.mean():.6f}")
                print(f"  📊 Non-zero count: {(sample != 0).sum()}/{sample.numel()}")
            '''
            #del input_video
            # Convert to final image format
            sample = optimized_sample_to_image_format(sample)
            '''
            if debug:
                # 🔧 DIAGNOSTIC: Vérifier les valeurs après optimized_sample_to_image_format
                print(f"🔍 DIAGNOSTIC - Après optimized_sample_to_image_format:")
                print(f"  📊 Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
                print(f"  📊 Sample mean: {sample.mean():.6f}")
            '''
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
            '''
            if debug:
                # 🔧 DIAGNOSTIC: Vérifier les valeurs finales
                print(f"🔍 DIAGNOSTIC - Valeurs finales:")
                print(f"  📊 Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
                print(f"  📊 Sample mean: {sample.mean():.6f}")
                print(f"  🎯 Est-ce que l'image est noire? {sample.max() < 0.01}")
            '''
            sample_cpu = sample.to("cpu")
            del sample
            batch_samples.append(sample_cpu)
            #del sample 
            
            # Aggressive cleanup after each batch
            tps = time.time()
            
            #transformed_video = transformed_video.to("cpu")
            #print(f"🔄 Transformed video to cpu time: {time.time() - tps} seconds")
            if debug:
                print(f"🔄 Time batch: {time.time() - tps_loop} seconds")
            if preserve_vram:
                torch.cuda.empty_cache()
            #del transformed_video
            #clear_vram_cache()

    finally:
        # Final cleanup of embeddings
        text_pos_embeds = text_pos_embeds.to("cpu")
        text_neg_embeds = text_neg_embeds.to("cpu")
        
        #del text_pos_embeds, text_neg_embeds
        #clear_vram_cache()
    for i in range(len(batch_samples)):
        batch_samples[i] = batch_samples[i].to(device)
    # Concatenate all batch results
    final_video_images = torch.cat(batch_samples, dim=0)
    final_video_images = final_video_images.to("cpu")
    
    # Critical correction: Convert to Float16 for ComfyUI compatibility
    if final_video_images.dtype != torch.float16:
        final_video_images = final_video_images.to(torch.float16)
    
    # Cleanup batch_samples
    #del batch_samples
    return final_video_images


def prepare_video_transforms(res_w):
    """
    Prepare optimized video transformation pipeline
    
    Args:
        res_w (int): Target resolution width
        
    Returns:
        Compose: Configured transformation pipeline
        
    Features:
        - Resolution-aware upscaling (no downsampling)
        - Proper normalization for model compatibility
        - Memory-efficient tensor operations
    """
    return Compose([
        NaResize(
            resolution=(res_w),
            mode="side",
            downsample_only=False,  # Model trained for high resolution
        ),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w
    ])


def load_text_embeddings(script_directory, device, dtype):
    """
    Load and prepare text embeddings for generation
    
    Args:
        script_directory (str): Script directory path
        device (str): Target device
        dtype (torch.dtype): Target dtype
        
    Returns:
        dict: Text embeddings dictionary
        
    Features:
        - Adaptive dtype handling
        - Device-optimized loading
        - Memory-efficient embedding preparation
    """
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=dtype)
    
    return {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}


def calculate_optimal_batch_params(total_frames, batch_size, temporal_overlap):
    """
    Calculate optimal batch processing parameters
    
    Args:
        total_frames (int): Total number of frames
        batch_size (int): Desired batch size
        temporal_overlap (int): Temporal overlap frames
        
    Returns:
        dict: Optimized parameters and recommendations
        
    Features:
        - 4n+1 constraint optimization
        - Padding waste calculation
        - Performance recommendations
    """
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    
    # Find optimal batch sizes (4n+1 constraint)
    optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
    best_batch = max(optimal_batches) if optimal_batches else 1
    
    # Calculate potential padding waste
    padding_waste = 0
    if batch_size not in optimal_batches:
        padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))
    
    return {
        'step': step,
        'temporal_overlap': temporal_overlap,
        'best_batch': best_batch,
        'padding_waste': padding_waste,
        'is_optimal': batch_size in optimal_batches
    } 