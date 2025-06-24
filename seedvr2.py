# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
import torch
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
import folder_paths
import gc
import time
from huggingface_hub import snapshot_download, hf_hub_download

# Import SafeTensors avec fallback
try:
    from safetensors.torch import save_file as save_safetensors_file
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
    #print("✅ SafeTensors available")
except ImportError:
    print("⚠️ SafeTensors not available, recommended install: pip install safetensors")
    SAFETENSORS_AVAILABLE = False

from .data.image.transforms.divisible_crop import DivisibleCrop
from .data.image.transforms.na_resize import NaResize
from .data.video.transforms.rearrange import Rearrange
script_directory = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_directory, "./projects/video_diffusion_sr/color_fix.py")):
    from .projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not avaliable!')
from torchvision.transforms import Compose, Lambda, Normalize
#from torchvision.io.video import read_video

#print(script_directory)
folder_paths.add_model_folder_path("seedvr2", os.path.join(folder_paths.models_dir, "SEEDVR2"))
base_cache_dir = os.path.join(folder_paths.models_dir, "SEEDVR2")
#print(script_directory)
from .projects.video_diffusion_sr.infer import VideoDiffusionInfer

from .common.seed import set_seed
import os


def configure_runner(model):
    from .common.config import load_config, create_object
    from omegaconf import DictConfig, OmegaConf
    import importlib
    
    if "7b" in model:
        config_path = os.path.join(script_directory, './configs_7b', 'main.yaml')
    else:
        config_path = os.path.join(script_directory, './configs_3b', 'main.yaml')
    config = load_config(config_path)
    
    # Create the __object__ section directly in code to avoid YAML path issues
    if "7b" in model:
        # Try different import paths for 7B model
        model_paths = [
            "custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.dit.nadit",
            "ComfyUI.custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.dit.nadit",
            "models.dit.nadit"
        ]
    else:
        # Try different import paths for 3B model
        model_paths = [
            "custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.dit_v2.nadit",
            "ComfyUI.custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.dit_v2.nadit", 
            "models.dit_v2.nadit"
        ]
    
    # Try each path until one works by actually testing the import
    working_path = None
    for path in model_paths:
        try:
            # Test if we can actually import from this path
            importlib.import_module(path)
            working_path = path
            #print(f"Using model path: {path}")
            break
        except ImportError:
            continue
    
    if working_path is None:
        raise ImportError(f"Could not find working import path for model. Tried: {model_paths}")
    
    # Create the complete __object__ section in config
    config.dit.model.__object__ = DictConfig({
        "path": working_path,
        "name": "NaDiT", 
        "args": "as_params"
    })
    
    # Handle VAE model path dynamically
    vae_paths = [
        "custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.video_vae_v3.modules.attn_video_vae",
        "ComfyUI.custom_nodes.ComfyUI-SeedVR2_VideoUpscaler.models.video_vae_v3.modules.attn_video_vae",
        "models.video_vae_v3.modules.attn_video_vae"
    ]
    
    working_vae_path = None
    for path in vae_paths:
        try:
            importlib.import_module(path)
            working_vae_path = path
            #print(f"Using VAE path: {path}")
            break
        except ImportError:
            continue
    
    if working_vae_path is None:
        raise ImportError(f"Could not find working import path for VAE. Tried: {vae_paths}")
    
    # Load VAE config and merge with main config
    vae_config_path = os.path.join(script_directory, 'models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    vae_config = OmegaConf.load(vae_config_path)
    
    # Add the __object__ section to VAE config
    # Both 3B and 7B models use VideoAutoencoderKLWrapper
    # Get the downsample factors from the VAE config (they're required parameters)
    spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    
    # Set gradient_checkpoint as a direct parameter (not in __object__)
    vae_config.spatial_downsample_factor = spatial_downsample_factor
    vae_config.temporal_downsample_factor = temporal_downsample_factor
    vae_config.freeze_encoder = False

    if "7b" in model:
        # 7B model: gradient checkpointing disabled (as per config comment)
        vae_config.gradient_checkpoint = False
        vae_config.__object__ = DictConfig({
            "path": working_vae_path,
            "name": "VideoAutoencoderKLWrapper",
            "args": "as_params"  # Important: use as_params to pass individual parameters
        })
    else:
        # 3B model: gradient checkpointing enabled
        vae_config.gradient_checkpoint = False
        vae_config.__object__ = DictConfig({
            "path": working_vae_path,
            "name": "VideoAutoencoderKLWrapper",
            "args": "as_params"  # Important: use as_params to pass individual parameters
        })
    
    # Merge VAE config with main config
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)
    
    # Create runner without distributed setup
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    # Set device for single GPU usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models directly without distributed framework
    checkpoint_path = os.path.join(base_cache_dir, f'./{model}')
    
    # Configure models directly WITHOUT decorators
    configure_dit_model_inference(runner, device, checkpoint_path, config)
    configure_vae_model_inference(runner, config, device)
    
    # Set memory limit if available
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    
    preinitialize_rope_cache(runner)
    
    return runner

def load_quantized_state_dict(checkpoint_path, device="cpu", keep_native_fp8=True):
    """Load state dict from SafeTensors or PyTorch with FP8 native support"""
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors required to load this model. Install with: pip install safetensors")
        #print(f"🔄 Loading SafeTensors: {checkpoint_path}")
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.pth'):
        #print(f"🔄 Loading PyTorch: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    # 🚀 OPTIMISATION FP8: Garder le format natif pour performance maximale
    fp8_detected = False
    fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2) if hasattr(torch, 'float8_e4m3fn') else ()
    
    if fp8_types:
        # Vérifier si le modèle contient des tenseurs FP8
        for key, tensor in state.items():
            if hasattr(tensor, 'dtype') and tensor.dtype in fp8_types:
                fp8_detected = True
                break
    
    if fp8_detected:
        if keep_native_fp8:
            #print("🚀 FP8 model detected - Keeping native FP8 format for optimal performance!")
            #print("   ⚡ Benefits: ~50% less VRAM, ~2x faster inference")
            #print("   🎯 Using native FP8 precision for maximum speed")
            return state
        else:
            #print("🎯 FP8 model detected - Converting to BFloat16 for compatibility...")
            converted_state = {}
            converted_count = 0
            
            for key, tensor in state.items():
                if hasattr(tensor, 'dtype') and tensor.dtype in fp8_types:
                    # Convertir FP8 → BFloat16 seulement si demandé
                    converted_state[key] = tensor.to(torch.bfloat16)
                    converted_count += 1
                else:
                    converted_state[key] = tensor
            
            #print(f"   ✅ Converted {converted_count} FP8 tensors to BFloat16")
            return converted_state
    
    return state

def save_safetensors(state_dict, output_path):
    """Sauvegarder un state_dict en format SafeTensors"""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors requis. Installez avec: pip install safetensors")
    
    # Convertir tous les tenseurs en format compatible SafeTensors
    compatible_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # SafeTensors nécessite des tenseurs contigus
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            compatible_state_dict[key] = tensor
        else:
            # Ignorer les non-tenseurs (SafeTensors ne supporte que les tenseurs)
            print(f"⚠️ Ignored {key} (non-tensor): {type(tensor)}")
    
    save_safetensors_file(compatible_state_dict, output_path)

def configure_dit_model_inference(runner, device, checkpoint, config):
    """Configure DiT model for inference without distributed decorators"""
    #print("Entering configure_dit_model (inference)")
    
    from .common.config import create_object
    
    # Create dit model on CPU first
    with torch.device(device):
        runner.dit = create_object(config.dit.model)
    runner.dit.set_gradient_checkpointing(config.dit.gradient_checkpoint)

    if checkpoint:
        # Load model directly with format detection
        
        if checkpoint.endswith('.safetensors'):
            if 'fp8_e4m3fn' in checkpoint:
                print("🚀 Loading FP8 SafeTensors model")
            elif 'fp16' in checkpoint:
                print("🔄 Loading FP16 SafeTensors model")
            else:
                 print("🔄 Loading SafeTensors model")
        elif checkpoint.endswith('.pth'):
            print("🔄 Loading PyTorch model")
        else:
            raise ValueError(f"Unsupported format. Expected .safetensors or .pth, got: {checkpoint}")
        
        # 🚀 OPTIMISATION: Garder le FP8 natif par défaut pour performance maximale
        keep_native_fp8 = True  # Permet d'optimiser les modèles FP8
        
        # Support for quantized models with native FP8
        state = load_quantized_state_dict(checkpoint, device, keep_native_fp8=keep_native_fp8)
        runner.dit.load_state_dict(state, strict=True, assign=True)
        #print(f"Loading pretrained ckpt from {checkpoint}")
        #print(f"Loading info: {loading_info}")

    # Move to target device (preserve native dtype)
    # runner.dit.to(device)
    
    # 🚀 WRAPPER UNIVERSEL: Appliquer le wrapper à TOUS les modèles pour forcer RoPE en BFloat16
    #model_dtype = next(runner.dit.parameters()).dtype
    #print(f"🎯 Applying compatibility wrapper for all models (RoPE → BFloat16)...")
    runner.dit = FP8CompatibleDiT(runner.dit)
    #runner.dit.to("cpu")
    '''
    if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        print(f"✅ FP8 Native Pipeline with Universal Wrapper Active!")
        print(f"   🎯 Model dtype: {model_dtype} (FP8 native)")
        print(f"   🔄 Automatic BFloat16 conversion for calculations")
        print(f"   🛡️ RoPE forced to BFloat16 for compatibility")
        print(f"   ⚡ Expected speedup: ~2x vs FP16")
        print(f"   💾 Expected VRAM saving: ~50% vs FP16")
    elif model_dtype == torch.float16:
        print(f"✅ FP16 Pipeline with Universal Wrapper Active!")
        print(f"   🎯 Model dtype: {model_dtype} (FP16 native)")
        print(f"   🛡️ RoPE forced to BFloat16 for compatibility")
        print(f"   ⚡ Native FP16 performance")
    else:
        print(f"✅ BFloat16 Pipeline with Universal Wrapper Active!")
        print(f"   🎯 Model dtype: {model_dtype}")
        print(f"   🛡️ RoPE already BFloat16 compatible")
    '''


def configure_vae_model_inference(runner, config, device):
    """Configure VAE model for inference without distributed decorators"""
    #print("Entering configure_vae_model (inference)")
    
    from .common.config import create_object
    
    # Create vae model
    dtype = getattr(torch, config.vae.dtype)
    runner.vae = create_object(config.vae.model)
    runner.vae.requires_grad_(False).eval()
    runner.vae.to(device=device, dtype=dtype)

    # Load vae checkpoint with dynamic path resolution
    checkpoint_path = config.vae.checkpoint
    
    # Try different possible paths
    possible_paths = [
        checkpoint_path,  # Original path
        os.path.join("ComfyUI", checkpoint_path),  # With ComfyUI prefix
        os.path.join(script_directory, checkpoint_path),  # Relative to script directory
        os.path.join(script_directory, "..", "..", checkpoint_path),  # From ComfyUI root
    ]
    
    vae_checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            vae_checkpoint_path = path
            print(f"Found VAE checkpoint at: {vae_checkpoint_path}")
            break
    
    if vae_checkpoint_path is None:
        raise FileNotFoundError(f"VAE checkpoint not found. Tried paths: {possible_paths}")
    
    # 🚀 Support SafeTensors pour VAE (FP16 et FP8)
    if vae_checkpoint_path.endswith('.safetensors'):
        print(f"🚀 Loading VAE SafeTensors: {vae_checkpoint_path}")
        """
        if "fp16" in vae_checkpoint_path:
            print("   📊 FP16 VAE SafeTensors - optimized format")
        if 'fp8_e4m3fn' in vae_checkpoint_path:
            print("   📊 FP8 E4M3FN VAE detected - 50% smaller, optimized for inference")
        """
        # 🚀 Utiliser load_quantized_state_dict pour TOUS les SafeTensors (FP16 et FP8)
        # Cette fonction gère correctement le format SafeTensors
        if "fp8_e4m3fn" in vae_checkpoint_path:
            state = load_quantized_state_dict(vae_checkpoint_path, device, keep_native_fp8=True)
        else:
            # Pour FP16 SafeTensors, désactiver keep_native_fp8
            state = load_quantized_state_dict(vae_checkpoint_path, device, keep_native_fp8=False)
    else:
        print(f"🔄 Loading VAE PyTorch: {vae_checkpoint_path}")
        state = torch.load(vae_checkpoint_path, map_location=device, mmap=True)
    
    runner.vae.load_state_dict(state)

    # Set causal slicing
    if hasattr(runner.vae, "set_causal_slicing") and hasattr(config.vae, "slicing"):
        runner.vae.set_causal_slicing(**config.vae.slicing)

def get_vram_usage():
    """Obtenir l'utilisation VRAM actuelle (allouée, réservée, pic)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        return allocated, reserved, max_allocated
    return 0, 0, 0

def clear_vram_cache():
    """Nettoyer le cache VRAM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def reset_vram_peak():
    """Reset le compteur de pic VRAM pour un nouveau tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def check_vram_safety(operation_name="Opération", required_gb=2.0):
    """Vérifier si on a assez de VRAM pour continuer en sécurité"""
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated, reserved, peak = get_vram_usage()
        available = total_vram - allocated
        
        if available < required_gb:
            return False
        return True
    return True

def preinitialize_rope_cache(runner):
    """🚀 Pré-initialiser le cache RoPE pour éviter l'OOM au premier lancement"""
    #print("🔄 Pré-initialisation du cache RoPE...")
    
    # Sauvegarder l'état actuel des modèles
    dit_device = next(runner.dit.parameters()).device
    vae_device = next(runner.vae.parameters()).device
    
    try:
        # Temporairement déplacer les modèles sur CPU pour libérer VRAM
        #print("  📦 Déplacement temporaire des modèles sur CPU...")
        #runner.dit.to("cpu")
        #runner.vae.to("cpu")
        #clear_vram_cache()
        
        # Créer des tenseurs factices pour simuler les shapes communes
        # Format: [batch, channels, frames, height, width] pour vid_shape
        # Format: [batch, seq_len] pour txt_shape
        common_shapes = [
            # Résolutions communes pour vidéo
            (torch.tensor([[1, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 1 frame, 77 tokens
            (torch.tensor([[4, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 4 frames
            (torch.tensor([[5, 3, 3]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # 5 frames (4n+1 format)
            (torch.tensor([[1, 4, 4]], dtype=torch.long), torch.tensor([[77]], dtype=torch.long)),    # Plus grande résolution
        ]
        
        # Créer un cache mock pour la pré-initialisation
        from .common.cache import Cache
        temp_cache = Cache()
        
        # Pré-calculer les fréquences sur CPU avec des dimensions réduites
        #print("  🧮 Calcul des fréquences RoPE communes...")
        
        # Accéder aux modules RoPE dans DiT (recherche récursive)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(getattr(child, 'get_freqs')):
                    rope_modules.append((name, child))
            return rope_modules
        
        rope_modules = find_rope_modules(runner.dit)
        #print(f"  🎯 Trouvé {len(rope_modules)} modules RoPE")
        
        # Pré-calculer pour chaque module RoPE trouvé
        for name, rope_module in rope_modules:
            #print(f"    ⚙️ Pré-calcul pour {name}...")
            
            # Déplacer temporairement le module sur CPU si nécessaire
            original_device = next(rope_module.parameters()).device if list(rope_module.parameters()) else torch.device('cpu')
            rope_module.to('cpu')
            
            try:
                for vid_shape, txt_shape in common_shapes:
                    cache_key = f"720pswin_by_size_bysize_{tuple(vid_shape[0].tolist())}_sd3.mmrope_freqs_3d"
                    
                    def compute_freqs():
                        try:
                            # Calcul avec dimensions réduites pour éviter OOM
                            with torch.no_grad():
                                # Détecter le type de RoPE module
                                module_type = type(rope_module).__name__
                                
                                if module_type == 'NaRotaryEmbedding3d':
                                    # NaRotaryEmbedding3d: ne prend que shape (vid_shape)
                                    return rope_module.get_freqs(vid_shape.cpu())
                                else:
                                    # RoPE standard: prend vid_shape et txt_shape
                                    return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())
                                    
                        except Exception as e:
                            print(f"      ⚠️ Échec pour {cache_key}: {e}")
                            # Retourner des tenseurs vides comme fallback
                            return torch.zeros(1, 64)
                    
                    # Stocker dans le cache
                    temp_cache(cache_key, compute_freqs)
                    #print(f"      ✅ Cached: {cache_key}")
                
            except Exception as e:
                print(f"    ❌ Error module {name}: {e}")
            finally:
                # Remettre sur le device original
                rope_module.to(original_device)
        
        # Copier le cache temporaire vers le cache du runner
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache
        
        #print("  ✅ Cache RoPE pre-init success!")
        
    except Exception as e:
        print(f"  ⚠️ Error during pre-init RoPE: {e}")
        print("  🔄 Model will work but could have an OOM at first launch")
        
    #finally:
        # IMPORTANT: Remettre les modèles sur leurs devices originaux
        #print("  🔄 Restauration des modèles sur GPU...")
        # runner.dit.to(dit_device)
        #runner.vae.to(vae_device)
        #clear_vram_cache()
        
    #print("🎯 Pre-init RoPE done!")


def generation_step(runner, text_embeds_dict, preserve_vram, cond_latents, temporal_overlap):
    """
    model_name = "3B" if "3b" in model.lower() else "7B"
    print(f"\n🔍 {model_name} - Input Check:")
    print(f"  Cond latents: {cond_latents[0].shape} {cond_latents[0].dtype}")
    print(f"  Text pos: {text_embeds_dict['texts_pos'][0].shape}")
    print(f"  Text neg: {text_embeds_dict['texts_neg'][0].shape}")
    print(f"  Device: {cond_latents[0].device}")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🚀 OPTIMISATION: Détecter le dtype du modèle pour performance maximale
    model_dtype = next(runner.dit.parameters()).dtype
    
    # Adapter les dtypes selon le modèle (FP8 natif supporté)
    if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 natif: utiliser BFloat16 pour les calculs intermédiaires (compatible)
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        #print(f"🚀 Using FP8 pipeline with BFloat16 intermediates")
    elif model_dtype == torch.float16:
        dtype = torch.float16
        autocast_dtype = torch.float16
        #print(f"🎯 Using FP16 pipeline")
    else:
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        #print(f"🎯 Using BFloat16 pipeline")

    def _move_to_cuda(x):
        """Déplacer vers CUDA avec le dtype adaptatif optimal"""
        return [i.to(device, dtype=dtype) for i in x]

    # OPTIMISATION: Générer le bruit une seule fois et le réutiliser pour économiser VRAM
    with torch.cuda.device(device):
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    
    # print(f"Generating with noise shape: {noises[0].size()}, dtype: {noises[0].dtype}")
    
    # Déplacer avec le dtype adaptatif (optimisé pour FP8/FP16/BFloat16)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)
    
    # Nettoyer après déplacement
    #clear_vram_cache()
    
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # Utiliser le dtype adaptatif optimal
        t = (
            torch.tensor([1000.0], device=device, dtype=dtype)
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=device)[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    # Générer conditions avec nettoyage mémoire
    condition = runner.get_condition(
        noises[0],
        task="sr",
        latent_blur=_add_noise(cond_latents[0], aug_noises[0]),
    )
    conditions = [condition]
    t = time.time()
    # Utiliser l'autocast adaptatif pour performance optimale
    with torch.no_grad():
        with torch.autocast("cuda", autocast_dtype, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=preserve_vram,  # Offload important
                temporal_overlap=temporal_overlap,
                **text_embeds_dict,
                
            )
    
    print(f"🔄 INFERENCE time: {time.time() - t} seconds")
    # Traitement des échantillons avec OPTIMISATION 🚀
    t = time.time()
    samples = optimized_video_rearrange(video_tensors)
    last_latents = samples[-temporal_overlap:]
    print(f"🔄 sample size: {len(samples)}")
    print(f"🔄 Samples shape: {samples[0].shape}")
    print(f"🚀 OPTIMIZED REARRANGE time: {time.time() - t} seconds")
    
    # Nettoyage agressif des tenseurs intermédiaires
    #del video_tensors, noises, aug_noises, cond_latents, conditions
    #clear_vram_cache()
    
    return samples, last_latents

def auto_adjust_batch_size(initial_batch_size, available_vram_gb):
    """Ajuster automatiquement la taille de batch selon la VRAM et contrainte 4n+1"""
    # Ajustements selon le mode VRAM - OPTIMISÉ pour contrainte frames % 4 == 1
    
    if available_vram_gb < 24:
        candidates = [1]
    else:
        candidates = [i for i in range(1, 500) if i % 4 == 1]

    # Choisir la plus grande valeur 4n+1 qui ne dépasse pas initial_batch_size
    optimal_batch = 1
    for candidate in candidates:
        if candidate <= initial_batch_size:
            optimal_batch = candidate
        else:
            break
    
    #print(f"🎯 Optimal batch size (4n+1): {optimal_batch} (avoids padding)")
    return optimal_batch

def temporal_latent_blending(latents1, latents2, blend_frames):
    """
    🎨 Fondu temporel dans l'espace latent pour éviter les discontinuités
    
    Args:
        latents1: Latents du batch précédent (fins)
        latents2: Latents du batch actuel (début) 
        blend_frames: Nombre de frames à fondre
    
    Returns:
        Latents fondus pour transition douce
    """
    if latents1.shape[0] != latents2.shape[0]:
        # Ajuster les dimensions si nécessaire
        min_frames = min(latents1.shape[0], latents2.shape[0])
        latents1 = latents1[:min_frames]
        latents2 = latents2[:min_frames]
    
    # Créer des poids de fondu linéaire
    # Frame 0: 100% latents1, 0% latents2
    # Frame n: 0% latents1, 100% latents2
    weights1 = torch.linspace(1.0, 0.0, blend_frames).view(-1, 1, 1, 1).to(latents1.device)
    weights2 = torch.linspace(0.0, 1.0, blend_frames).view(-1, 1, 1, 1).to(latents2.device)
    
    # Appliquer le fondu
    blended_latents = weights1 * latents1 + weights2 * latents2
    
    return blended_latents


def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, preserve_vram="auto", temporal_overlap=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🚀 OPTIMISATION: Détecter le dtype réel du modèle pour performance maximale
    model_dtype = None
    try:
        # Obtenir le dtype réel du modèle DiT chargé
        model_dtype = next(runner.dit.parameters()).dtype
        #print(f"🎯 Model dtype detected: {model_dtype}")
        
        # Adapter les dtypes selon le modèle
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            #print("🚀 Using native FP8 pipeline for maximum performance!")
            # Pour FP8, utiliser BFloat16 pour les calculs intermédiaires (compatible)
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16  # VAE reste en BFloat16 pour compatibilité
        elif model_dtype == torch.float16:
            #print("🎯 Using FP16 pipeline")
            compute_dtype = torch.float16
            autocast_dtype = torch.float16
            vae_dtype = torch.float16
        else:  # BFloat16 ou autres
            #print("🎯 Using BFloat16 pipeline")
            compute_dtype = torch.bfloat16
            autocast_dtype = torch.bfloat16
            vae_dtype = torch.bfloat16
            
    except Exception as e:
        print(f"⚠️ Could not detect model dtype: {e}, falling back to BFloat16")
        model_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
        vae_dtype = torch.bfloat16

    # Obtenir VRAM disponible et ajuster batch_size
    if torch.cuda.is_available():
        """
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_alloc, vram_reserved, vram_max = get_vram_usage()
        
        available_vram = total_vram - vram_alloc
        
        # Ajuster automatiquement le batch_size avec bonus FP8
        #original_batch_size = batch_size
        
        # Bonus VRAM pour les modèles FP8 (50% moins de mémoire)
        effective_vram = available_vram
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            effective_vram = available_vram * 1.5  # 50% de bonus VRAM virtuel
            #print(f"🚀 FP8 model bonus: {available_vram:.1f}GB → {effective_vram:.1f}GB effective VRAM")
        
        batch_size = auto_adjust_batch_size(batch_size, effective_vram, vram_mode)
        
        if batch_size != original_batch_size:
            fp8_note = " (with FP8 bonus)" if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else ""
            #print(f"⚠️ Batch size adjusted: {original_batch_size} → {batch_size} (Mode: {vram_mode}{fp8_note})")
        
        # TABLE DE RÉFÉRENCE CORRECTE: frames % 4 == 1
        print("\n📋 Model constraint: frames % 4 == 1 (format 4n+1)")
        print("   • 1 frame → 1 frame ✅ (no padding)")
        print("   • 2-4 frames → 5 frames (padding +3,+1)")  
        print("   • 5 frames → 5 frames ✅ (no padding)")
        print("   • 6-8 frames → 9 frames (padding +3,+1)")
        print("   • 9 frames → 9 frames ✅ (no padding)")
        print("   • 10-12 frames → 13 frames (padding +3,+1)")
        print("   • 13 frames → 13 frames ✅ (no padding)")
        print("   • 14-16 frames → 17 frames (padding +3,+1)")
        print("   • 17 frames → 17 frames ✅ (no padding)")
        """
        # CONSEILS D'OPTIMISATION
        total_frames = len(images)
        optimal_batches = [x for x in [i for i in range(1, 200) if i % 4 == 1] if x <= total_frames]
        if optimal_batches:
            best_batch = max(optimal_batches)
            if best_batch != batch_size:
                print(f"\n💡 TIP: For {total_frames} frames, use batch_size={best_batch} to avoid padding")
                if batch_size not in optimal_batches:
                    padding_waste = sum(((i // 4) + 1) * 4 + 1 - i for i in range(batch_size, total_frames, batch_size))
                    print(f"   Currently: ~{padding_waste} wasted padding frames")

    def cut_videos(videos):
        """Version CORRECTE qui respecte la contrainte: frames % 4 == 1"""
        t = videos.size(1)
        
        if t % 4 == 1:
            return videos
        
        # Calculer le prochain nombre valide (4n + 1)
        padding_needed = (4 - (t % 4)) % 4 + 1
        
        # Appliquer le padding pour atteindre la forme 4n+1
        last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
        result = torch.cat([videos, last_frame], dim=1)
        
        return result

    # classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # set random seed
    set_seed(seed)

    # 🚀 OPTIMISATION: Remplacer Rearrange par PyTorch natif pour gain marginal
    video_transform = Compose(
        [
            NaResize(
                resolution=(res_w),
                mode="side",
                # Upsample image, model only trained for high res.
                downsample_only=False,
            ),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),  # t c h w -> c t h w (plus rapide que Rearrange)
        ]
    )

    # generation loop
    batch_samples = []
    final_tensor = None
    
    # Load text embeddings avec dtype adaptatif
    #print("🔄 Loading text embeddings...")
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device, dtype=compute_dtype)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device, dtype=compute_dtype)
    text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
    # Nettoyer après chargement
    #clear_vram_cache()
    reset_vram_peak()
    #runner.dit.to(device)
    #runner.vae.to(device)
    step = batch_size - temporal_overlap
    if step <= 0:
        step = batch_size
        temporal_overlap = 0
    t = time.time()
    images = images.to("cpu")
    print(f"🔄 Images to CPU time: {time.time() - t} seconds")
    #print(f"🎬 Processing {len(images)} images with step={step}, overlap={temporal_overlap}")
    try:
        for batch_idx in range(0, len(images), step):
            # 📊 Calcul des indices avec chevauchement
            if batch_idx == 0:
                # Premier batch: pas de chevauchement
                start_idx = 0
                end_idx = min(batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = True
            else:
                # Batches suivants: chevauchement temporal_overlap frames
                start_idx = batch_idx 
                end_idx = min(start_idx + batch_size, len(images))
                effective_batch_size = end_idx - start_idx
                is_first_batch = False
                if effective_batch_size <= temporal_overlap:
                    break  # Pas assez de nouvelles frames, arrêter

            tps_loop = time.time()
            t = time.time()
            batch_number = (batch_idx // step + 1) if step > 0 else 1
            print(f"\n🎬 Batch {batch_number}: frames {start_idx}-{end_idx-1}")
            #print(f"   {'Standard diffusion' if is_first_batch else f'Context-aware: {temporal_overlap} overlap + {effective_batch_size-temporal_overlap} new'}")
        
           
            # Reset pic VRAM pour ce batch
            
            #print(f"🔄 Reset VRAM time: {time.time() - t} seconds")
            #t = time.time()
            video = images[start_idx:end_idx]
            #print(f"Video size: {video.size()}")
            
            # Utiliser le dtype de calcul adaptatif 
            video = video.permute(0, 3, 1, 2).to(device, dtype=compute_dtype)
            # print(f"Read video size: {video.size()}, dtype: {video.dtype}")
            #print(f"🔄 Permute video time: {time.time() - t} seconds")
            #t = time.time()
            # OPTIMISATION: Transformations vidéo avec gestion mémoire améliorée
            transformed_video = video_transform(video)
            ori_lengths = [transformed_video.size(1)]
            #print(f"🔄 Transform video time: {time.time() - t} seconds")
            #t = time.time()
            # GESTION CORRECTE: Respecter la contrainte frames % 4 == 1
            t = transformed_video.size(1)
            print(f"📹 Sequence of {t} frames")
            #t = time.time()
            # Vérifier si déjà au format correct (4n + 1)
            #t = transformed_video.size(1)
            print(f"🔄 Transformed video shape before cut: {transformed_video.shape}")
            if len(images)>=5 and t % 4 != 1:
                transformed_video = cut_videos(transformed_video)
                print(f"🔄 Transformed video shape: {transformed_video.shape}")
            # 🎯 STRATÉGIE TEMPORELLE CONTEXT-AWARE
            if is_first_batch or temporal_overlap == 0:
                # 🆕 PREMIER BATCH: Diffusion standard complète
                tps_vae = time.time()
                with torch.autocast("cuda", autocast_dtype, enabled=True):
                    cond_latents = runner.vae_encode([transformed_video])
                print(f"🔄 Cond latents shape: {cond_latents[0].shape}, time: {time.time() - tps_vae} seconds")
                #text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
                
                # Génération normale
                samples, previous_latents = generation_step(runner, text_embeds, preserve_vram, cond_latents=cond_latents, temporal_overlap=temporal_overlap)
                    
            else:
                # 🔄 BATCHES SUIVANTS: Context-aware avec chevauchement
                print(f"   🎯 Using context-aware inference...")
                
                # Construire la séquence avec chevauchement
                # 1. Frames de chevauchement (du batch précédent)
                #overlap_frames = transformed_video[:temporal_overlap]
                
                # 2. Nouvelles frames
                #new_frames = transformed_video[temporal_overlap:]
                
                # 3. Créer les latents avec continuité
                overlap_latents = previous_latents[0] # Dernières frames du batch précédent
                print(f"🔄 Overlap latents shape: {overlap_latents.shape}")
                
                # Encoder seulement les nouvelles frames
                with torch.autocast("cuda", autocast_dtype, enabled=True):
                    new_latents = runner.vae_encode([transformed_video])
                print(f"🔄 New frames shape: {new_latents[0].shape}")
                # 4. Combiner les latents avec fondu dans l'espace latent
                combined_latents = temporal_latent_blending(
                    overlap_latents, 
                    new_latents[0][:temporal_overlap] if new_latents[0].shape[0] >= temporal_overlap else new_latents[0],
                    blend_frames=temporal_overlap
                )
                print(f"🔄 Combined latents shape: {combined_latents.shape}")
                # 5. Latents finaux: chevauchement fondu + nouvelles frames
                if new_latents[0].shape[0] > temporal_overlap:
                    final_latents = torch.cat([combined_latents, new_latents[0][temporal_overlap:]], dim=0)
                else:
                    final_latents = combined_latents
                print(f"🔄 Final latents shape: {final_latents.shape}")
                cond_latents = [final_latents]
                #text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
                
                # Génération context-aware
                samples, previous_latents = generation_step(runner, text_embeds, preserve_vram, cond_latents=cond_latents, temporal_overlap=temporal_overlap)
                
                # Mettre à jour les latents pour le prochain batch
                previous_latents = [final_latents]
            
            #sample = samples[0].to("cpu")
            sample = samples[0]
            if ori_lengths[0] < sample.shape[0]:
                sample = sample[:ori_lengths[0]]
            if temporal_overlap>0 and not is_first_batch and sample.shape[0] > effective_batch_size - temporal_overlap:
                sample = sample[temporal_overlap:]  # Supprimer les frames de chevauchement en sortie
            # 🚀 OPTIMISATION: Utiliser PyTorch natif au lieu de rearrange (2-5x plus rapide)
            input_video = [optimized_single_video_rearrange(transformed_video)]
            #print(f"🔄 Optimized single video rearrange time: {time.time() - t} seconds")
            t = time.time()
            if use_colorfix:
                sample = wavelet_reconstruction(sample, input_video[0][: sample.size(0)])
            print(f"🔄 Wavelet reconstruction time: {time.time() - t} seconds")
            t = time.time()
            # 🚀 OPTIMISATION: Remplacer rearrange par fonction optimisée
            sample = optimized_sample_to_image_format(sample)
            #print(f"🔄 Optimized sample format time: {time.time() - t} seconds")
            #t = time.time()     
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
            sample = sample.to("cpu")
            batch_samples.append(sample)
            print(f"🔄 Batch samples time: {time.time() - t} seconds")
            #t = time.time()
            # Nettoyage ultra-agressif après chaque batch
            print(f"🔄 Time batch: {time.time() - tps_loop} seconds")
            #input_video = input_video[0].to("cpu")
            video = video.to("cpu")
            del samples, sample, input_video, video, transformed_video

            
    finally:
        # Cleanup final des embeddings
        text_pos_embeds = text_pos_embeds.to("cpu")
        text_neg_embeds = text_neg_embeds.to("cpu")
        
        del text_pos_embeds, text_neg_embeds
        clear_vram_cache()
    
    final_video_images = torch.cat(batch_samples, dim=0)
    final_video_images = final_video_images.to("cpu")
    
    # 🚀 CORRECTION CRITIQUE: Convertir en Float16 pour ComfyUI
    if final_video_images.dtype != torch.float16:
        #print(f"🔧 Converting final video from {final_video_images.dtype} to Float16 for ComfyUI compatibility")
        final_video_images = final_video_images.to(torch.float16)
    
    # Cleanup batch_samples
    del batch_samples
    return final_video_images

def download_weight(model):
    model_path = os.path.join(base_cache_dir, model)
    vae_fp16_path = os.path.join(base_cache_dir, "ema_vae_fp16.safetensors")
    vae_fp8_path = os.path.join(base_cache_dir, "ema_vae_fp8_e4m3fn.safetensors")
    vae_legacy_path = os.path.join(base_cache_dir, "ema_vae.pth")
    
    # Download based on model type
    repo_id = "numz/SeedVR2_comfyUI"
    
    if not os.path.exists(model_path):
        print(f"📥 Downloading model: {model}")
        """
        if 'fp8_e4m3fn' in model:
            print("   📊 FP8 E4M3FN model - 50% smaller than FP16, optimized precision")
        elif 'fp8_e5m2' in model:
            print("   📊 FP8 E5M2 model - 50% smaller than FP16, extended range")
        elif 'fp16' in model:
            print("   📊 FP16 model - Standard precision")
        """
        hf_hub_download(repo_id=repo_id, filename=model, local_dir=base_cache_dir)
        print(f"✅ Downloaded: {model}")
    
    # 🚀 Priorité au VAE FP16 SafeTensors (format optimisé)
    if not os.path.exists(vae_fp16_path):
        print("📥 Downloading FP16 VAE SafeTensors...")
        try:
            hf_hub_download(repo_id=repo_id, filename="ema_vae_fp16.safetensors", local_dir=base_cache_dir)
            print("✅ Downloaded: ema_vae_fp16.safetensors (FP16 SafeTensors)")
        except Exception as e:
            print(f"⚠️ FP16 SafeTensors VAE not available: {e}")
            
            # Essayer le VAE FP8 en fallback
            if not os.path.exists(vae_fp8_path):
                print("📥 Trying FP8 VAE as fallback...")
                try:
                    hf_hub_download(repo_id=repo_id, filename="ema_vae_fp8_e4m3fn.safetensors", local_dir=base_cache_dir)
                    print("✅ Downloaded: ema_vae_fp8_e4m3fn.safetensors (FP8 fallback)")
                except Exception as e2:
                    print(f"⚠️ FP8 VAE also not available: {e2}")
                    
                    # Dernier fallback vers le VAE legacy
                    if not os.path.exists(vae_legacy_path):
                        print("📥 Downloading legacy VAE as final fallback...")
                        hf_hub_download(repo_id=repo_id, filename="ema_vae.pth", local_dir=base_cache_dir)
                        print("✅ Downloaded: ema_vae.pth (legacy fallback)")
        
    return

def clear_rope_cache(runner):
    """🧹 Nettoyer le cache RoPE pour libérer la VRAM"""
    print("🧹 Nettoyage du cache RoPE...")
    
    if hasattr(runner, 'cache') and hasattr(runner.cache, 'cache'):
        # Compter les entrées avant nettoyage
        cache_size = len(runner.cache.cache)
        
        # Libérer tous les tenseurs du cache
        for key, value in runner.cache.cache.items():
            if isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, 'cpu'):
                        item.cpu()
                        del item
            elif hasattr(value, 'cpu'):
                value.cpu()
                del value
        
        # Vider le cache
        runner.cache.cache.clear()
        print(f"  ✅ Cache RoPE vidé ({cache_size} entrées supprimées)")
    
    # Nettoyage VRAM agressif
    clear_vram_cache()
    torch.cuda.empty_cache()
    
    print("🎯 Nettoyage cache RoPE terminé!")

class FP8CompatibleDiT(torch.nn.Module):
    """
    Wrapper pour modèles DiT avec gestion automatique de compatibilité + optimisations avancées
    - FP8: Garde les paramètres FP8 natifs, convertit inputs/outputs
    - FP16: Utilise FP16 natif
    - RoPE: TOUJOURS forcé en BFloat16 pour compatibilité maximale
    - Flash Attention: Optimisation automatique des couches d'attention
    """
    def __init__(self, dit_model):
        super().__init__()
        self.dit_model = dit_model
        self.model_dtype = self._detect_model_dtype()
        self.is_fp8_model = self.model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        self.is_fp16_model = self.model_dtype == torch.float16
        
        # Détecter le type de modèle
        is_nadit_7b = self._is_nadit_model()      # NaDiT 7B (dit/nadit)
        is_nadit_v2_3b = self._is_nadit_v2_model()  # NaDiT v2 3B (dit_v2/nadit)
        """
        # 🔍 DEBUG: Afficher les informations de détection
        print(f"🔍 Model detection debug:")
        print(f"   Model type: {type(self.dit_model)}")
        print(f"   Model module: {self.dit_model.__class__.__module__}")
        print(f"   Model dtype: {self.model_dtype}")
        print(f"   Is FP8 model: {self.is_fp8_model}")
        print(f"   Is NaDiT 7B: {is_nadit_7b}")
        print(f"   Is NaDiT v2 3B: {is_nadit_v2_3b}")
        print(f"   Has emb_scale: {hasattr(self.dit_model, 'emb_scale')}")
        print(f"   Has vid_in: {hasattr(self.dit_model, 'vid_in')}")
        print(f"   Has txt_in: {hasattr(self.dit_model, 'txt_in')}")
        """
        
        if is_nadit_7b:
            # 🎯 CORRECTION CRITIQUE: TOUS les modèles NaDiT 7B (FP8 ET FP16) nécessitent une conversion BFloat16
            # L'architecture 7B a des problèmes de compatibilité dtype indépendamment du format de stockage
            if self.is_fp8_model:
                print("🎯 Detected NaDiT 7B FP8 - Converting all parameters to BFloat16")
                #self._force_nadit_bfloat16()
            else:
                print("🎯 Detected NaDiT 7B FP16")
            self._force_nadit_bfloat16()
            
        elif self.is_fp8_model and is_nadit_v2_3b:
            # Pour NaDiT v2 3B FP8: Convertir TOUT le modèle en BFloat16
            print("🎯 Detected NaDiT v2 3B FP8 - Converting all parameters to BFloat16")
            self._force_nadit_bfloat16()
        #else:
            # Pour les autres modèles (3B FP16, etc.): Forcer seulement RoPE en BFloat16
            #print("🎯 Standard model - Converting only RoPE to BFloat16")
            #self._force_rope_bfloat16()
        
        # 🚀 OPTIMISATION FLASH ATTENTION (Phase 2)
        self._apply_flash_attention_optimization()
    
    def _detect_model_dtype(self):
        """Détecter le dtype principal du modèle"""
        try:
            return next(self.dit_model.parameters()).dtype
        except:
            return torch.bfloat16
    
    def _is_nadit_model(self):
        """Détecter si c'est un modèle NaDiT (7B) avec une logique précise"""
        # 🎯 MÉTHODE PRINCIPALE: Vérifier l'attribut emb_scale (spécifique au 7B)
        # C'est le critère le plus fiable pour distinguer 7B vs 3B
        if hasattr(self.dit_model, 'emb_scale'):
            return True
        
        # 🎯 MÉTHODE SECONDAIRE: Vérifier le chemin du module pour NaDiT 7B (dit/nadit, pas dit_v2)
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit.nadit' in model_module and 'dit_v2' not in model_module:
            return True
        
        # 🚫 SUPPRIMÉ: Méthode par nom du type (trop générale, détecte aussi les 3B)
        # 🚫 SUPPRIMÉ: Méthode par structure sans emb_scale (trop générale)
        
        return False
    
    def _is_nadit_v2_model(self):
        """Détecter si c'est un modèle NaDiT v2 (3B) avec une logique précise"""
        # 🎯 MÉTHODE PRINCIPALE: Vérifier le chemin du module pour NaDiT v2 (dit_v2/nadit)
        model_module = str(self.dit_model.__class__.__module__).lower()
        if 'dit_v2' in model_module:
            return True
        
        # 🎯 MÉTHODE SECONDAIRE: Vérifier la structure spécifique au 3B
        # NaDiT v2 3B a vid_in, txt_in, emb_in mais PAS d'emb_scale
        if (hasattr(self.dit_model, 'vid_in') and 
            hasattr(self.dit_model, 'txt_in') and 
            hasattr(self.dit_model, 'emb_in') and
            not hasattr(self.dit_model, 'emb_scale')):  # Absence d'emb_scale = 3B
            return True
        
        return False
    
    def _force_rope_bfloat16(self):
        """🎯 Forcer TOUS les modules RoPE en BFloat16 pour compatibilité maximale"""
        #print("🔧 Forcing ALL RoPE modules to BFloat16 for maximum compatibility...")
        
        rope_count = 0
        for name, module in self.dit_model.named_modules():
            # Identifier modules RoPE par nom ou type
            if any(keyword in name.lower() for keyword in ['rope', 'rotary', 'embedding']):
                # Convertir tous les paramètres de ce module en BFloat16
                for param_name, param in module.named_parameters():
                    if param.dtype != torch.bfloat16:
                        param.data = param.data.to(torch.bfloat16)
                        rope_count += 1
                        
                # Convertir aussi les buffers (non-trainable parameters)
                for buffer_name, buffer in module.named_buffers():
                    if buffer.dtype != torch.bfloat16:
                        buffer.data = buffer.data.to(torch.bfloat16)
                        rope_count += 1
        
        #if rope_count > 0:
            #print(f"   ✅ Converted {rope_count} RoPE parameters/buffers to BFloat16")
        #else:
            #print(f"   ✅ RoPE modules already in BFloat16 or not found")
    
    def _force_nadit_bfloat16(self):
        """🎯 Forcer TOUS les paramètres NaDiT en BFloat16 pour éviter les erreurs de promotion"""
        print("🔧 Converting ALL NaDiT parameters to BFloat16 for type compatibility...")
        
        converted_count = 0
        original_dtype = None
        
        # Convertir TOUS les paramètres vers BFloat16 (FP8, FP16, etc.)
        for name, param in self.dit_model.named_parameters():
            if original_dtype is None:
                original_dtype = param.dtype
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
                converted_count += 1
        
        # Convertir aussi les buffers
        for name, buffer in self.dit_model.named_buffers():
            if buffer.dtype != torch.bfloat16:
                buffer.data = buffer.data.to(torch.bfloat16)
                converted_count += 1
        
        print(f"   ✅ Converted {converted_count} parameters/buffers from {original_dtype} to BFloat16")
        
        # Mettre à jour le dtype détecté
        self.model_dtype = torch.bfloat16
        self.is_fp8_model = False  # Le modèle n'est plus FP8 après conversion
    
    def _apply_flash_attention_optimization(self):
        """🚀 OPTIMISATION FLASH ATTENTION - Accélération 30-50% des couches d'attention"""
        #print("🚀 Applying Flash Attention optimization to all attention layers...")
        
        attention_layers_optimized = 0
        flash_attention_available = self._check_flash_attention_support()
        
        for name, module in self.dit_model.named_modules():
            # Identifier toutes les couches d'attention
            if self._is_attention_layer(name, module):
                # Appliquer l'optimisation selon la disponibilité
                if self._optimize_attention_layer(name, module, flash_attention_available):
                    attention_layers_optimized += 1
        
        #optimization_type = "Flash Attention" if flash_attention_available else "SDPA"
        #print(f"   ✅ Optimized {attention_layers_optimized} attention layers with {optimization_type}")
        
        if not flash_attention_available:
            print("   ℹ️ Flash Attention not available, using PyTorch SDPA as fallback")
        
    def _check_flash_attention_support(self):
        """Vérifier si Flash Attention est disponible"""
        try:
            # Vérifier PyTorch SDPA (inclut Flash Attention sur H100/A100)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
            
            # Vérifier flash-attn package
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _is_attention_layer(self, name, module):
        """Identifier si un module est une couche d'attention"""
        attention_keywords = [
            'attention', 'attn', 'self_attn', 'cross_attn', 'mhattn', 'multihead',
            'transformer_block', 'dit_block'
        ]
        
        # Vérifier par nom
        if any(keyword in name.lower() for keyword in attention_keywords):
            return True
        
        # Vérifier par type de module
        module_type = type(module).__name__.lower()
        if any(keyword in module_type for keyword in attention_keywords):
            return True
        
        # Vérifier par attributs (modules avec q, k, v projections)
        if hasattr(module, 'q_proj') or hasattr(module, 'qkv') or hasattr(module, 'to_q'):
            return True
        
        return False
    
    def _optimize_attention_layer(self, name, module, flash_attention_available):
        """Optimiser une couche d'attention spécifique"""
        try:
            # Sauvegarder la méthode forward originale
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            
            # Créer la nouvelle méthode forward optimisée
            if flash_attention_available:
                optimized_forward = self._create_flash_attention_forward(module, name)
            else:
                optimized_forward = self._create_sdpa_forward(module, name)
            
            # Remplacer la méthode forward
            module.forward = optimized_forward
            return True
            
        except Exception as e:
            print(f"   ⚠️ Failed to optimize attention layer '{name}': {e}")
            return False
    
    def _create_flash_attention_forward(self, module, layer_name):
        """Créer un forward optimisé avec Flash Attention"""
        original_forward = module._original_forward
        
        def flash_attention_forward(*args, **kwargs):
            try:
                # Essayer d'utiliser Flash Attention via SDPA
                return self._sdpa_attention_forward(original_forward, module, *args, **kwargs)
            except Exception as e:
                # Fallback vers l'implémentation originale
                print(f"   ⚠️ Flash Attention failed for {layer_name}, using original: {e}")
                return original_forward(*args, **kwargs)
        
        return flash_attention_forward
    
    def _create_sdpa_forward(self, module, layer_name):
        """Créer un forward optimisé avec SDPA PyTorch"""
        original_forward = module._original_forward
        
        def sdpa_forward(*args, **kwargs):
            try:
                return self._sdpa_attention_forward(original_forward, module, *args, **kwargs)
            except Exception as e:
                # Fallback vers l'implémentation originale
                return original_forward(*args, **kwargs)
        
        return sdpa_forward
    
    def _sdpa_attention_forward(self, original_forward, module, *args, **kwargs):
        """Forward pass optimisé utilisant SDPA (Scaled Dot Product Attention)"""
        # Détecter si on peut intercepter et optimiser cette couche
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            
            # Vérifier les dimensions pour s'assurer que c'est une attention standard
            if len(input_tensor.shape) >= 3:  # [batch, seq_len, hidden_dim] ou similaire
                try:
                    return self._optimized_attention_computation(module, input_tensor, *args[1:], **kwargs)
                except:
                    pass
        
        # Fallback vers l'implémentation originale
        return original_forward(*args, **kwargs)
    
    def _optimized_attention_computation(self, module, input_tensor, *args, **kwargs):
        """Calcul d'attention optimisé avec SDPA"""
        # Essayer de détecter le format d'attention standard
        batch_size, seq_len = input_tensor.shape[:2]
        
        # Vérifier si le module a des projections Q, K, V standard
        if hasattr(module, 'qkv') or (hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj')):
            return self._compute_sdpa_attention(module, input_tensor, *args, **kwargs)
        
        # Si pas de format standard détecté, utiliser l'original
        return module._original_forward(input_tensor, *args, **kwargs)
    
    def _compute_sdpa_attention(self, module, x, *args, **kwargs):
        """Calcul SDPA optimisé pour modules d'attention standard"""
        try:
            # Cas 1: Module avec projection QKV combinée
            if hasattr(module, 'qkv'):
                qkv = module.qkv(x)
                # Reshape pour séparer Q, K, V
                batch_size, seq_len, _ = qkv.shape
                qkv = qkv.reshape(batch_size, seq_len, 3, -1)
                q, k, v = qkv.unbind(dim=2)
                
            # Cas 2: Projections Q, K, V séparées
            elif hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                q = module.q_proj(x)
                k = module.k_proj(x)
                v = module.v_proj(x)
            else:
                # Format non supporté, utiliser l'original
                return module._original_forward(x, *args, **kwargs)
            
            # Détecter le nombre de têtes
            head_dim = getattr(module, 'head_dim', None)
            num_heads = getattr(module, 'num_heads', None)
            
            if head_dim is None or num_heads is None:
                # Essayer de deviner à partir des dimensions
                hidden_dim = q.shape[-1]
                if hasattr(module, 'num_heads'):
                    num_heads = module.num_heads
                    head_dim = hidden_dim // num_heads
                else:
                    # Valeurs par défaut raisonnables
                    head_dim = 64
                    num_heads = hidden_dim // head_dim
            
            # Reshape pour multi-head attention
            batch_size, seq_len = q.shape[:2]
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Utiliser SDPA optimisé
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            ):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False
                )
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, num_heads * head_dim
            )
            
            # Projection de sortie si elle existe
            if hasattr(module, 'out_proj') or hasattr(module, 'o_proj'):
                proj = getattr(module, 'out_proj', None) or getattr(module, 'o_proj', None)
                attn_output = proj(attn_output)
            
            return attn_output
            
        except Exception as e:
            # En cas d'erreur, utiliser l'implémentation originale
            return module._original_forward(x, *args, **kwargs)
    
    def _convert_inputs_for_compatibility(self, *args, **kwargs):
        """Convertir les inputs selon le type de modèle et l'architecture"""
        converted_args = []
        converted_kwargs = {}
        
        # Détecter si c'est un modèle nadit (7B) qui gère mieux les types mixtes
        is_nadit_model = self._is_nadit_model()
        
        # 🔍 DEBUG: Afficher les types d'inputs reçus
        if not hasattr(self, '_debug_shown'):
            self._debug_shown = True
            print(f"🔍 Input types debug:")
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    print(f"   arg[{i}]: {arg.dtype} shape={arg.shape}")
                else:
                    print(f"   arg[{i}]: {type(arg)}")
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.dtype} shape={value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
        
        # Pour modèles FP8: différencier selon l'architecture
        if self.is_fp8_model:
            if is_nadit_model:
                # NaDiT (7B): Le modèle a été converti en BFloat16, convertir aussi les inputs Float32
                print("🎯 NaDiT model: Converting Float32 inputs to BFloat16 for compatibility")
                
                conversions_made = 0
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if arg.dtype == torch.float32:
                            # Convertir Float32 → BFloat16 pour correspondre au modèle converti
                            converted_args.append(arg.to(torch.bfloat16))
                            conversions_made += 1
                        elif arg.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            # Convertir FP8 → BFloat16 aussi
                            converted_args.append(arg.to(torch.bfloat16))
                            conversions_made += 1
                        else:
                            # Garder les autres types (BFloat16, int, long, etc.)
                            converted_args.append(arg)
                    else:
                        converted_args.append(arg)
                
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        if value.dtype == torch.float32:
                            # Convertir Float32 → BFloat16 pour correspondre au modèle converti
                            converted_kwargs[key] = value.to(torch.bfloat16)
                        elif value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            # Convertir FP8 → BFloat16 aussi
                            converted_kwargs[key] = value.to(torch.bfloat16)
                        else:
                            # Garder les autres types
                            converted_kwargs[key] = value
                    else:
                        converted_kwargs[key] = value
                
                print(f"   ✅ Made {conversions_made} type conversions")
            else:
                # DiT standard (3B): Convertir FP8 → BFloat16 pour compatibilité
                print("🔄 Standard DiT FP8 model: Converting FP8 inputs to BFloat16")
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_args.append(arg.to(torch.bfloat16))
                    else:
                        converted_args.append(arg)
                
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_kwargs[key] = value.to(torch.bfloat16)
                    else:
                        converted_kwargs[key] = value
        
        # Pour modèles FP16: garder FP16 mais s'assurer que RoPE reste BFloat16
        else:
            converted_args = list(args)
            converted_kwargs = kwargs.copy()
                
        return tuple(converted_args), converted_kwargs
    
    def _convert_outputs_back(self, outputs):
        """Reconvertir les outputs selon le modèle original"""
        # Détecter si c'est un modèle nadit
        is_nadit_model = self._is_nadit_model()
        
        if not self.is_fp8_model:
            return outputs  # FP16 et autres: pas de conversion
            
        if is_nadit_model:
            # NaDiT: Le modèle a été converti en BFloat16, outputs déjà corrects
            return outputs
        
        # DiT standard: reconvertir outputs BFloat16 → FP8 natif
        target_dtype = self.model_dtype
        
        if hasattr(outputs, 'vid_sample'):
            # Format standard avec .vid_sample
            if outputs.vid_sample.dtype != target_dtype:
                outputs.vid_sample = outputs.vid_sample.to(target_dtype)
        elif isinstance(outputs, torch.Tensor):
            # Tensor simple
            if outputs.dtype != target_dtype:
                outputs = outputs.to(target_dtype)
        
        return outputs
    
    def forward(self, *args, **kwargs):
        """Forward pass avec gestion intelligente des types selon l'architecture"""
        is_nadit_7b = self._is_nadit_model()
        is_nadit_v2_3b = self._is_nadit_v2_model()
        
        # Conversion des inputs selon l'architecture
        if is_nadit_7b or is_nadit_v2_3b:
            # Pour les modèles NaDiT (7B et v2 3B): Tout en BFloat16
            converted_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    if arg.dtype in (torch.float32, torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_args.append(arg.to(torch.bfloat16))
                    else:
                        converted_args.append(arg)
                else:
                    converted_args.append(arg)
            
            converted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype in (torch.float32, torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_kwargs[key] = value.to(torch.bfloat16)
                    else:
                        converted_kwargs[key] = value
                else:
                    converted_kwargs[key] = value
            
            args = tuple(converted_args)
            kwargs = converted_kwargs
        else:
            # Pour les modèles standards: Conversion selon le dtype du modèle
            if self.is_fp8_model:
                # Convertir FP8 → BFloat16 pour calculs
                converted_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_args.append(arg.to(torch.bfloat16))
                    else:
                        converted_args.append(arg)
                
                converted_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_kwargs[key] = value.to(torch.bfloat16)
                    else:
                        converted_kwargs[key] = value
                
                args = tuple(converted_args)
                kwargs = converted_kwargs
            elif self.is_fp16_model:
                # Convertir Float32 → FP16 pour modèles FP16
                converted_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float32:
                        converted_args.append(arg.to(torch.float16))
                    else:
                        converted_args.append(arg)
                
                converted_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                        converted_kwargs[key] = value.to(torch.float16)
                    else:
                        converted_kwargs[key] = value
                
                args = tuple(converted_args)
                kwargs = converted_kwargs
        
        try:
            return self.dit_model(*args, **kwargs)
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            print(f"   Model type: NaDiT 7B={is_nadit_7b}, NaDiT v2 3B={is_nadit_v2_3b}")
            print(f"   Args dtypes: {[arg.dtype if isinstance(arg, torch.Tensor) else type(arg) for arg in args]}")
            print(f"   Kwargs dtypes: {[(k, v.dtype if isinstance(v, torch.Tensor) else type(v)) for k, v in kwargs.items()]}")
            raise
    
    def __getattr__(self, name):
        """Rediriger tous les autres attributs vers le modèle original"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model', '_forward_count']:
            return super().__getattr__(name)
        return getattr(self.dit_model, name)
    
    def __setattr__(self, name, value):
        """Rediriger les assignments vers le modèle original sauf pour nos attributs"""
        if name in ['dit_model', 'model_dtype', 'is_fp8_model', 'is_fp16_model', '_forward_count']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, 'dit_model'):
                setattr(self.dit_model, name, value)
            else:
                super().__setattr__(name, value)

def apply_fp8_compatibility_hooks(model):
    """
    Système de hooks pour intercepter les modules problématiques FP8
    Alternative si le wrapper ne suffit pas.
    """
    def create_fp8_safe_hook(original_dtype):
        def hook_fn(module, input, output):
            # Convertir output FP8 → BFloat16 si nécessaire pour compatibilité
            if isinstance(output, torch.Tensor) and output.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # Conserver temporairement en BFloat16 pour éviter les erreurs downstream
                return output.to(torch.bfloat16)
            elif isinstance(output, (tuple, list)):
                # Traiter les outputs multiples
                converted_output = []
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        converted_output.append(item.to(torch.bfloat16))
                    else:
                        converted_output.append(item)
                return type(output)(converted_output)
            return output
        return hook_fn
    
    # Appliquer hooks sur modules critiques
    problematic_modules = []
    for name, module in model.named_modules():
        # Identifier modules RoPE et attention qui causent des problèmes FP8
        if any(keyword in name.lower() for keyword in ['rope', 'rotary', 'attention', 'mmattn']):
            if hasattr(module, 'register_forward_hook'):
                hook = module.register_forward_hook(create_fp8_safe_hook(torch.float8_e4m3fn))
                problematic_modules.append((name, hook))
    
    print(f"🔧 Applied FP8 compatibility hooks to {len(problematic_modules)} modules")
    return problematic_modules



def optimized_video_rearrange(video_tensors):
    """
    🚀 VERSION OPTIMISÉE du réarrangement vidéo
    Remplace la boucle lente par des opérations vectorisées
    
    Transforme:
    - 3D: c h w -> t c h w (avec t=1)  
    - 4D: c t h w -> t c h w
    
    Gains attendus: 5-10x plus rapide
    """
    if not video_tensors:
        return []
    
    # 🔍 Analyser les dimensions pour optimiser le traitement
    videos_3d = []
    videos_4d = []
    indices_3d = []
    indices_4d = []
    
    for i, video in enumerate(video_tensors):
        if video.ndim == 3:
            videos_3d.append(video)
            indices_3d.append(i)
        else:  # ndim == 4
            videos_4d.append(video)
            indices_4d.append(i)
    
    # 🎯 Préparer le résultat final
    samples = [None] * len(video_tensors)
    
    # 🚀 TRAITEMENT BATCH pour vidéos 3D (c h w -> 1 c h w)
    if videos_3d:
        # Méthode 1: Stack + permute (plus rapide que rearrange)
        # c h w -> c 1 h w -> 1 c h w
        batch_3d = torch.stack([v.unsqueeze(1) for v in videos_3d])  # [batch, c, 1, h, w]
        batch_3d = batch_3d.permute(0, 2, 1, 3, 4)  # [batch, 1, c, h, w]
        
        for i, idx in enumerate(indices_3d):
            samples[idx] = batch_3d[i]  # [1, c, h, w]
    
    # 🚀 TRAITEMENT BATCH pour vidéos 4D (c t h w -> t c h w)  
    if videos_4d:
        # Vérifier si toutes les vidéos 4D ont la même forme pour optimisation maximale
        shapes = [v.shape for v in videos_4d]
        if len(set(shapes)) == 1:
            # 🎯 OPTIMISATION MAXIMALE: Toutes les formes identiques
            # Stack + permute en une seule opération
            batch_4d = torch.stack(videos_4d)  # [batch, c, t, h, w]
            batch_4d = batch_4d.permute(0, 2, 1, 3, 4)  # [batch, t, c, h, w]
            
            for i, idx in enumerate(indices_4d):
                samples[idx] = batch_4d[i]  # [t, c, h, w]
        else:
            # 🔄 FALLBACK: Formes différentes, traitement individuel optimisé
            for i, idx in enumerate(indices_4d):
                # Utiliser permute au lieu de rearrange (plus rapide)
                samples[idx] = videos_4d[i].permute(1, 0, 2, 3)  # c t h w -> t c h w
    
    return samples

def optimized_single_video_rearrange(video):
    """
    🚀 VERSION OPTIMISÉE pour un seul tensor vidéo
    Remplace rearrange() par des opérations PyTorch natives
    
    Transforme:
    - 3D: c h w -> 1 c h w (ajouter dimension temporelle)
    - 4D: c t h w -> t c h w (permuter dimensions)
    
    Gains attendus: 2-5x plus rapide que rearrange()
    """
    if video.ndim == 3:
        # c h w -> 1 c h w (ajouter dimension temporelle t=1)
        return video.unsqueeze(0)
    else:  # ndim == 4
        # c t h w -> t c h w (permuter channels et temporal)
        return video.permute(1, 0, 2, 3)
    
def optimized_sample_to_image_format(sample):
    """
    🚀 VERSION OPTIMISÉE pour convertir sample vers format image
    Remplace rearrange() par des opérations PyTorch natives
    
    Transforme:
    - 3D: c h w -> 1 h w c (ajouter dimension temporelle + permuter vers format image)
    - 4D: t c h w -> t h w c (permuter vers format image)
    
    Gains attendus: 2-5x plus rapide que rearrange()
    """
    if sample.ndim == 3:
        # c h w -> 1 h w c (ajouter dimension temporelle puis permuter)
        return sample.unsqueeze(0).permute(0, 2, 3, 1)
    else:  # ndim == 4
        # t c h w -> t h w c (permuter channels vers la fin)
        return sample.permute(0, 2, 3, 1)


class SeedVR2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": ([
                    "seedvr2_ema_3b_fp16.safetensors", 
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_fp16.safetensors",
                    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                ], "seedvr2_ema_3b_fp16.safetensors"),
                "seed": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 1}),
                "new_width": ("INT", {"default": 1280, "min": 1, "max": 4320, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 2.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 2048, "step": 4}),
                "preserve_vram": ("BOOLEAN", {"default": True})
                
            },
        }
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images, model, seed, new_width, cfg_scale, batch_size, preserve_vram):
        temporal_overlap = 0
        t_tot = time.time()
        download_weight(model)
        #print(f"🔄 Download weight time: {time.time() - t_tot} seconds")
        t = time.time()
        runner = configure_runner(model)
        print(f"🔄 Configure runner time: {time.time() - t} seconds")
        t = time.time()
        #vram_mode = preserve_vram
        
        try:
            sample = generation_loop(runner, images, cfg_scale, seed, new_width, batch_size, preserve_vram, temporal_overlap)
        finally:
            # Aggressive cleanup
            # Move models to CPU before deletion
            clear_rope_cache(runner)
            if hasattr(runner, 'dit') and runner.dit is not None:
                runner.dit.cpu()
                del runner.dit
            if hasattr(runner, 'vae') and runner.vae is not None:
                runner.vae.cpu()
                del runner.vae
            if hasattr(runner, 'schedule'):
                del runner.schedule
            if hasattr(runner, 'config'):
                del runner.config
                
            del runner
            _, _, peak = get_vram_usage()
            images.to("cpu")
            del images
            print(f"🔄 VRAM peak: {peak:.2f}GB peak")
            # Multiple cleanup passes
            gc.collect()
            torch.cuda.empty_cache()
            #if torch.cuda.is_available():
                #torch.cuda.synchronize()
        print(f"🔄 Execution time: {time.time() - t_tot} seconds")
        return (sample, )



NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
}


