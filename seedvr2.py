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
import mediapy
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
#print(os.getcwd())
import datetime
import itertools
import folder_paths
from tqdm import tqdm
#from models.dit import na
import gc
import time
from huggingface_hub import snapshot_download, hf_hub_download

# Import SafeTensors avec fallback
try:
    from safetensors.torch import save_file as save_safetensors_file
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
    print("✅ SafeTensors available")
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
            print(f"Using model path: {path}")
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
            print(f"Using VAE path: {path}")
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

def load_quantized_state_dict(checkpoint_path, device="cpu"):
    """Load state dict from SafeTensors or PyTorch"""
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors required to load this model. Install with: pip install safetensors")
        print(f"🔄 Loading SafeTensors: {checkpoint_path}")
        state = load_safetensors_file(checkpoint_path, device=device)
    elif checkpoint_path.endswith('.pth'):
        print(f"🔄 Loading PyTorch: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, mmap=True)
    else:
        raise ValueError(f"Unsupported format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
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
    print("Entering configure_dit_model (inference)")
    
    from .common.config import create_object
    
    # Create dit model on CPU first
    with torch.device("cpu"):
        runner.dit = create_object(config.dit.model)
    runner.dit.set_gradient_checkpointing(config.dit.gradient_checkpoint)

    if checkpoint:
        # Load model directly
        if checkpoint.endswith('.safetensors'):
            print("🔄 Loading SafeTensors model")
        elif checkpoint.endswith('.pth'):
            print("🔄 Loading PyTorch model")
        else:
            raise ValueError(f"Unsupported format. Expected .safetensors or .pth, got: {checkpoint}")
        
        # Support for quantized models
        state = load_quantized_state_dict(checkpoint, "cpu")
        loading_info = runner.dit.load_state_dict(state, strict=True, assign=True)
        #print(f"Loading pretrained ckpt from {checkpoint}")
        print(f"Loading info: {loading_info}")

    # Move to target device (already in FP16)
    runner.dit.to(device)


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
    print("🔄 Pré-initialisation du cache RoPE...")
    
    # Sauvegarder l'état actuel des modèles
    dit_device = next(runner.dit.parameters()).device
    vae_device = next(runner.vae.parameters()).device
    
    try:
        # Temporairement déplacer les modèles sur CPU pour libérer VRAM
        print("  📦 Déplacement temporaire des modèles sur CPU...")
        runner.dit.to("cpu")
        runner.vae.to("cpu")
        clear_vram_cache()
        
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
        print("  🧮 Calcul des fréquences RoPE communes...")
        
        # Accéder aux modules RoPE dans DiT (recherche récursive)
        def find_rope_modules(module):
            rope_modules = []
            for name, child in module.named_modules():
                if hasattr(child, 'get_freqs') and callable(getattr(child, 'get_freqs')):
                    rope_modules.append((name, child))
            return rope_modules
        
        rope_modules = find_rope_modules(runner.dit)
        print(f"  🎯 Trouvé {len(rope_modules)} modules RoPE")
        
        # Pré-calculer pour chaque module RoPE trouvé
        for name, rope_module in rope_modules:
            print(f"    ⚙️ Pré-calcul pour {name}...")
            
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
                                return rope_module.get_freqs(vid_shape.cpu(), txt_shape.cpu())
                        except Exception as e:
                            print(f"      ⚠️ Échec pour {cache_key}: {e}")
                            # Retourner des tenseurs vides comme fallback
                            return torch.zeros(1, 64), torch.zeros(1, 64)
                    
                    # Stocker dans le cache
                    temp_cache(cache_key, compute_freqs)
                    print(f"      ✅ Cached: {cache_key}")
                
            except Exception as e:
                print(f"    ❌ Erreur module {name}: {e}")
            finally:
                # Remettre sur le device original
                rope_module.to(original_device)
        
        # Copier le cache temporaire vers le cache du runner
        if hasattr(runner, 'cache'):
            runner.cache.cache.update(temp_cache.cache)
        else:
            runner.cache = temp_cache
        
        print("  ✅ Cache RoPE pré-initialisé avec succès!")
        
    except Exception as e:
        print(f"  ⚠️ Erreur lors de la pré-initialisation RoPE: {e}")
        print("  🔄 Le modèle fonctionnera mais pourrait avoir un OOM au premier lancement")
        
    finally:
        # IMPORTANT: Remettre les modèles sur leurs devices originaux
        print("  🔄 Restauration des modèles sur GPU...")
        runner.dit.to(dit_device)
        runner.vae.to(vae_device)
        clear_vram_cache()
        
    print("🎯 Pré-initialisation RoPE terminée!")


def generation_step(runner, text_embeds_dict, cond_latents, model="seedvr2_ema_3b_fp16.safetensors"):
    """
    model_name = "3B" if "3b" in model.lower() else "7B"
    print(f"\n🔍 {model_name} - Input Check:")
    print(f"  Cond latents: {cond_latents[0].shape} {cond_latents[0].dtype}")
    print(f"  Text pos: {text_embeds_dict['texts_pos'][0].shape}")
    print(f"  Text neg: {text_embeds_dict['texts_neg'][0].shape}")
    print(f"  Device: {cond_latents[0].device}")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # COMME L'ANCIEN CODE: bfloat16 pour TOUT (3B et 7B)
    dtype = torch.bfloat16
    autocast_dtype = torch.bfloat16

    def _move_to_cuda(x):
        """Version simplifiée comme l'ancien code - pas de conversion dtype forcée"""
        return [i.to(device) for i in x]

    # OPTIMISATION: Générer le bruit une seule fois et le réutiliser pour économiser VRAM
    with torch.cuda.device(device):
        base_noise = torch.randn_like(cond_latents[0], dtype=dtype)
        noises = [base_noise]
        aug_noises = [base_noise * 0.1 + torch.randn_like(base_noise) * 0.05]
    
    # print(f"Generating with noise shape: {noises[0].size()}, dtype: {noises[0].dtype}")
    
    # Déplacer sans forcer dtype (il sera automatiquement converti par autocast)
    noises, aug_noises, cond_latents = _move_to_cuda(noises), _move_to_cuda(aug_noises), _move_to_cuda(cond_latents)
    
    # Nettoyer après déplacement
    clear_vram_cache()
    
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        # COMME L'ANCIEN CODE: pas de dtype forcé
        t = (
            torch.tensor([1000.0], device=device)
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
    
    # COMME L'ANCIEN CODE: bfloat16 pour TOUT
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=True,  # Offload important
                **text_embeds_dict,
            )
    
    # Traitement des échantillons avec nettoyage
    samples = []
    for video in video_tensors:
        sample = (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        samples.append(sample)
        """
        print(f"🔍 SAMPLE OUTPUT:")
        print(f"  Sample shape: {sample.shape}, dtype: {sample.dtype}")
        print(f"  Sample min/max: {sample.min():.4f}/{sample.max():.4f}")
        print(f"  Sample has_nan: {torch.isnan(sample).any()}")
        """
    
    # Nettoyage agressif des tenseurs intermédiaires
    del video_tensors, noises, aug_noises, cond_latents, conditions
    clear_vram_cache()
    
    return samples

def auto_adjust_batch_size(initial_batch_size, available_vram_gb, vram_mode="auto"):
    """Ajuster automatiquement la taille de batch selon la VRAM et contrainte 4n+1"""
    # Ajustements selon le mode VRAM - OPTIMISÉ pour contrainte frames % 4 == 1
    if vram_mode == "extreme_economy":
        # Privilégier les valeurs 4n+1 : 1, 5, 9, 13
        candidates = [1, 5, 9, 13]
    elif vram_mode == "economy":
        # Privilégier les valeurs 4n+1 : 1, 5, 9, 13, 17
        candidates = [1, 5, 9, 13, 17]
    else:  # auto
        if available_vram_gb < 12:
            candidates = [1, 5, 9]
        elif available_vram_gb < 16:
            candidates = [1, 5, 9, 13]
        elif available_vram_gb < 20:
            candidates = [1, 5, 9, 13, 17]
        elif available_vram_gb < 24:
            candidates = [1, 5, 9, 13, 17, 21]
        else:
            candidates = [i for i in range(1, 200) if i % 4 == 1]
    
    # Choisir la plus grande valeur 4n+1 qui ne dépasse pas initial_batch_size
    optimal_batch = 1
    for candidate in candidates:
        if candidate <= initial_batch_size:
            optimal_batch = candidate
        else:
            break
    
    #print(f"🎯 Optimal batch size (4n+1): {optimal_batch} (avoids padding)")
    return optimal_batch

def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_w=720, batch_size=90, vram_mode="auto", model="seedvr2_ema_3b_fp16.safetensors"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # COMME L'ANCIEN CODE: bfloat16 pour TOUT
    model_dtype = torch.bfloat16
    vae_dtype = torch.bfloat16

    # Obtenir VRAM disponible et ajuster batch_size
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_alloc, vram_reserved, vram_max = get_vram_usage()
        available_vram = total_vram - vram_alloc
        
        # Ajuster automatiquement le batch_size
        original_batch_size = batch_size
        batch_size = auto_adjust_batch_size(batch_size, available_vram, vram_mode)
        
        if batch_size != original_batch_size:
            print(f"⚠️ Batch size adjusted: {original_batch_size} → {batch_size} (Mode: {vram_mode})")
        
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
    
    video = images

    def cut_videos(videos):
        """Version CORRECTE qui respecte la contrainte: frames % 4 == 1"""
        t = videos.size(1)
        
        print(f"🔍 cut_videos: {t} frames → ", end="")
        
        # CONTRAINTE CRITIQUE: Le modèle exige que le nombre de frames % 4 == 1
        # Donc les valeurs valides sont: 1, 5, 9, 13, 17, 21, etc.
        
        # Vérifier si déjà dans le bon format
        if t % 4 == 1:
            print(f"{t} frames ✅ (already 4n+1 format)")
            return videos
        
        # Calculer le prochain nombre valide (4n + 1)
        target_frames = ((t // 4) + 1) * 4 + 1
        padding_needed = target_frames - t
        
        print(f"{target_frames} frames (padding: +{padding_needed} for 4n+1 format)")
        
        # Appliquer le padding pour atteindre la forme 4n+1
        last_frame = videos[:, -1:].expand(-1, padding_needed, -1, -1).contiguous()
        result = torch.cat([videos, last_frame], dim=1)
        
        # Vérification de sécurité
        final_t = result.size(1)
        assert final_t % 4 == 1, f"ERREUR: {final_t} % 4 = {final_t % 4} ≠ 1"
        
        return result

    # classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # set random seed
    set_seed(seed)

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
            Rearrange("t c h w -> c t h w"),
        ]
    )

    # generation loop
    batch_samples = []
    final_tensor = None
    
    # Load text embeddings COMME L'ANCIEN CODE
    print("🔄 Loading text embeddings...")
    text_pos_embeds = torch.load(os.path.join(script_directory, 'pos_emb.pt')).to(device)
    text_neg_embeds = torch.load(os.path.join(script_directory, 'neg_emb.pt')).to(device)
    
    # Nettoyer après chargement
    clear_vram_cache()
    
    try:
        for batch_idx in range(0, len(images), batch_size):
            print(f"\n🔄 Processing batch {batch_idx//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
            # Reset pic VRAM pour ce batch
            reset_vram_peak()
            
            video = images[batch_idx:batch_idx+batch_size]
            #print(f"Video size: {video.size()}")
            
            # COMME L'ANCIEN CODE: pas de dtype forcé
            video = video.permute(0, 3, 1, 2).to(device)
            # print(f"Read video size: {video.size()}, dtype: {video.dtype}")
            
            # OPTIMISATION: Transformations vidéo avec gestion mémoire améliorée
            transformed_video = video_transform(video)
            ori_lengths = [transformed_video.size(1)]
            
            # GESTION CORRECTE: Respecter la contrainte frames % 4 == 1
            t = transformed_video.size(1)
            print(f"📹 Sequence of {t} frames")
            
            # Vérifier si déjà au format correct (4n + 1)
            if t % 4 == 1:
                #print("✅ Correct format (4n+1), skipping cut_videos")
                cond_latents = [transformed_video]
                input_video = [transformed_video.clone()]
            else:
                # Nécessite cut_videos pour respecter la contrainte
                #print(f"🔧 Padding needed for 4n+1 format")
                cond_latents = [cut_videos(transformed_video)]
                input_video = [transformed_video]
            
            # Nettoyer après transformation
            del video, transformed_video
            clear_vram_cache()
            
            # Encodage VAE avec optimisation mémoire
            # print(f"🔄 Encodage VAE: {list(map(lambda x: x.size(), cond_latents))}")
            
            # OPTIMISATION: Vérifier la VRAM avant l'encodage VAE
            # if not check_vram_safety("Encodage VAE", 3.0):
                #print("🔄 Additional cleanup before VAE encoding...")
            #    for _ in range(2):
            #        clear_vram_cache()
            #        time.sleep(0.1)
            
            runner.dit.to("cpu")  # Libérer VRAM pour VAE
            clear_vram_cache()
            
            runner.vae.to(device, dtype=torch.bfloat16)
            # COMME L'ANCIEN CODE: pas d'autocast spécifique pour VAE
            cond_latents = runner.vae_encode(cond_latents)
            
            
            runner.dit.to(device)  # Recharger DiT
            
            # Text embeddings (comme l'ancien code)
            text_embeds = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
            
            # OPTIMISATION: Vérification sécurité VRAM avant génération
            #if not check_vram_safety("Génération DiT", 4.0):
                #print("🔄 Emergency cleanup before generation...")
            #    runner.vae.to("cpu")
            #    for _ in range(3):
            #        clear_vram_cache()
            #        time.sleep(0.1)
            #    runner.dit.to(device)
            
            samples = generation_step(runner, text_embeds, cond_latents=cond_latents, model=model)
            
            # Nettoyer immédiatement après génération
            runner.dit.to("cpu")
            del cond_latents, text_embeds
            clear_vram_cache()
            
            # Post-traitement sur CPU
            sample = samples[0].to("cpu")
            if ori_lengths[0] < sample.shape[0]:
                sample = sample[:ori_lengths[0]]

            input_video[0] = (
                rearrange(input_video[0][:, None], "c t h w -> t c h w")
                if input_video[0].ndim == 3
                else rearrange(input_video[0], "c t h w -> t c h w")
            ).to("cpu")
            
            if use_colorfix:
                sample = wavelet_reconstruction(sample, input_video[0][: sample.size(0)])
            
            sample = (
                rearrange(sample[:, None], "t c h w -> t h w c")
                if sample.ndim == 3
                else rearrange(sample, "t c h w -> t h w c")
            )
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5)
            batch_samples.append(sample)
            
            # Nettoyage ultra-agressif après chaque batch
            del samples, sample, input_video
            
            # ÉTAPE 1: Forcer tous les modèles sur CPU
            #print(f"🔄 Nettoyage ultra-agressif batch {batch_idx//batch_size + 1}...")
            runner.dit.to("cpu")
            runner.vae.to("cpu")
            
            # ÉTAPE 2: Nettoyer toutes les variables locales possibles
            if 'cond_latents' in locals():
                del cond_latents
            if 'text_embeds' in locals():
                del text_embeds
            
            # ÉTAPE 3: Nettoyage mémoire agressif multiple
            for _ in range(3):  # Triple nettoyage
                clear_vram_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)  # Petit délai pour laisser le temps au nettoyage
            
            # ÉTAPE 4: Recharger seulement si nécessaire pour le prochain batch
            if batch_idx + batch_size < len(images):
                print(f"🔄 Preparing next batch...")
                # Attendre un peu avant de recharger
                time.sleep(0.5)
                runner.dit.to(device, dtype=torch.bfloat16)
                # Garder VAE sur CPU jusqu'au besoin
                clear_vram_cache()
            
    finally:
        # Cleanup final des embeddings
        del text_pos_embeds, text_neg_embeds
        clear_vram_cache()
    
    final_video_images = torch.cat(batch_samples, dim=0)
    final_video_images = final_video_images.to("cpu")
    
    # Cleanup batch_samples
    del batch_samples
    return final_video_images

def download_weight(model):
    model_path = os.path.join(base_cache_dir, model)
    vae_path = os.path.join(base_cache_dir, "ema_vae.pth")
    
    # Download based on model type
    # Download from your repo for SafeTensors
    if not os.path.exists(model_path):
        repo_id = "numz/SeedVR2_comfyUI"
        hf_hub_download(repo_id=repo_id, filename=model, local_dir=base_cache_dir)
    if not os.path.exists(vae_path):
        hf_hub_download(repo_id=repo_id, filename="ema_vae.pth", local_dir=base_cache_dir)
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

class SeedVR2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "model": (["seedvr2_ema_3b_fp16.safetensors", "seedvr2_ema_7b_fp16.safetensors"], "seedvr2_ema_3b_fp16.safetensors"),
                "seed": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 1}),
                "new_width": ("INT", {"default": 1280, "min": 1, "max": 2048, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 2.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 2048, "step": 1})
            },
        }
    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images, model, seed, new_width, cfg_scale, batch_size):
        download_weight(model)
        runner = configure_runner(model)
        vram_mode = "auto"
        
        try:
            sample = generation_loop(runner, images, cfg_scale, seed, new_width, batch_size, vram_mode, model)
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
            
            # Multiple cleanup passes
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return (sample, )



NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
}
