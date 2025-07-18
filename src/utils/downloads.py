"""
Downloads utility module for SeedVR2
Handles model and VAE downloads from HuggingFace Hub

Extracted from: seedvr2.py (line 968-1015)
"""

import os
import sys
from huggingface_hub import hf_hub_download
try:
    import folder_paths
    # Configuration des chemins
    base_cache_dir = os.path.join(folder_paths.models_dir, "SEEDVR2")

    # S'assurer que le dossier de cache existe
    folder_paths.add_model_folder_path("seedvr2", os.path.join(folder_paths.models_dir, "SEEDVR2"))
except:
    base_cache_dir = "./seedvr2_models"


class ProgressCapture:
    """Capture download progress from tqdm"""
    def __init__(self, desc):
        self.desc = desc
        self.shown_percentages = set()
        
    def write(self, text):
        if not text or text.isspace():
            return
        # Skip HF duplicate messages
        if any(skip in text for skip in ["Downloading '", "Download complete.", "to '/data/"]):
            return
        # Extract and show progress
        import re
        match = re.search(r'(\d+)%', text)
        if match:
            percent = int(match.group(1))
            if percent not in self.shown_percentages:
                self.shown_percentages.add(percent)
                speed_match = re.search(r'(\d+\.?\d*[KMGT]?B/s)', text)
                speed = f" @ {speed_match.group(1)}" if speed_match else ""
                sys.stdout.write(f"\r{self.desc}: {percent:3d}%{speed}")
                sys.stdout.flush()
                if percent == 100:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
    
    def flush(self):
        pass


def download_weight(model, model_dir=None):
    """
    Télécharge un modèle SeedVR2 et son VAE associé depuis HuggingFace Hub
    
    Args:
        model (str): Nom du fichier modèle à télécharger
                    (ex: "seedvr2_ema_3b_fp16.safetensors")
    
    Gère automatiquement:
    - Téléchargement du modèle principal
    - Téléchargement du VAE avec fallbacks:
      1. ema_vae_fp16.safetensors (priorité)
      2. ema_vae_fp8_e4m3fn.safetensors (fallback)  
      3. ema_vae.pth (legacy fallback)
    """
    if model_dir is None:
        model_path = os.path.join(base_cache_dir, model)
        vae_fp16_path = os.path.join(base_cache_dir, "ema_vae_fp16.safetensors")
    else:
        model_path = os.path.join(model_dir, model)
        vae_fp16_path = os.path.join(model_dir, "ema_vae_fp16.safetensors")
   
    # Configuration HuggingFace
    repo_id = "numz/SeedVR2_comfyUI"
    
    # 🚀 Téléchargement du modèle principal
    if not os.path.exists(model_path):
        print(f"📥 Downloading model: {model}")
        
        # Add progress capture
        old_stderr = sys.stderr
        sys.stderr = ProgressCapture(model)
        try:
            hf_hub_download(repo_id=repo_id, filename=model, local_dir=base_cache_dir if model_dir is None else model_dir)
        finally:
            sys.stderr = old_stderr
            
        print(f"✅ Downloaded: {model}")
    
    # 🚀 Téléchargement du VAE avec stratégie de fallback
    if not os.path.exists(vae_fp16_path):
        print("📥 Downloading FP16 VAE SafeTensors...")
        
        # Add progress capture
        old_stderr = sys.stderr
        sys.stderr = ProgressCapture("ema_vae_fp16.safetensors")
        try:
            hf_hub_download(
                repo_id=repo_id, 
                filename="ema_vae_fp16.safetensors", 
                local_dir=base_cache_dir if model_dir is None else model_dir
            )
            print("✅ Downloaded: ema_vae_fp16.safetensors (FP16 SafeTensors)")
        except Exception as e:
            print(f"⚠️ FP16 SafeTensors VAE not available: {e}")
        finally:
            sys.stderr = old_stderr
    
    return


def get_base_cache_dir():
    """
    Retourne le répertoire de cache base pour les modèles SeedVR2
    
    Returns:
        str: Chemin du répertoire de cache
    """
    return base_cache_dir