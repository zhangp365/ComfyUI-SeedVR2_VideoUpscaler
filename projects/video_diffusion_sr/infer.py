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

from typing import List, Optional, Tuple, Union
import torch
import gc
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from ...common.config import create_object
from ...common.decorators import log_on_entry, log_runtime
from ...common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from ...common.distributed import (
    get_device,
    get_global_rank,
)

from ...common.distributed.meta_init_utils import (
    meta_non_persistent_buffer_init_fn,
)
# from common.fs import download

from ...models.dit_v2 import na

class VideoDiffusionInfer():
    def __init__(self, config: DictConfig):
        # print(config)
        self.config = config

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    @log_on_entry
    @log_runtime
    def configure_dit_model(self, device="cpu", checkpoint=None):
        # Load dit checkpoint.
        # For fast init & resume,
        #   when training from scratch, rank0 init DiT on cpu, then sync to other ranks with FSDP.
        #   otherwise, all ranks init DiT on meta device, then load_state_dict with assign=True.
        if self.config.dit.get("init_with_meta_device", False):
            init_device = "cpu" if get_global_rank() == 0 and checkpoint is None else "meta"
        else:
            init_device = "cpu"

        # Create dit model.
        with torch.device(init_device):
            self.dit = create_object(self.config.dit.model)
        self.dit.set_gradient_checkpointing(self.config.dit.gradient_checkpoint)

        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", mmap=True)
            loading_info = self.dit.load_state_dict(state, strict=True, assign=True)
            print(f"Loading pretrained ckpt from {checkpoint}")
            print(f"Loading info: {loading_info}")
            self.dit = meta_non_persistent_buffer_init_fn(self.dit)

        if device in [get_device(), "cuda"]:
            self.dit.to(get_device())

        # Print model size.
        num_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        print(f"DiT trainable parameters: {num_params:,}")

    @log_on_entry
    @log_runtime
    def configure_vae_model(self):
        # Create vae model.
        dtype = getattr(torch, self.config.vae.dtype)
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=get_device(), dtype=dtype)

        # Load vae checkpoint.
        state = torch.load(
            self.config.vae.checkpoint, map_location=get_device(), mmap=True
        )
        self.vae.load_state_dict(state)

        # Set causal slicing.
        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)

    # ------------------------------ Diffusion ------------------------------ #

    def configure_diffusion(self):
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=get_device(),
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=get_device(),
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        samples = []
        if len(latents) > 0:
            device = get_device()
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group latents of the same shape to batches if enabled.
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            # Vae process by each group avec gestion OOM.
            for latent in latents:
                latent = latent.to(device, dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)
                
                # Essayer décodage normal, puis chunked si OOM
                try:
                    with torch.autocast("cuda", torch.float16, enabled=True):
                        sample = self.vae.decode(latent).sample
                except torch.OutOfMemoryError:
                    print(f"⚠️ OOM pendant VAE decode, passage en mode chunk...")
                    sample = self.vae_decode_chunked(latent)
                
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                samples.append(sample)
                
                # Nettoyage après chaque décodage
                torch.cuda.empty_cache()

            # Ungroup back to individual sample with the original order.
            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples

    def vae_decode_chunked(self, latent):
        """Decode latent par chunks pour économiser la VRAM."""
        print(f"🔄 VAE decode chunked - shape: {latent.shape}")
        
        # Diviser en chunks plus petits selon la dimension batch
        batch_size = latent.shape[0]
        if batch_size <= 1:
            # Si déjà 1, essayer une approche différente selon les dimensions
            if len(latent.shape) == 5 and latent.shape[2] > 1:
                # Diviser sur la dimension temporelle/depth (dim 2)
                return self.vae_decode_temporal_chunks(latent)
            else:
                # Essayer division spatiale en dernier recours
                return self.vae_decode_spatial_chunks(latent)
        
        chunk_size = max(1, batch_size // 2)  # Diviser par 2
        
        results = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = latent[i:end_idx]
            
            print(f"🔄 Processing VAE chunk {i//chunk_size + 1}/{(batch_size + chunk_size - 1)//chunk_size}")
            
            # Nettoyer avant chaque chunk
            torch.cuda.empty_cache()
            gc.collect()
            
            # Decoder le chunk avec FP16
            with torch.autocast("cuda", torch.float16, enabled=True):
                try:
                    chunk_result = self.vae.decode(chunk).sample
                except torch.OutOfMemoryError:
                    # Si encore OOM, essayer division spatiale
                    chunk_result = self.vae_decode_spatial_chunks(chunk)
            
            # Déplacer immédiatement sur CPU pour libérer VRAM
            results.append(chunk_result.cpu())
            del chunk_result, chunk
            
            # Nettoyage après chaque chunk
            torch.cuda.empty_cache()
            
        # Recombiner les résultats sur GPU
        print(f"🔄 Recombining VAE chunks...")
        final_result = torch.cat([r.to(get_device()) for r in results], dim=0)
        
        # Nettoyer les résultats temporaires
        del results
        torch.cuda.empty_cache()
        
        return final_result

    def vae_decode_temporal_chunks(self, latent):
        """Decode par chunks temporels pour tenseurs 5D."""
        print(f"🔄 VAE decode temporal chunks - shape: {latent.shape}")
        
        # Pour tenseur 5D: [batch, channels, depth, height, width]
        batch_size, channels, depth, height, width = latent.shape
        
        # Diviser sur la dimension temporelle/depth
        chunk_size = max(1, depth // 2)
        results = []
        
        for i in range(0, depth, chunk_size):
            end_idx = min(i + chunk_size, depth)
            chunk = latent[:, :, i:end_idx, :, :]
            
            print(f"🔄 Processing temporal chunk {i//chunk_size + 1}/{(depth + chunk_size - 1)//chunk_size} - shape: {chunk.shape}")
            
            # Nettoyer avant chaque chunk
            torch.cuda.empty_cache()
            gc.collect()
            
            # Decoder le chunk avec FP16
            with torch.autocast("cuda", torch.float16, enabled=True):
                try:
                    chunk_result = self.vae.decode(chunk).sample
                except torch.OutOfMemoryError:
                    # Si encore OOM, essayer division spatiale sur ce chunk
                    print(f"⚠️ OOM sur chunk temporel, tentative spatiale...")
                    chunk_result = self.vae_decode_spatial_chunks(chunk)
            
            # Déplacer immédiatement sur CPU pour libérer VRAM
            results.append(chunk_result.cpu())
            del chunk_result, chunk
            
            # Nettoyage après chaque chunk
            torch.cuda.empty_cache()
            
        # Recombiner les résultats sur GPU
        print(f"🔄 Recombining temporal chunks...")
        final_result = torch.cat([r.to(get_device()) for r in results], dim=2)  # Concat sur dim temporelle
        
        # Nettoyer les résultats temporaires
        del results
        torch.cuda.empty_cache()
        
        return final_result

    def vae_decode_spatial_chunks(self, latent):
        """Decode par chunks spatiaux si la division par batch ne suffit pas."""
        print(f"🔄 VAE decode spatial chunks - shape: {latent.shape}")
        
        # Gérer différentes dimensions de tenseur
        if len(latent.shape) == 5:
            # Format: [batch, channels, depth, height, width]
            _, _, _, h, w = latent.shape
        elif len(latent.shape) == 4:
            # Format: [batch, channels, height, width]
            _, _, h, w = latent.shape
        else:
            raise ValueError(f"Format de tenseur non supporté: {latent.shape}")
        
        # Éviter de diviser si déjà trop petit
        if h <= 64 or w <= 64:
            print(f"⚠️ Tenseur trop petit pour division spatiale ({h}x{w}), tentative directe...")
            try:
                with torch.autocast("cuda", torch.float16, enabled=True):
                    return self.vae.decode(latent).sample
            except torch.OutOfMemoryError:
                print(f"❌ VAE decode failed - tensor too large for this GPU")
                raise
        
        h_chunk = max(32, h // 2)  # Minimum 32 pixels
        w_chunk = max(32, w // 2)  # Minimum 32 pixels
        
        results = []
        
        # Traiter chaque quadrant selon le format du tenseur
        for i in range(0, h, h_chunk):
            for j in range(0, w, w_chunk):
                h_end = min(i + h_chunk, h)
                w_end = min(j + w_chunk, w)
                
                # Découpage selon le nombre de dimensions
                if len(latent.shape) == 5:
                    chunk = latent[:, :, :, i:h_end, j:w_end]
                else:  # 4D
                    chunk = latent[:, :, i:h_end, j:w_end]
                
                print(f"🔄 Processing spatial chunk [{i}:{h_end}, {j}:{w_end}] - shape: {chunk.shape}")
                
                torch.cuda.empty_cache()
                
                with torch.autocast("cuda", torch.float16, enabled=True):
                    chunk_result = self.vae.decode(chunk).sample
                
                results.append((chunk_result.cpu(), i, h_end, j, w_end))
                del chunk_result, chunk
                torch.cuda.empty_cache()
        
        # Recombiner les chunks spatiaux
        print(f"🔄 Recombining spatial chunks...")
        # Créer un tensor de sortie vide basé sur le premier résultat
        first_result = results[0][0]
        output_shape = list(first_result.shape)
        
        # Ajuster les dimensions selon le format
        if len(first_result.shape) >= 4:
            # Dimensions spatiales sont les 2 dernières
            h_dim = -2
            w_dim = -1
            output_shape[h_dim] = h
            output_shape[w_dim] = w
        
        final_result = torch.zeros(output_shape, dtype=first_result.dtype, device=get_device())
        
        for chunk_result, i, h_end, j, w_end in results:
            chunk_gpu = chunk_result.to(get_device())
            
            # Assignation selon le format du tenseur
            if len(final_result.shape) == 5:
                final_result[:, :, :, i:h_end, j:w_end] = chunk_gpu
            else:  # 4D ou autre
                final_result[:, :, i:h_end, j:w_end] = chunk_gpu
            
            del chunk_gpu
        
        del results
        torch.cuda.empty_cache()
        
        return final_result

    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps

    def get_vram_usage(self):
        """Obtenir l'utilisation VRAM actuelle (allouée et réservée)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            return allocated, reserved, max_allocated
        return 0, 0, 0

    def get_vram_peak(self):
        """Obtenir le pic VRAM depuis le dernier reset"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0

    def reset_vram_peak(self):
        """Reset le compteur de pic VRAM"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def clear_vram_cache(self):
        """Nettoyer le cache VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
        dit_offload: bool = False,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Monitoring VRAM initial et reset des pics
        self.reset_vram_peak()
        
        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)

        # Forcer FP16 pour les embeddings texte
        if isinstance(text_pos_embeds, torch.Tensor):
            text_pos_embeds = text_pos_embeds.half()
        if isinstance(text_neg_embeds, torch.Tensor):
            text_neg_embeds = text_neg_embeds.half()

        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        # Forcer FP16 pour les latents
        latents = latents.half() if latents.dtype != torch.float16 else latents
        latents_cond = latents_cond.half() if latents_cond.dtype != torch.float16 else latents_cond

        # Enter eval mode.
        # was_training = self.dit.training
        self.dit.eval()

        # Fonction optimisée pour le DiT avec réduction VRAM robuste
        def optimized_dit_call(args):
            # Nettoyage préventif
            torch.cuda.empty_cache()
            gc.collect()
            
            # Préparation des tenseurs en FP16
            device = args.x_t.device
            x_t_half = args.x_t.half() if args.x_t.dtype != torch.float16 else args.x_t
            latents_cond_half = latents_cond.half() if latents_cond.dtype != torch.float16 else latents_cond
            timestep_half = args.t.repeat(batch_size).half()
            
            with torch.cuda.device(device):
                with torch.autocast("cuda", torch.float16, enabled=True):
                    # ÉTAPE 1: Calcul positif
                    vid_input_pos = torch.cat([x_t_half, latents_cond_half], dim=-1)
                    pos_result = self.dit(
                        vid=vid_input_pos,
                        txt=text_pos_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_pos_shapes,
                        timestep=timestep_half,
                    ).vid_sample
                    
                    # Nettoyer immédiatement
                    del vid_input_pos
                    torch.cuda.empty_cache()
                    
                    # ÉTAPE 2: Calcul négatif avec offloading si nécessaire
                    current_vram = self.get_vram_usage()[0]
                    if current_vram > 18.0:  # Offloader si VRAM élevée
                        pos_result_cpu = pos_result.cpu()
                        del pos_result
                        torch.cuda.empty_cache()
                        use_cpu_offload = True
                        #print(f"🔄 Offloading pos_result to CPU")
                    else:
                        use_cpu_offload = False
                    
                    vid_input_neg = torch.cat([x_t_half, latents_cond_half], dim=-1)
                    neg_result = self.dit(
                        vid=vid_input_neg,
                        txt=text_neg_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_neg_shapes,
                        timestep=timestep_half,
                    ).vid_sample
                    
                    # Nettoyer immédiatement
                    del vid_input_neg
                    torch.cuda.empty_cache()
                    
                    # ÉTAPE 3: CFG
                    cfg_scale_current = (
                        cfg_scale
                        if (args.i + 1) / len(self.sampler.timesteps)
                        <= self.config.diffusion.cfg.get("partial", 1)
                        else 1.0
                    )
                    
                    # Recharger pos_result si nécessaire
                    if use_cpu_offload:
                        pos_result = pos_result_cpu.to(device, dtype=torch.float16)
                        del pos_result_cpu
                        torch.cuda.empty_cache()
                    
                    # Calcul CFG optimisé
                    if cfg_scale_current != 1.0:
                        # Utiliser des opérations in-place pour économiser la mémoire
                        diff = pos_result.sub_(neg_result)  # diff = pos_result - neg_result
                        diff.mul_(cfg_scale_current)        # diff *= cfg_scale
                        result = neg_result.add_(diff)      # result = neg_result + diff
                        del diff
                    else:
                        result = neg_result
                    
                    # Nettoyer
                    if 'pos_result' in locals():
                        del pos_result
                    del neg_result
                    torch.cuda.empty_cache()
                    
                    # Appliquer rescale si nécessaire
                    rescale = self.config.diffusion.cfg.rescale
                    if rescale > 0 and rescale != 1.0:
                        result.mul_(rescale)
            
            # Nettoyage final
            torch.cuda.empty_cache()
            gc.collect()
            
            return result

        # Sampling avec optimisations VRAM
        print(f"🔄 Starting EulerSampler sampling...")
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        # Exit eval mode.
        #self.dit.train(was_training)

        # Unflatten.
        latents = na.unflatten(latents, latents_shapes)

        if dit_offload:
            self.dit.to(get_device())

        # VAE decode avec optimisations VRAM
        self.vae.to(get_device())
        
        # Nettoyer avant VAE decode
        torch.cuda.empty_cache()
        gc.collect()
        
        # VAE decode avec autocast FP16
        with torch.autocast("cuda", torch.float16, enabled=True):
            samples = self.vae_decode(latents)
        
        # Nettoyer immédiatement après VAE
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        if dit_offload:
            self.dit.to(get_device())
        
        return samples