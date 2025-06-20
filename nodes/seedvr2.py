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
print(os.getcwd())
import datetime
from tqdm import tqdm
#from models.dit import na
import gc
from huggingface_hub import snapshot_download

from .data.image.transforms.divisible_crop import DivisibleCrop
from .data.image.transforms.na_resize import NaResize
from .data.video.transforms.rearrange import Rearrange
if os.path.exists("./projects/video_diffusion_sr/color_fix.py"):
    from projects.video_diffusion_sr.color_fix import wavelet_reconstruction
    use_colorfix=True
else:
    use_colorfix = False
    print('Note!!!!!! Color fix is not avaliable!')
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video


from common.distributed import (
    get_device,
    init_torch,
)

from common.distributed.advanced import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_sequence_parallel,
)

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.distributed.ops import sync_data
from common.seed import set_seed
from common.partition import partition_by_groups, partition_by_size
import argparse

def configure_sequence_parallel(sp_size):
    if sp_size > 1:
        init_sequence_parallel(sp_size)

def configure_runner(model, sp_size):
    if model == "seedvr2_ema_7b.pth":
        config_path = os.path.join('./configs_7b', 'main.yaml')
    else:
        config_path = os.path.join('./configs_3b', 'main.yaml')
    config = load_config(config_path)
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    configure_sequence_parallel(sp_size)
    if model == "seedvr2_ema_7b.pth":
        runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr2_ema_7b.pth')
    else:
        runner.configure_dit_model(device="cuda", checkpoint='./ckpts/seedvr2_ema_3b.pth')
    runner.configure_vae_model()
    # Set memory limit.
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner

def generation_step(runner, text_embeds_dict, cond_latents):
    def _move_to_cuda(x):
        return [i.to(get_device()) for i in x]

    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    print(f"Generating with noise shape: {noises[0].size()}.")
    noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
    noises, aug_noises, cond_latents = list(
        map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents))
    )
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        t = (
            torch.tensor([1000.0], device=get_device())
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        print(
            f"Timestep shifting from"
            f" {1000.0 * cond_noise_scale} to {t}."
        )
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors

    return samples

def generation_loop(runner, images, cfg_scale=1.0, seed=666, res_h=1280, res_w=720):
    video = images

    def cut_videos(videos, sp_size):
        t = videos.size(1)
        if t <= 4 * sp_size:
            print(f"Cut input video size: {videos.size()}")
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        else:
            padding = [videos[:, -1].unsqueeze(1)] * (
                4 * sp_size - ((t - 1) % (4 * sp_size))
            )
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            assert (videos.size(1) - 1) % (4 * sp_size) == 0
            return videos

    # classifier-free guidance
    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = 0.0
    # sampling steps
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # set random seed
    set_seed(seed, same_across_ranks=True)

    video_transform = Compose(
        [
            NaResize(
                resolution=(
                    res_h * res_w
                )
                ** 0.5,
                mode="area",
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

        # read condition latents
    cond_latents = []

    print(f"Read video size: {video.size()}")
    cond_latents.append(video_transform(video.to(get_device())))

    ori_lengths = [video.size(1) for video in cond_latents]
    input_videos = cond_latents
    cond_latents = [[cut_videos(video, 1) for video in cond_latents]]

    runner.dit.to("cpu")
    print(f"Encoding videos: {list(map(lambda x: x.size(), cond_latents))}")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode(cond_latents)
    runner.vae.to("cpu")
    runner.dit.to(get_device())

    for i, emb in enumerate(text_embeds["texts_pos"]):
        text_embeds["texts_pos"][i] = emb.to(get_device())
    for i, emb in enumerate(text_embeds["texts_neg"]):
        text_embeds["texts_neg"][i] = emb.to(get_device())

    samples = generation_step(runner, text_embeds, cond_latents=cond_latents)
    runner.dit.to("cpu")
    del cond_latents

    # dump samples to the output directory
    if get_sequence_parallel_rank() == 0:
        for path, input, sample, ori_length in zip(
            videos, input_videos, samples, ori_lengths
        ):
            if ori_length < sample.shape[0]:
                sample = sample[:ori_length]
            filename, file_extension = os.path.splitext(path)
            filename = os.path.join(tgt_path, filename + "_" + str(start_image) + "_" + str(length) + file_extension)
            # color fix
            input = (
                rearrange(input[:, None], "c t h w -> t c h w")
                if input.ndim == 3
                else rearrange(input, "c t h w -> t c h w")
            )
            if use_colorfix:
                sample = wavelet_reconstruction(
                    sample.to("cpu"), input[: sample.size(0)].to("cpu")
                )
            else:
                sample = sample.to("cpu")
            sample = (
                rearrange(sample[:, None], "t c h w -> t h w c")
                if sample.ndim == 3
                else rearrange(sample, "t c h w -> t h w c")
            )
            sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
            sample = sample.to(torch.uint8).numpy()

            if sample.shape[0] == 1:
                mediapy.write_image(filename, sample.squeeze(0))
            else:
                mediapy.write_video(
                    filename, sample, fps=fps
                )
    gc.collect()
    torch.cuda.empty_cache()

def download_weight(model):
    save_dir = "ckpts/"
    
    if model == "seedvr2_ema_3b.pth" and os.path.exists("ckpts/seedvr2_ema_3b.pth"):
        return

    if model == "seedvr2_ema_3b.pth" and os.path.exists("ckpts/seedvr2_ema_7b.pth"):
        return

    if model == "seedvr2_ema_3b.pth":
        repo_id = "ByteDance-Seed/SeedVR2-3B"
    else:
        repo_id = "ByteDance-Seed/SeedVR2-7B"

    # cache_dir = save_dir + "/cache"

    snapshot_download(local_dir=save_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
    )


class SeedVR2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGES", ),
                "model": (["seedvr2_ema_3b.pth", "seedvr2_ema_7b.pth",], "seedvr2_ema_3b.pth"),
                "seed": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 1}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 720, "min": 1, "max": 1440, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 2.0, "step": 0.1}),
                "positive_prompt":("TEXT", {"default": 0, "multiline":True}),
                "negative_prompt":("TEXT", {"default": 0, "multiline":True})

            },
        }
    RETURN_NAMES = ("images", )
    RETURN_TYPES = ("IMAGES",)
    FUNCTION = "execute"
    CATEGORY = "SEEDVR2"

    def execute(self, images, model, seed, width, height, skip_first_frames, frame_load_cap, cfg_scale, positive_prompt, negative_prompt):
        download_weight(model)
        runner = configure_runner(model, 1)
        images = generation_loop(runner, images, cfg_scale, seed, height, width, positive_prompt, negative_prompt)
        return [images]



NODE_CLASS_MAPPINGS = {
    "SeedVR2": SeedVR2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2": "SeedVR2 Video Upscaler",
}
