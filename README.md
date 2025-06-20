# ComfyUI-SeedVR2_VideoUpscaler

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

A Non official custom nodes for ComfyUI that enables Upscale Video generation using [SeedVR2](https://github.com/ByteDance-Seed/SeedVR).

<video width="700px" controls>
  <source src="https://github.com/user-attachments/assets/8fbd6c1f-4246-4dbe-8819-4e684490c5f2" type="video/mp4">
  Your browser does not support the video tag.
</video>

<img src="docs/usage.png" width="700px">

## Features

- High-quality Upscaling
- Suitable for any video length once the right settings are found
- Model Will Be Download Automatically from [Models](https://huggingface.co/numz/SeedVR2_comfyUI/tree/main)

## Requirements

- Last ComfyUI version with python 3.12.9 (may be works with older versions but I haven't test it)

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
```

2. Install the required dependencies:

load venv and :

```bash
pip install -r ComfyUI-SeedVR2_VideoUpscaler/requirements.txt
```

install flash_attn if it ask for it

```bash
pip install -r flash_attn
```

Or use python_embeded :

```bash
python_embeded\python.exe -m pip install -r ComfyUI-SeedVR2_VideoUpscaler/requirements.txt
```

```bash
python_embeded\python.exe -m pip install -r flash_attn
```

## Usage

1. In ComfyUI, locate the **SeedVR2 Video Upscaler** node in the node menu.

<img src="docs/node.png" width="100%">

2. Configure the node parameters:

   - `model`: Select your 3B or 7B model
   - `seed`: a seed but it generate another seed from this one
   - `new_width`: New desired Width, will keep ration on height
   - `cfg_scale`:
   - `batch_size`: VERY IMPORTANT!, this model consume a lot of VRAM, All your VRAM, even for the 3B model, so for GPU under 24GB VRAM keep this value Low, good values are [1,5,9,13,...]
   - `vram_mode`: It will try to help with VRAM, but 'auto' is good

## Performance

1. **NVIDIA H100 93GB VRAM**

3B or 7B spike to 90+GB VRAM!! but fast!!

- 3B Model, 97 images, from 512x768 to 1280x1920, batch_size=50 => Prompt executed in 338.63 seconds
- 3B Model, 97 images, from 512x768 to 1280x1920, batch_size=10 => Prompt executed in 540.22 seconds
- 3B Model, 97 images, from 512x768 to 720x1080, batch_size=10 => Prompt executed in 183.64 seconds
- 7B Model, 50 images, 512x768 to 1080x1620, batch_size=50, Prompt executed in 166.89 seconds
- 7B Model, 97 images, 512x768 to 1080x1620, batch_size=97, Prompt executed in 146.72 seconds
- 7B Model, 200 images, 512x768 to 1080x1620, batch_size=200, Prompt executed in 266.14 seconds

2. **NVIDIA RTX4090 24GB VRAM**

- 3B Model, 20 images, from 512x768 to 1080x1620, batch_size=1, Prompt executed in 1022.26 seconds
-

## Limitations

- Use a lot of VRAM, it will take alllllll!!!!
- Processing speed depends on GPU capabilities

## Know issues

- On windows, sometime when you click "run" it will break, but if you press "run" again it works

## Credits

- Original [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) implementation

# 📜 License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
