# ComfyUI-SeedVR2_VideoUpscaler

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

A Non official custom nodes for ComfyUI that enables Upscale Video generation using [SeedVR2](https://github.com/ByteDance-Seed/SeedVR).

<video width="700px" controls>
  <source src="https://github.com/user-attachments/assets/8fbd6c1f-4246-4dbe-8819-4e684490c5f2" type="video/mp4">
  Your browser does not support the video tag.
</video>

<img src="docs/usage.png" width="700px">

## 🆙 Todo

- Fixed unloading the 3B model when the process is finished (sorry about that, I'm trying to find out what's going on)

## 🚀 Updates

**2025.06.22**

- 💪 FP8 compatibility !
- 🚀 Speed Up all Process
- 🚀 less VRAM consumption (Stay high, batch_size=1 for RTX4090 max, I'm trying to fix that)
- 🛠️ Better benchmark coming soon

**2025.06.20**

- 🛠️ Initial push

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

3. Models

   Will be automtically download into :
   `models/SEEDVR2`

   or can be found here ([MODELS](https://huggingface.co/numz/SeedVR2_comfyUI/tree/main))

## Usage

1. In ComfyUI, locate the **SeedVR2 Video Upscaler** node in the node menu.

<img src="docs/node.png" width="100%">

2. Configure the node parameters:

   - `model`: Select your 3B or 7B model
   - `seed`: a seed but it generate another seed from this one
   - `new_width`: New desired Width, will keep ration on height
   - `cfg_scale`:
   - `batch_size`: VERY IMPORTANT!, this model consume a lot of VRAM, All your VRAM, even for the 3B model, so for GPU under 24GB VRAM keep this value Low, good value is "1", you have
   - `vram_mode`: It will try to help with VRAM, but 'auto' is good

## Performance

**NVIDIA H100 93GB VRAM**
| Model | Images | Resolution | Batch Size | Time (seconds) | FPS |
| ------------------------- | ------ | ------------------- | ---------- | -------------- | --- |
| 3B fp16 | 97 | 512x768 → 1280x1920 | 50 | 338.63 | 0.29 |
| 3B fp16 | 97 | 512x768 → 1280x1920 | 10 | 540.22 | 0.18 |
| 3B fp16 | 97 | 512x768 → 720x1080 | 10 | 183.64 | 0.53 |
| 7B fp16 | 50 | 512x768 → 1080x1620 | 50 | 166.89 | 0.30 |
| 7B fp16 | 97 | 512x768 → 1080x1620 | 97 | 146.72 | 0.66 |
| 7B fp16 | 200 | 512x768 → 1080x1620 | 200 | 266.14 | 0.75 |

**NVIDIA RTX4090 24GB VRAM**
| Model | Images | Resolution | Batch Size | Time (seconds) | FPS |
| ------------------------- | ------ | ------------------- | ---------- | -------------- | --- |
| 3B fp8 | 20 | 512x768 → 1080x1620 | 1 | 144.23 | 0.14 |
| 3B fp16 | 20 | 512x768 → 1080x1620 | 1 | 214.82 | 0.09 |
| 3B fp8 | 20 | 512x768 → 1280x1920 | 1 | 248.38 | 0.08 |
| 3B fp8 | 20 | 512x768 → 1480x2220 | 1 | 319.06| 0.06 |
| 3B fp8 | 20 | 512x768 → 1620x2430 | 1 | 359.28 | 0.06 |
| 7B fp16 | 5 | 512x768 → 1080x1620 | 1 | 124.36 | 0.04 |
| 7B fp8 | 5 | 512x768 → 1080x1620 | 1 | 118.27 | 0.04 |
| 3B fp16 | 5 | 512x768 → 1080x1620 | 1 | 58.23 | 0.08 |
| 3B fp8 | 5 | 512x768 → 1080x1620 | 1 | 56.76 | 0.09 |
| 3B fp8 | 10 | 512x768 → 1080x1620 | 5 | 128.73 | 0,07 |

## Limitations

- Use a lot of VRAM, it will take all!!!!
-
- Processing speed depends on GPU capabilities

## Credits

- Original [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) implementation

# 📜 License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
