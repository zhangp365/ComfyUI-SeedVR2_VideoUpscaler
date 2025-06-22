# ComfyUI-SeedVR2_VideoUpscaler

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

A Non official custom nodes for ComfyUI that enables Upscale Video generation using [SeedVR2](https://github.com/ByteDance-Seed/SeedVR).

<img src="docs/demo_01.jpg" width="700px">
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

| Model  | Images | Resolution          | Batch Size | Time (s) | FPS  |
| ------ | ------ | ------------------- | ---------- | -------- | ---- |
| 7B FP8 | 3      | 512×768 → 1080×1620 | 1          | 58.10    | 0.05 |
| 7B FP8 | 15     | 512×768 → 1080×1620 | 5          | 135.63   | 0.11 |
| 7B FP8 | 27     | 512×768 → 1080×1620 | 9          | 163.22   | 0.17 |
| 7B FP8 | 39     | 512×768 → 1080×1620 | 13         | 189.36   | 0.21 |
| 7B FP8 | 51     | 512×768 → 1080×1620 | 17         | 215.80   | 0.24 |
| 7B FP8 | 63     | 512×768 → 1080×1620 | 21         | 241.79   | 0.26 |
| 7B FP8 | 75     | 512×768 → 1080×1620 | 25         | 267.93   | 0.28 |
| 7B FP8 | 123    | 512×768 → 1080×1620 | 41         | 373.60   | 0.33 |
| 7B FP8 | 243    | 512×768 → 1080×1620 | 81         | 642.20   | 0.38 |
| 7B FP8 | 363    | 512×768 → 1080×1620 | 121        | 913.61   | 0.40 |
| 7B FP8 | 453    | 512×768 → 1080×1620 | 151        | 1132.01  | 0.40 |
| 7B FP8 | 633    | 512×768 → 1080×1620 | 211        | 1541.09  | 0.41 |
| 7B FP8 | 903    | 512×768 → 1080×1620 | 301        | OOM      | OOM  |

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

- Use a lot of VRAM, it will take all!!
- Processing speed depends on GPU capabilities

## Credits

- Original [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) implementation

# 📜 License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
