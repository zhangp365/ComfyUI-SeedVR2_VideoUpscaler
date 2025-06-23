# ComfyUI-SeedVR2_VideoUpscaler

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

A Non official custom nodes for ComfyUI that enables Upscale Video/Images generation using [SeedVR2](https://github.com/ByteDance-Seed/SeedVR).

<img src="docs/demo_01.jpg">
<img src="docs/demo_02.jpg">

<img src="docs/usage.png">

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
   - `batch_size`: VERY IMPORTANT!, this model consume a lot of VRAM, All your VRAM, even for the 3B model, so for GPU under 24GB VRAM keep this value Low, good value is "1"
   - `preserve_vram`: If true, It will unload unused models during process, longer but works, otherwise probably OOM with

## Performance

**NVIDIA H100 93GB VRAM**

| Images | Resolution          | Batch Size | Time 7B FP8 (s) | Time 7B FP16 (s) | FPS 7B FP8 | FPS 7B FP16 |
| ------ | ------------------- | ---------- | --------------- | ---------------- | ---------- | ----------- |
| 3      | 512×768 → 1080×1620 | 1          | 58.10           | 60.13            | 0.05       | 0.05        |
| 15     | 512×768 → 1080×1620 | 5          | 135.63          | 144.18           | 0.11       | 0.10        |
| 27     | 512×768 → 1080×1620 | 9          | 163.22          | 177.61           | 0.17       | 0.15        |
| 39     | 512×768 → 1080×1620 | 13         | 189.36          | 210.11           | 0.21       | 0.19        |
| 51     | 512×768 → 1080×1620 | 17         | 215.80          | 242.64           | 0.24       | 0.21        |
| 63     | 512×768 → 1080×1620 | 21         | 241.79          | 275.55           | 0.26       | 0.23        |
| 75     | 512×768 → 1080×1620 | 25         | 267.93          | 308.51           | 0.28       | 0.24        |
| 123    | 512×768 → 1080×1620 | 41         | 373.60          | 440.01           | 0.33       | 0.28        |
| 243    | 512×768 → 1080×1620 | 81         | 642.20          | 780.20           | 0.38       | 0.31        |
| 363    | 512×768 → 1080×1620 | 121        | 913.61          | 1114.32          | 0.40       | 0.33        |
| 453    | 512×768 → 1080×1620 | 151        | 1132.01         | 1384.86          | 0.40       | 0.33        |
| 633    | 512×768 → 1080×1620 | 211        | 1541.09         | 1887.62          | 0.41       | 0.34        |
| 903    | 512×768 → 1080×1620 | 301        | OOM             | OOM              | OOM        | OOM         |

**NVIDIA RTX4090 24GB VRAM** (preserved_vram=off)
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
