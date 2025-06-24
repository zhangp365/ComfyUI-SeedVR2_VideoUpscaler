# ComfyUI-SeedVR2_VideoUpscaler

[![View Code](https://img.shields.io/badge/📂_View_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)

A Non official custom nodes for ComfyUI that enables Upscale Video/Images generation using [SeedVR2](https://github.com/ByteDance-Seed/SeedVR).

<img src="docs/demo_01.jpg">
<img src="docs/demo_02.jpg">

<img src="docs/usage.png">

## 🆙 Todo

- Fixed unloading the 3B model when the process is finished (sorry about that, I'm trying to find out what's going on)

## 🚀 Updates

**2025.06.24**

- 🚀 Speed up the process until x4 (see new benchmark)

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

- A Huge VRAM capabilities is better, from my test, even the 3B version need a lot of VRAM at least 18GB.
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

2. things to know

**temporal consistency** : at least a batch_size of 5 is required to activate temporal consistency

2. Configure the node parameters:

   - `model`: Select your 3B or 7B model
   - `seed`: a seed but it generate another seed from this one
   - `new_width`: New desired Width, will keep ration on height
   - `cfg_scale`:
   - `batch_size`: VERY IMPORTANT!, this model consume a lot of VRAM, All your VRAM, even for the 3B model, so for GPU under 24GB VRAM keep this value Low, good value is "1" without temporal consistency
   - `preserve_vram`: for VRAM < 24GB, If true, It will unload unused models during process, longer but works, otherwise probably OOM with

## Performance

**NVIDIA H100 93GB VRAM** (values in parentheses are from the previous benchmark):

| Images | Resolution          | Batch Size | Time fp8 (s)     | FPS fp8     | Time fp16 (s)    | FPS fp16    |
| ------ | ------------------- | ---------- | ---------------- | ----------- | ---------------- | ----------- |
| 3      | 512×768 → 1080×1620 | 1          | 10.18 (58.10)    | 0.29 (0.05) | 10.67 (60.13)    | 0.28 (0.05) |
| 15     | 512×768 → 1080×1620 | 5          | 26.71 (135.63)   | 0.56 (0.11) | 27.75 (144.18)   | 0.54 (0.10) |
| 27     | 512×768 → 1080×1620 | 9          | 33.97 (163.22)   | 0.79 (0.17) | 35.08 (177.61)   | 0.77 (0.15) |
| 39     | 512×768 → 1080×1620 | 13         | 41.01 (189.36)   | 0.95 (0.21) | 42.08 (210.11)   | 0.93 (0.19) |
| 51     | 512×768 → 1080×1620 | 17         | 48.12 (215.80)   | 1.06 (0.24) | 49.44 (242.64)   | 1.03 (0.21) |
| 63     | 512×768 → 1080×1620 | 21         | 55.40 (241.79)   | 1.14 (0.26) | 56.70 (275.55)   | 1.11 (0.23) |
| 75     | 512×768 → 1080×1620 | 25         | 62.60 (267.93)   | 1.20 (0.28) | 63.80 (308.51)   | 1.18 (0.24) |
| 123    | 512×768 → 1080×1620 | 41         | 91.38 (373.60)   | 1.35 (0.33) | 92.90 (440.01)   | 1.32 (0.28) |
| 243    | 512×768 → 1080×1620 | 81         | 164.25 (642.20)  | 1.48 (0.38) | 166.09 (780.20)  | 1.46 (0.31) |
| 363    | 512×768 → 1080×1620 | 121        | 238.18 (913.61)  | 1.52 (0.40) | 239.80 (1114.32) | 1.51 (0.33) |
| 453    | 512×768 → 1080×1620 | 151        | 296.52 (1132.01) | 1.53 (0.40) | 298.65 (1384.86) | 1.52 (0.33) |
| 633    | 512×768 → 1080×1620 | 211        | 406.65 (1541.09) | 1.56 (0.41) | 409.44 (1887.62) | 1.55 (0.34) |
| 903    | 512×768 → 1080×1620 | 301        | OOM (OOM)        | OOM (OOM)   | OOM (OOM)        | OOM (OOM)   |

**NVIDIA RTX4090 24GB VRAM** (preserved_vram=off)
| Model | Images | Resolution | Batch Size | Time (seconds) | FPS | Note |
| ------------------------- | ------ | ------------------- | ---------- | -------------- | --- | --- |
| 3B fp8 | 5 | 512x768 → 1080x1620 | 1 | 22.52 | 0.22 | |
| 3B fp16 | 5 | 512x768 → 1080x1620 | 1 | 27.84 | 0.18 | |
| 7B fp8 | 5 | 512x768 → 1080x1620 | 1 | 75.51 | 0.07 | |
| 7B fp16 | 5 | 512x768 → 1080x1620 | 1 | 78.93 | 0.06 | |
| 3B fp8 | 10 | 512x768 → 1080x1620 | 5 | 39.75 | 0.15 | preserve_memory=on|
| 3B fp8 | 20 | 512x768 → 1080x1620 | 1 | 65.40 | 0.31 | |
| 3B fp16 | 20 | 512x768 → 1080x1620 | 1 | 91.12 | 0.22 | |
| 3B fp8 | 20 | 512x768 → 1280x1920 | 1 | 89.10 | 0.22 | |
| 3B fp8 | 20 | 512x768 → 1480x2220 | 1 | 136.08| 0.15 | |
| 3B fp8 | 20 | 512x768 → 1620x2430 | 1 | 191.28 | 0.10 | preserve_memory=on without GPU overload so longer 320sec |

## Limitations

- Use a lot of VRAM, it will take all!!
- Processing speed depends on GPU capabilities

## Credits

- Original [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) implementation

# 📜 License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
