# Personal Experiment: Real-Time Depth Anything V3 with NVIDIA DeepStream

This repository contains a personal project/experiment. In this project, real-time **Depth Anything V3** monocular depth estimation has been applied to an **RTSP camera stream (with dewarping)**.

> **Note:** Only the specific files modified for this custom implementation have been uploaded to this repository. For the rest of the project structure and complete codebase, please refer to the main repository: [https://github.com/Raessan/depth_anything_v3_deepstream](https://github.com/Raessan/depth_anything_v3_deepstream).

---

## Pre-trained Models

To save time during setup and build, pre-trained models can be downloaded from the Google Drive link below:

📁 **Google Drive Download Link:**  
[Download Models](https://drive.google.com/drive/folders/1hdbGWJG1Kb8YGN_mjaQZUVnpi5nHNYk6?usp=sharing)

---

## Key Features & Details

- **Application:** Real-time Depth Anything V3 depth estimation on an RTSP live camera stream with dewarping.
- **Base Repository:** [depth_anything_v3_deepstream](https://github.com/Raessan/depth_anything_v3_deepstream)
- **DeepStream Pipeline:** Production-level DeepStream 8.0 / TensorRT pipeline with CUDA-accelerated visualization.

---

## Original Features & Usage

### Features
- **Real-time inference**: Full TensorRT FP16 pipeline via DeepStream 8.0.
- **Multiple models supported**: DA3METRIC-LARGE, DA3MONO-LARGE, DA3-SMALL/BASE/LARGE-1.1.
- **GPU-accelerated visualization**: Percentile-normalized inferno colormap via CUDA.
- **Side-by-side view**: `--show-original` option to tile original frames next to depth maps.

### Benchmarks (RTX 5070 Laptop GPU, FP16, 504×504, batch size 1)
| Model | Params | FPS | GPU Mem |
|-------|--------|-----|---------|
| DA3-SMALL | 34M | 30 | 522 MiB |
| DA3-BASE | 135M | 30 | 744 MiB |
| DA3-LARGE-1.1 | 411M | 30 | 1312 MiB |
| DA3METRIC-LARGE | 334M | 30 | 1208 MiB |
| DA3MONO-LARGE | 334M | 30 | 1208 MiB |

---

## Build & Execution

1. **Clone Main Repo Structure:** Refer to [upstream repo](https://github.com/Raessan/depth_anything_v3_deepstream) for full code setup.
2. **Download Models:** Place downloaded ONNX/Engine files inside `models/`.
3. **Build:**
   ```bash
   cd depth_anything_v3_deepstream
   mkdir -p build && cd build
   cmake ..
   make -j$(nproc)
