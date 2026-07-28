# FMCW Radar 3D Hand Pose Estimation (HPE-3D)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA Supported](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An end-to-end deep learning repository for **3D Hand Pose Estimation directly from FMCW Radar signals**, featuring an offline pseudo-labeling pipeline using **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** to automatically generate metric 3D hand ground truth annotations.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Experimental System Environment](#-experimental-system-environment)
- [Dataset Preparation](#-dataset-preparation)
- [Pipeline Workflow](#-pipeline-workflow)
  - [1. Download Offline Depth Model](#1-download-offline-depth-model)
  - [2. Generate 3D Ground Truth](#2-generate-3d-ground-truth)
  - [3. Train Radar HPE-3D Network](#3-train-radar-hpe-3d-network)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. Inference & Video Generation](#5-inference--video-generation)
- [License](#-license)

---

## 🌐 Overview

Traditional vision-based 3D Hand Pose Estimation (HPE) suffers in poor lighting conditions, thermal variation, and privacy-sensitive scenarios. FMCW (Frequency-Modulated Continuous-Wave) Radar provides a robust, privacy-preserving alternative that works in complete darkness.

This repository provides:
1. **Radar 3D Hand Pose Estimation Model (`RadarHPE3DNet`)**: A 2D Convolutional Neural Network with Residual Blocks designed to predict **21 3D hand keypoints $(x, y, z)$**, **hand presence**, and **handedness** directly from $(8, 64, 64)$ radar tensor frames.
2. **Offline 3D Pseudo-Labeling Pipeline**: Automatically converts 2D [MediaPipe](https://github.com/google-ai-edge/mediapipe) hand keypoints (from the [2D Hand Pose Dataset](https://github.com/thetuantrinh/UWB-Radar-Hand-Pose-Estimation/tree/main/dataset)) into 3D metric annotations by extracting depth maps using **[Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)** calibrated to hand interaction distances ($10\text{--}55\text{ cm}$).

---

## ✨ Key Features

- **Direct Radar-to-3D Inference**: Camera images are required *only* during offline 3D dataset generation. Inference operates **100% on radar input**.
- **Metric Depth Scaling**: Calibrates relative monocular depth maps into metric depth ($z$ in cm) tailored to close-range hand interactions ($10\text{--}55\text{ cm}$).
- **Production-Ready Architecture**: Features modular PyTorch design, multi-GPU DataParallel support, mixed-precision (FP16), TensorBoard logging, and automatic checkpointing.
- **HPC Offline Compatibility**: Built to execute in strict HPC environments where GPU compute nodes operate completely offline without internet connectivity.

---

## 🏗 System Architecture

```text
               +-------------------------------------------------------+
               |            OFFLINE 3D DATASET GENERATION              |
               |                                                       |
  RGB Images --+--> Depth Anything V2 --> Depth Map (z)                |
               |                               |                       |
2D Hand JSONs -+-------------------------------+---> 3D Hand GT JSONs |
               +-------------------------------------------------------+
                                                               |
                                                               v
               +-------------------------------------------------------+
               |              RADAR HPE-3D MODEL TRAINING              |
               |                                                       |
Radar Input ---+--> RadarHPE3DNet Backbone ----------------------------+
(8, 64, 64)    |   ├── ResNet Feature Extractor                        |
               |   ├── 3D Keypoint Regressor  --> 21 x 3 Coordinates   |
               |   ├── Hand Presence Head     --> Sigmoid Confidence   |
               |   └── Handedness Head        --> Left / Right Class   |
               +-------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
FMCW-Radar-3D-Hand-Pose-Estimation/
├── core/                         # Core Deep Learning Modules
│   ├── __init__.py
│   ├── network.py                # RadarHPE3DNet PyTorch Network Architecture
│   ├── dataloader.py             # Radar & 3D GT JSON Pair DataLoader
│   ├── training_engine.py        # Trainer class with multi-loss logic
│   ├── transformations.py        # Tensor transforms & augmentations
│   ├── parser.py                 # Hyperparameter CLI argument parser
│   ├── logger.py                 # Formatted logging system
│   └── helper.py                 # Dataset splitting & utility functions
├── scripts/                      # Execution Scripts
│   ├── train.py                  # Model training pipeline
│   ├── val.py                    # Evaluation script (MPJPE, PCK@th)
│   ├── infer.py                  # Radar inference & MP4 video renderer
│   └── eval_norm.py              # Depth normalization verification
├── tools/                        # Pipeline Tools & 3D Ground Truth Tools
│   ├── download_model.py         # HuggingFace model downloader for offline use
│   ├── depth_engine.py           # Offline Depth Anything V2 engine
│   ├── generate_depth.py         # Batch 3D ground truth generator
│   ├── annotation_updater.py     # Depth scaling & 3D JSON updater
│   └── visualize_dataset.py      # Dataset visualization tool
├── demos/                        # Demo Utilities & Scripts
│   ├── d_l_demo_v1.py            # Live / offline display demo script
│   └── dark_demo.py              # Low-light radar demo script
├── docs/                         # HPC Documentation & Setup Guides
│   ├── INSTALL.md                # Gateway node setup instructions
│   ├── RUN.md                    # GPU node execution guide
│   └── prompt/                   # Specifications & task prompts
├── dataset/                      # Dataset Directory Placeholder
│   └── README.md                 # Dataset organization guide
├── output/                       # Example Output Annotations
│   └── example_output.json       # Sample 3D JSON annotation format
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
├── README.md                     # Repository documentation
└── requirements.txt              # Dependencies file
```

---

## 💻 Installation & Setup

### Environment Requirements

- Linux OS (Ubuntu 20.04 / 22.04 or HPC Linux cluster)
- Python 3.9+
- CUDA 11.8 / 12.1 compatible GPU (NVIDIA RTX 3090, A100, V100, etc.)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/FMCW-Radar-3D-Hand-Pose-Estimation.git
cd FMCW-Radar-3D-Hand-Pose-Estimation

# 2. Create and activate a Conda environment
conda create -n hpe3d python=3.9 -y
conda activate hpe3d

# 3. Install PyTorch with CUDA support (adjust index URL if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## 🧪 Experimental System Environment

- **Training & Testing**: AlmaLinux 8.5 (Arctic Sphynx) (NVIDIA DGX A100-SXM4-40GB)
- **Inference Deployment**: NVIDIA Jetson Nano (Tegra X1, Quad-core ARM Cortex-A57 CPU, 128-core Maxwell GPU)

---

## 📊 Dataset Preparation

Organize your raw dataset under `dataset/raw_ds/`:

```text
dataset/raw_ds/
├── matched_radar/                 # Radar numpy tensors (.npy, shape: 8x64x64)
├── matched_camera/                # Synchronized RGB frames (.jpg / .png)
├── matched_camera_hand_json/      # 2D MediaPipe annotations (.json)
└── matched_camera_hand_json_3d/   # Generated 3D annotations (.json)
```

The 2D ground truth labels can be obtained from the [2D Radar Hand Pose Estimation Dataset](https://github.com/thetuantrinh/UWB-Radar-Hand-Pose-Estimation/tree/main/dataset).
Refer to [`dataset/README.md`](dataset/README.md) for full schema details.

---

## 🚀 Pipeline Workflow

### 1. Download Offline Depth Model

On a node with internet access (e.g. HPC Gateway), download **[Depth Anything V2 Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)**:

```bash
python3 tools/download_model.py \
    --model depth-anything/Depth-Anything-V2-Large-hf \
    --save_dir ./tools/weights/Depth-Anything-V2-Large-hf
```

### 2. Generate 3D Ground Truth

On a GPU-enabled node (offline compatible), extract 3D metric annotations:

```bash
python3 tools/generate_depth.py \
    --image_dir ./dataset/raw_ds/matched_camera \
    --hand_json_dir ./dataset/raw_ds/matched_camera_hand_json \
    --radar_dir ./dataset/raw_ds/matched_radar \
    --output_dir ./dataset/raw_ds/matched_camera_hand_json_3d \
    --model_path ./tools/weights/Depth-Anything-V2-Large-hf \
    --batch_size 16 \
    --device cuda
```

### 3. Train Radar HPE-3D Network

Train the `RadarHPE3DNet` network directly on radar tensors supervised by 3D GT:

```bash
python3 scripts/train.py \
    --data_dir ./dataset/raw_ds \
    --radar_dir matched_radar \
    --json_dir matched_camera_hand_json_3d \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --saved_model_path results/run_latest/checkpoints
```

### 4. Model Evaluation

Evaluate the trained checkpoint against the validation split to compute MPJPE and PCK metrics:

```bash
python3 scripts/val.py \
    --data_dir ./dataset/raw_ds \
    --radar_dir matched_radar \
    --json_dir matched_camera_hand_json_3d \
    --checkpoint_path results/run_latest/checkpoints/best_model.pt \
    --batch_size 32
```

### 5. Inference & Video Generation

Run inference on radar frames and render side-by-side video comparisons against GT:

```bash
python3 scripts/infer.py
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

