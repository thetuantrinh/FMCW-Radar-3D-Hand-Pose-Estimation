# Installation & Setup on HPC

The following steps must be executed strictly on the **hpc-gw (Gateway Node)** which has internet access. Do not run these commands on the GPU compute nodes.

## 1. Environment Setup

Initialize your environment and activate the correct conda environment:

```bash
/work/tuan.tt19010226/miniconda/bin/conda init
source ~/.bashrc
conda activate py39
```

## 2. Dependency Installation

Install the required packages using pip. (Assuming standard PyTorch is already installed in `py39`. If not, please install torch with cuda support).

```bash
pip install transformers accelerate opencv-python tqdm scipy huggingface_hub
```

## 3. Download the Depth Anything Model Offline

Because the GPU nodes do not have internet access, we must download the model on the gateway node first.

We have provided a helper script to automatically download the `DepthAnythingV2-Large` model (or Base) to a local directory.

Run the following command:

```bash
cd /work/tuan.tt19010226/HPE-3D/
python3 tools/download_model.py --model depth-anything/Depth-Anything-V2-Large-hf --save_dir ./weights/Depth-Anything-V2-Large-hf
```

This will save all model weights and configurations to `./weights/Depth-Anything-V2-Large-hf`.

You can now proceed to the GPU nodes (e.g., `ssh hpc23`) for offline depth generation.
