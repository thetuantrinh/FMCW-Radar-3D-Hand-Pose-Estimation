# Execution on HPC GPU Nodes

After you have completed the steps in `INSTALL.md` on the gateway node (`hpc-gw`), follow these steps to run the inference pipeline on a GPU compute node.

## 1. Accessing a GPU Node

SSH into the GPU node (e.g., `hpc23`):

```bash
ssh hpc23
```

## 2. Environment Setup

Initialize your environment and activate the correct conda environment:

```bash
/work/tuan.tt19010226/miniconda/bin/conda init
source ~/.bashrc
conda activate py39
```

## 3. Running the Pipeline

Navigate to the project directory:

```bash
cd /work/tuan.tt19010226/HPE-3D/
```

Run the `generate_depth.py` pipeline. Make sure you use the `--model_path` pointing to the directory where you downloaded the model weights on the gateway node.

**Example Command:**

```bash
python3 tools/generate_depth.py \
    --image_dir /work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_camera \
    --hand_json_dir /work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_camera_hand_json \
    --radar_dir /work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_radar \
    --output_dir /work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_camera_hand_json_3d \
    --model_path ./weights/Depth-Anything-V2-Large-hf \
    --batch_size 16 \
    --num_workers 8 \
    --recursive \
    --device cuda
```

### Script Arguments

- `--image_dir`: The root directory containing the RGB images.
- `--hand_json_dir`: The root directory containing the existing 2D ground truth JSONs.
- `--radar_dir`: (Optional) The directory containing synchronized radar data, used purely for correspondence checking if implemented.
- `--output_dir`: The directory to save the newly generated 3D ground truth JSONs.
- `--model_path`: Local path to the downloaded Depth Anything Model.
- `--batch_size`: Batch size for inference (default 8).
- `--num_workers`: Dataloader workers for async image loading.
- `--recursive`: Whether to search recursively in the image directory.
- `--overwrite`: Overwrite existing output files (otherwise skips processed samples).
- `--device`: Target device (MUST be `cuda` or `cuda:X`).
