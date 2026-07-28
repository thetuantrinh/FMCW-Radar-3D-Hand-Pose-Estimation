You are an expert Computer Vision, PyTorch, and HPC software engineer.

Your task is to extend the HPE (Hand Pose Estimation) project by automatically generating 3D ground-truth annotations from synchronized RGB images using an best offline Depth Anything Model.

===============================================================================
PROJECT OVERVIEW
===============================================================================

Current HPE predicts only:
- hand_or_no_hand
- handedness (left/right)
- 21 hand keypoints (x, y)

The current dataset DOES NOT contain 3D information such as the Depth of 21 hand keypoints (x, y, z).

The objective is to automatically generate these 3D ground truth annotations from synchronized RGB images so they can later be used for training a 3D HPE model.
To estimate the depth of each keypoint, you must:
1. Run the offline Depth Anything Model (DAM) on the RGB image to generate a full depth map.
2. Map the 2D pixel positions (x, y) of the existing 21 keypoints onto the generated depth map to extract their corresponding depth (z) values.

===============================================================================
DEPTH SCALING REQUIREMENT
===============================================================================

The raw depth output from the Depth Anything Model (DAM) is typically relative or unscaled. 
You MUST implement a scaling/calibration mechanism to map the predicted depth values to absolute metric depth. This calibration must be optimized for maximum accuracy in the close-range domain of 10-55 cm, as this is the typical interaction range for hands in the RadarLM-HPE dataset.

===============================================================================
DATASET STRUCTURE
===============================================================================

The dataset contains synchronized modalities.

-------------------------------------------------------------------------------
1. RGB Images
-------------------------------------------------------------------------------
Directory: `/work/tuan.tt19010226/HPE-3D/raw_ds/matched_camera/`

These RGB images are the ONLY input to the Depth Anything Model.

Search recursively. Support:
- png
- jpg
- jpeg

-------------------------------------------------------------------------------
2. Existing Ground Truth
-------------------------------------------------------------------------------
Directory: `/work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_camera_hand_json/`

This directory contains multiple `.json` files which are the authoritative 2D truth labels. 
Each JSON annotation contains existing labels such as:
- hand_or_no_hand
- handedness
- 21 hand keypoints (x, y)
- other hand annotations

DO NOT overwrite or modify these original existing files.
Instead, you must create NEW `.json` files in the specified output directory. These new files must exactly mirror the format of the original files, but you must add the new `z` (depth) dimension to the existing 21 keypoints.

-------------------------------------------------------------------------------
3. Radar Data
-------------------------------------------------------------------------------
Directory: `/work/tuan.tt19010226/HPE-3D/dataset/raw_ds/matched_radar/`

Radar data are synchronized:
Radar ↔ RGB image ↔ Hand JSON

Radar files are NOT used during depth generation. However, the implementation must preserve one-to-one correspondence between:
- Radar
- RGB
- Hand JSON
- 3D Annotations

===============================================================================
HPC EXECUTION POLICY
===============================================================================

The HPC contains two environments.

-------------------------------------------------------------------------------
1. hpc-gw (Gateway Node)
-------------------------------------------------------------------------------
Internet available.
Purpose:
- Install Python packages and dependencies
- Download pretrained models (Depth Anything Model)
- Download tokenizer/configs (if any)
- Prepare HuggingFace cache

STRICTLY FORBIDDEN:
- inference
- training
- benchmarking
- depth generation
- GPU computation

Never execute the model on hpc-gw.

-------------------------------------------------------------------------------
2. GPU Compute Nodes
-------------------------------------------------------------------------------
Access: `ssh hpc23`
GPU nodes have NO Internet.
Purpose:
- inference
- depth generation
- benchmarking
- future training

Never install packages or download models on GPU nodes.

===============================================================================
OFFLINE REQUIREMENT
===============================================================================

GPU nodes must run completely offline.
Requirements:
- load models only from local disk
- local_files_only=True (if applicable)
- never contact external servers or HuggingFace
- never download weights/configs

If model files do not exist, terminate immediately and print:
"Model not found locally. Please download it on hpc-gw first."

===============================================================================
GPU ONLY REQUIREMENT
===============================================================================

GPU execution is mandatory.
Under NO circumstance should inference or training run on CPU.

Requirements:
- detect CUDA before loading the model
- If CUDA unavailable, terminate immediately.
- Never use `device = "cuda" if available else "cpu"`. Never allow CPU fallback.
- Load model directly on CUDA. All tensors/parameters must be on CUDA.
- Abort if any parameter resides on CPU.

Print:
- GPU name, CUDA version, GPU memory, GPU count

Support:
- DataParallel or DistributedDataParallel
- Enable FP16/BF16 whenever supported.

===============================================================================
PERFORMANCE
===============================================================================

Support:
- batch inference
- configurable batch size
- pinned memory
- asynchronous loading
- tqdm
- resume interrupted jobs
- skip processed samples
- maximize GPU utilization

===============================================================================
COMMAND LINE ARGUMENTS
===============================================================================

Provide:
--image_dir
--hand_json_dir
--radar_dir
--output_dir
--model_path
--batch_size
--device
--recursive
--overwrite
--num_workers

===============================================================================
LOGGING & FINAL REPORT
===============================================================================

Use logging. Display:
- current image, processed count, remaining time, GPU information

Print Final Report:
- Total images, Processed, Skipped, Failed
- Average inference time, Total runtime, Peak GPU memory

===============================================================================
DEPENDENCIES & RUNNING
===============================================================================

If dependencies are missing, generate installation commands.

Environment setup (on both hpc-gw and GPU node):
`/work/tuan.tt19010226/miniconda/bin/conda init`
`source ~/.bashrc`
`conda activate py39`

Install ONLY on hpc-gw using `pip install` or `conda install`.

Running on GPU (hpc23):
`ssh hpc23`
`cd /work/tuan.tt19010226/HPE-3D/`
... activate environment ...
`python3 tools/generate_depth.py [arguments]`

===============================================================================
CODE QUALITY
===============================================================================

The implementation must:
- follow PEP8
- include type hints, docstrings, comments, logging, exception handling
- be modular and production-ready
- be fully offline on GPU nodes
- preserve existing annotations exactly
- append only 3D depth information

===============================================================================
DELIVERABLES
===============================================================================

Produce:
1. Complete analysis of the existing Radar-HPE project and depth extraction task.
2. Analyze the current dataset organization and verify correspondence among RGB images, existing hand-label JSON files, and radar data.
3. Recommend the best offline Depth Anything Model (e.g., DepthAnythingV2) with a detailed justification.
4. Generate installation commands for hpc-gw only.
5. Generate model download instructions for hpc-gw only.
6. Implement the complete depth generation pipeline.
7. Create all required Python modules.
8. Create `tools/generate_depth.py`.
9. Generate example execution commands.
10. Generate example output JSON showing the appended 3D (z) coordinates.
11. Verify that:
    - existing labels are never modified
    - 3D depth information is appended correctly
    - GPU execution is enforced (no CPU fallback)
    - no Internet access is required on GPU nodes
12. Produce a final verification checklist confirming the implementation is complete, executable, production-ready, fully offline, and compatible with future training.

Do not leave TODOs, placeholders, mock code, or partially implemented functions. Deliver complete, production-quality code ready to run on the HPC environment.