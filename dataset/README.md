# Dataset Structure Guidelines

The FMCW Radar 3D Hand Pose Estimation dataset relies on synchronized radar tensors, 2D MediaPipe hand keypoint JSONs, and offline 3D depth ground truth.

## Expected Directory Layout

Place your data files under `dataset/raw_ds/` following this structure:

```text
dataset/raw_ds/
├── matched_radar/
│   ├── frame_0001.npy       # Shape: (8, 64, 64) - Float32
│   ├── frame_0002.npy
│   └── ...
├── matched_camera/          # (Optional) Synchronized RGB images for 3D GT creation
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── matched_camera_hand_json/# 2D ground truth keypoints (MediaPipe)
│   ├── frame_0001.json
│   ├── frame_0002.json
│   └── ...
└── matched_camera_hand_json_3d/ # Output generated 3D ground truth keypoint JSONs
    ├── frame_0001.json
    ├── frame_0002.json
    └── ...
```

## Ground Truth JSON Schema (3D)

Each generated `.json` file contains:
- `hand_or_no_hand`: `"hand"` or `"no hand"`
- `hands`: List of detected hands containing:
  - `handedness`: `"Left"` or `"Right"`
  - `confidence_score`: float
  - `keypoints`: List of 21 hand keypoints with `id`, `x_norm`, `y_norm`, `pixel_x`, `pixel_y`, and estimated metric depth `z` (in cm).

See `output/example_output.json` for a complete reference annotation.
