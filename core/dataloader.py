# dataloader.py
import os
import glob
import json
import numpy as np
from torch.utils.data import Dataset


def load_pair_data(radar_path, json_path):
    """Loads a radar NumPy frame and its corresponding hand tracking JSON ground truth."""
    # 1. Load Radar frame as raw numpy array
    radar_data = np.load(radar_path).astype(np.float32)

    # 2. Load Hand JSON data
    with open(json_path, 'r') as f:
        hand_data = json.load(f)

    # Initialize ground truth variables as numpy arrays
    landmarks_gt = np.zeros((21, 3), dtype=np.float32)
    hand_presence_gt = np.array([0.0], dtype=np.float32)
    handedness_gt = np.array([-1.0], dtype=np.float32)  # -1.0 acts as a 'no hand' sentinel
    sample_id = os.path.splitext(os.path.basename(radar_path))[0]

    if hand_data.get("hand_or_no_hand") == "hand":
        hand_presence_gt[0] = 1.0  # Hand is present
        
        if "hands" in hand_data and len(hand_data["hands"]) > 0:
            primary_hand = hand_data["hands"][0]
            
            # Map Handedness: Left -> 0.0, Right -> 1.0
            handedness_str = primary_hand.get("handedness", "Left")
            handedness_gt[0] = 1.0 if handedness_str == "Right" else 0.0
            
            # Map 21 Keypoints
            for kp in primary_hand.get("keypoints", []):
                kp_id = kp["id"]
                if kp_id < 21:
                    landmarks_gt[kp_id] = [kp["x_norm"], kp["y_norm"], kp.get("z", 0.0)]

    return {
        "radar_tensor": radar_data,
        "keypoints_3d": landmarks_gt,
        "hand_status": hand_presence_gt,
        "handedness": handedness_gt,
        "sample_id": sample_id
    }


class RadarHandTrackingDataset(Dataset):
    """Directory glob-based dataset loader for matched .npy and .json pairs."""
    def __init__(self, radar_dir, json_dir, transforms=None):
        self.radar_dir = radar_dir
        self.json_dir = json_dir
        self.transforms = transforms
        
        self.radar_files = glob.glob(os.path.join(self.radar_dir, "*.npy"))
        self.matched_pairs = []

        for r_path in self.radar_files:
            base_name = os.path.splitext(os.path.basename(r_path))[0]
            j_path = os.path.join(self.json_dir, f"{base_name}.json")
            
            if os.path.exists(j_path):
                self.matched_pairs.append({
                    "radar": r_path,
                    "json": j_path
                })
                
        print(f"Dataset loaded with {len(self.matched_pairs)} radar-to-label pairs.")

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        pair = self.matched_pairs[idx]
        sample = load_pair_data(pair["radar"], pair["json"])
        if self.transforms:
            sample = self.transforms(sample)
        return sample