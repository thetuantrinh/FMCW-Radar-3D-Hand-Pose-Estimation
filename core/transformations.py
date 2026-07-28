# transformations.py
import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {
                "radar_tensor": torch.from_numpy(sample["radar_tensor"]).float() if isinstance(sample["radar_tensor"], np.ndarray) else sample["radar_tensor"].clone().detach().float(),
                "keypoints_3d": torch.from_numpy(sample["keypoints_3d"]).float() if isinstance(sample["keypoints_3d"], np.ndarray) else sample["keypoints_3d"].clone().detach().float(),
                "hand_status": torch.from_numpy(sample["hand_status"]).float() if isinstance(sample["hand_status"], np.ndarray) else sample["hand_status"].clone().detach().float(),
                "handedness": torch.from_numpy(sample["handedness"]).float() if isinstance(sample["handedness"], np.ndarray) else sample["handedness"].clone().detach().float(),
                "sample_id": sample.get("sample_id", "")
            }
        else:
            radar, landmarks, hand_presence, handedness = sample
            return [
                torch.from_numpy(radar).float() if isinstance(radar, np.ndarray) else radar.clone().detach().float(),
                torch.from_numpy(landmarks).float() if isinstance(landmarks, np.ndarray) else landmarks.clone().detach().float(),
                torch.from_numpy(hand_presence).float() if isinstance(hand_presence, np.ndarray) else hand_presence.clone().detach().float(),
                torch.from_numpy(handedness).float() if isinstance(handedness, np.ndarray) else handedness.clone().detach().float(),
            ]


class Normalize(object):
    """Normalize input frame with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
        # Buffer torch representation to minimize runtime device casting bottlenecks
        if isinstance(mean, np.ndarray):
            self.mean_tensor = torch.from_numpy(mean).float()
        else:
            self.mean_tensor = torch.tensor(mean).float()
            
        if isinstance(std, np.ndarray):
            self.std_tensor = torch.from_numpy(std).float()
        else:
            self.std_tensor = torch.tensor(std).float()

    def __call__(self, sample):
        if isinstance(sample, dict):
            frame = sample["radar_tensor"]
        else:
            frame = sample[0]
            
        if torch.is_tensor(frame):
            mean = self.mean_tensor.to(frame.device, dtype=frame.dtype)
            std = self.std_tensor.to(frame.device, dtype=frame.dtype)
            frame = (frame - mean) / std
        else:
            frame = (frame - self.mean) / self.std

        if isinstance(sample, dict):
            sample_out = sample.copy()
            sample_out["radar_tensor"] = frame
            return sample_out
        else:
            return [frame, *sample[1:]]


class BackgroundRemoval(object):
    """Remove background array values from radar signal."""

    def __init__(self, background):
        self.background = background

    def __call__(self, sample):
        if isinstance(sample, dict):
            radar_frame = sample["radar_tensor"]
            background_subtracted = radar_frame - self.background
            sample_out = sample.copy()
            sample_out["radar_tensor"] = background_subtracted
            return sample_out
        else:
            radar_frame = sample[0]
            background_subtracted = radar_frame - self.background
            return [background_subtracted, *sample[1:]]