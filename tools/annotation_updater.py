import json
import os
import copy
import numpy as np

def load_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def scale_depth(raw_depth_map: np.ndarray, min_depth_cm: float = 10.0, max_depth_cm: float = 55.0) -> np.ndarray:
    """
    Scales the relative depth map from Depth Anything to an absolute metric depth (in cm).
    Depth Anything outputs larger values for closer objects.
    This function normalizes the depth map and maps it linearly to [min_depth_cm, max_depth_cm].
    
    Args:
        raw_depth_map: The raw relative depth map from the model.
        min_depth_cm: The minimum depth (closest point) in cm.
        max_depth_cm: The maximum depth (furthest point) in cm.
        
    Returns:
        metric_depth_map: The scaled absolute depth map in cm.
    """
    d_min = raw_depth_map.min()
    d_max = raw_depth_map.max()
    
    # Avoid division by zero if depth map is uniform
    if d_max - d_min < 1e-6:
        return np.full_like(raw_depth_map, (min_depth_cm + max_depth_cm) / 2.0)
        
    # Normalize to [0, 1]
    normalized = (raw_depth_map - d_min) / (d_max - d_min)
    
    # Larger raw value -> closer (min_depth)
    # Smaller raw value -> further (max_depth)
    metric_depth = (1.0 - normalized) * (max_depth_cm - min_depth_cm) + min_depth_cm
    
    return metric_depth

def append_depth_to_annotations(original_data: dict, depth_map: np.ndarray) -> dict:
    """
    Appends the 'z' coordinate to each keypoint in the annotation data.
    
    Args:
        original_data: The original 2D JSON data.
        depth_map: The scaled metric depth map (numpy array).
        
    Returns:
        new_data: The updated JSON data with 'z' appended to keypoints.
    """
    # Create a deep copy to ensure original data is not modified
    new_data = copy.deepcopy(original_data)
    
    if new_data.get("hand_or_no_hand") == "hand" and "hands" in new_data:
        height, width = depth_map.shape
        
        for hand in new_data["hands"]:
            if "keypoints" in hand:
                for kp in hand["keypoints"]:
                    px = kp.get("pixel_x")
                    py = kp.get("pixel_y")
                    
                    if px is not None and py is not None:
                        # Ensure pixel coordinates are within bounds
                        px = max(0, min(px, width - 1))
                        py = max(0, min(py, height - 1))
                        
                        # Extract depth value
                        z_val = float(depth_map[py, px])
                        kp["z"] = z_val
                        
    return new_data

def process_annotation(input_json_path: str, output_json_path: str, raw_depth_map: np.ndarray):
    """
    End-to-end function to read, scale depth, append, and save.
    
    Args:
        input_json_path: Path to the original JSON file.
        output_json_path: Path to save the updated JSON file.
        raw_depth_map: The raw depth map tensor/array for this image.
    """
    # Load original 2D annotations
    data = load_json(input_json_path)
    
    # Convert raw depth map to numpy if it's a tensor
    if hasattr(raw_depth_map, 'cpu'):
        raw_depth_map = raw_depth_map.cpu().numpy()
        
    # Scale depth to metric (10-55 cm)
    metric_depth_map = scale_depth(raw_depth_map, min_depth_cm=10.0, max_depth_cm=55.0)
    
    # Append 3D depth
    new_data = append_depth_to_annotations(data, metric_depth_map)
    
    # Save the updated annotations
    save_json(new_data, output_json_path)
