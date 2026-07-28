import os
import glob
import json
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def process_radar(radar_path, target_height):
    try:
        radar_data = np.load(radar_path)
        radar_2d = np.max(radar_data, axis=0)
        
        radar_min, radar_max = np.min(radar_2d), np.max(radar_2d)
        if radar_max > radar_min:
            radar_norm = ((radar_2d - radar_min) / (radar_max - radar_min) * 255).astype(np.uint8)
        else:
            radar_norm = np.zeros_like(radar_2d, dtype=np.uint8)
            
        radar_color = cv2.applyColorMap(radar_norm, cv2.COLORMAP_JET)
        radar_resized = cv2.resize(radar_color, (target_height, target_height), interpolation=cv2.INTER_NEAREST)
        return radar_resized
    except Exception as e:
        print(f"Error processing radar {radar_path}: {e}")
        return np.zeros((target_height, target_height, 3), dtype=np.uint8)

def process_depth(depth_model, processor, img, device, target_height):
    try:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            
        predicted_depth = outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(target_height, img.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_np = prediction.cpu().numpy()
        
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_norm = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_np, dtype=np.uint8)
            
        # Using INFERNO for depth map
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        return depth_color
    except Exception as e:
        print(f"Error computing depth: {e}")
        return np.zeros((target_height, img.shape[1], 3), dtype=np.uint8)

def draw_annotations(img, json_path):
    if not os.path.exists(json_path):
        cv2.putText(img, "No JSON found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        hand_status = data.get("hand_or_no_hand", "unknown")
        cv2.putText(img, f"Hand: {hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if hand_status.lower() == "hand" and "hands" in data and len(data["hands"]) > 0:
            hand_info = data["hands"][0]
            handedness = hand_info.get("handedness", "Unknown")
            cv2.putText(img, f"Type: {handedness}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            keypoints = hand_info.get("keypoints", [])
            for kp in keypoints:
                x, y, z = kp.get("pixel_x"), kp.get("pixel_y"), kp.get("z")
                if x is not None and y is not None:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
                    if z is not None:
                        cv2.putText(img, f"{z:.1f}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    except Exception as e:
        print(f"Error parsing JSON {json_path}: {e}")
        
    return img

def main():
    parser = argparse.ArgumentParser(description="Visualize Dataset with Radar, Depth, RGB, and 3D Ground Truth")
    parser.add_argument("--radar_dir", type=str, required=True, help="Directory containing radar .npy files")
    parser.add_argument("--camera_dir", type=str, required=True, help="Directory containing camera .png files")
    parser.add_argument("--json_dir", type=str, required=True, help="Directory containing 3D JSON annotations")
    parser.add_argument("--model_path", type=str, default="/work/tuan.tt19010226/HPE-3D/tools/weights/Depth-Anything-V2-Large-hf", help="Path to depth model")
    parser.add_argument("--output_video", type=str, required=True, help="Path to output .mp4 file")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames to process")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for output video (default: 15)")
    args = parser.parse_args()

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Depth model on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.model_path, local_files_only=True).to(device)
    depth_model.eval()

    # Find common stems
    radar_files = glob.glob(os.path.join(args.radar_dir, "*.npy"))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in radar_files]
    stems.sort()
    
    if len(stems) > args.num_frames:
        stems = stems[:args.num_frames]
        
    print(f"Processing {len(stems)} frames...")
    video_writer = None
    
    for stem in tqdm(stems):
        radar_path = os.path.join(args.radar_dir, f"{stem}.npy")
        camera_path = os.path.join(args.camera_dir, f"{stem}.png")
        json_path = os.path.join(args.json_dir, f"{stem}.json")
        
        cam_img = cv2.imread(camera_path)
        if cam_img is None:
            continue
            
        h, w, _ = cam_img.shape
        
        # 1. Process radar
        radar_img = process_radar(radar_path, target_height=h)
        
        # 2. Process depth heatmap
        depth_img = process_depth(depth_model, processor, cam_img, device, h)
        
        # 3. Annotate camera
        cam_annotated = draw_annotations(cam_img.copy(), json_path)
        
        # Concatenate horizontally: [Radar | Depth | Camera]
        combined = np.hstack((radar_img, depth_img, cam_annotated))
        
        # Initialize writer
        if video_writer is None:
            out_h, out_w, _ = combined.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
            video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (out_w, out_h))
            
        video_writer.write(combined)
        
    if video_writer is not None:
        video_writer.release()
        print(f"Saved 3-window video to {args.output_video}")
    else:
        print("Failed to create video.")

if __name__ == "__main__":
    main()
