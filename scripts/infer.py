# predict.py
import os
import glob
import json
import numpy as np
import torch
import cv2

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import DevModel7
from core import transformations

# ==============================================================================
# --- Configuration (Modify these paths directly before running) ---
# ==============================================================================
MODEL_CHECKPOINT = "results/run_20260721_153135/checkpoints/best_model_20260721_153135.pt"
RADAR_INPUT      = "dataset/raw_ds/matched_radar"
JSON_INPUT       = "dataset/raw_ds/matched_camera_hand_json_3d"  # Set to None if you do not want to draw ground truth overlays
OUTPUT_VIDEO     = "prediction_output.mp4"
FPS              = 10
WIDTH            = 640
HEIGHT           = 480
# ==============================================================================


def draw_keypoints(img, landmarks, color, radius=4):
    """Draws keypoint nodes on the frame without any connecting lines."""
    h, w, _ = img.shape
    for kp in landmarks:
        pt = (int(kp[0] * w), int(kp[1] * h))
        cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # No background subtraction or normalization is applied
    to_tensor = transformations.ToTensor()

    # --- Load Model ---
    model = DevModel7().to(device)
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_CHECKPOINT}")
        
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    
    # Support loading direct models or Trainer checkpoint envelopes
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully from {MODEL_CHECKPOINT}")

    # --- Resolve Input Radar Sequences ---
    frames_list = []
    file_basenames = []

    if os.path.isdir(RADAR_INPUT):
        sorted_files = sorted(glob.glob(os.path.join(RADAR_INPUT, "*.npy")))
        print(f"Found {len(sorted_files)} individual radar frames in: {RADAR_INPUT}")
        for file_path in sorted_files:
            frames_list.append(np.load(file_path).astype(np.float32))
            file_basenames.append(os.path.splitext(os.path.basename(file_path))[0])
    elif os.path.isfile(RADAR_INPUT):
        data = np.load(RADAR_INPUT).astype(np.float32)
        if len(data.shape) == 4:
            print(f"Loaded sequential file containing {data.shape[0]} frames.")
            for i in range(data.shape[0]):
                frames_list.append(data[i])
                file_basenames.append(f"frame_{i:04d}")
        elif len(data.shape) == 3:
            print("Loaded a single isolated 3D frame.")
            frames_list.append(data)
            file_basenames.append("single_frame")
        else:
            raise ValueError(f"Invalid input shape format: {data.shape}. Expected (C, H, W) or (N, C, H, W)")
    else:
        raise FileNotFoundError(f"Radar input path not found: {RADAR_INPUT}")

    if len(frames_list) == 0:
        print("No radar data loaded. Exiting.")
        return

    # --- Setup Video Writer ---
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) if os.path.dirname(OUTPUT_VIDEO) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    print(f"Processing frames and compiling output to '{OUTPUT_VIDEO}'...")

    for idx, raw_frame in enumerate(frames_list):
        # 1. Initialize a solid black canvas
        visual_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # 2. Preprocess frame for the network
        processed_frame = raw_frame.copy()

        # Format inputs through transformations
        dummy_label = np.zeros((21, 3))
        dummy_pres = np.array([0.0])
        dummy_hand = np.array([-1.0])
        sample = {
            "radar_tensor": processed_frame,
            "keypoints_3d": dummy_label,
            "hand_status": dummy_pres,
            "handedness": dummy_hand
        }
        
        # Convert NumPy arrays to Tensors
        sample_tensor = to_tensor(sample)
        
        # Extract the radar frame, add the batch dimension: (1, C, H, W), and move to device
        input_tensor = sample_tensor["radar_tensor"].unsqueeze(0).to(device)

        # 3. Model Inference
        with torch.no_grad():
            landmarks_pred, presence_pred, handedness_pred = model(input_tensor)

        pred_landmarks = landmarks_pred[0].cpu().numpy()
        presence_score = presence_pred[0].item()
        handedness_score = handedness_pred[0].item()

        # 4. Draw Ground Truth overlay if folder is provided and matched JSON exists
        if JSON_INPUT and os.path.isdir(JSON_INPUT):
            json_file = os.path.join(JSON_INPUT, f"{file_basenames[idx]}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        gt_data = json.load(f)
                    if gt_data.get("hand_or_no_hand") == "hand" and "hands" in gt_data:
                        primary_hand = gt_data["hands"][0]
                        gt_landmarks = np.zeros((21, 3))
                        for kp in primary_hand.get("keypoints", []):
                            kp_id = kp["id"]
                            if kp_id < 21:
                                gt_landmarks[kp_id] = [kp["x_norm"], kp["y_norm"], kp.get("z", 0.0)]
                        
                        # Ground Truth Keypoints in GREEN
                        draw_keypoints(visual_canvas, gt_landmarks, color=(0, 255, 0), radius=4)
                        
                        if presence_score > 0.85:
                            print(f"\n--- Frame {idx} Depth Comparison ---")
                            print(f"{'Keypoint':<12} | {'Ground Truth Z':<14} | {'Predicted Z':<14}")
                            print(f"{'Wrist (0)':<12} | {gt_landmarks[0][2]:<14.4f} | {pred_landmarks[0][2]:<14.4f}")
                            print(f"{'Thumb (4)':<12} | {gt_landmarks[4][2]:<14.4f} | {pred_landmarks[4][2]:<14.4f}")
                            print(f"{'Index (8)':<12} | {gt_landmarks[8][2]:<14.4f} | {pred_landmarks[8][2]:<14.4f}")
                            
                            # Draw on video canvas
                            y_start = HEIGHT - 80
                            cv2.putText(visual_canvas, "Depth (Z) Comparison", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(visual_canvas, f"Wrist: GT {gt_landmarks[0][2]:.2f} | Pred {pred_landmarks[0][2]:.2f}", (20, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(visual_canvas, f"Thumb: GT {gt_landmarks[4][2]:.2f} | Pred {pred_landmarks[4][2]:.2f}", (20, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(visual_canvas, f"Index: GT {gt_landmarks[8][2]:.2f} | Pred {pred_landmarks[8][2]:.2f}", (20, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            
                except Exception as e:
                    print(e)

        # 5. Draw Model Predictions (threshold is 0.5)
        # Predicted Keypoints in CYAN
        if presence_score > 0.85:
            draw_keypoints(visual_canvas, pred_landmarks, color=(255, 242, 0), radius=4)
            handedness_str = "Right" if handedness_score > 0.5 else "Left"
            hand_text = f"Hand: {handedness_str} ({presence_score*100:.1f}%)"
            text_color = (0, 255, 255)
        else:
            hand_text = f"No Hand Detected ({presence_score*100:.1f}%)"
            text_color = (150, 150, 150)

        # 6. Render text overlays
        cv2.putText(visual_canvas, f"Frame: {idx}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(visual_canvas, hand_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        if JSON_INPUT:
            cv2.putText(visual_canvas, "GT: Green", (WIDTH - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(visual_canvas, "Pred: Cyan", (WIDTH - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 1, cv2.LINE_AA)

        video_writer.write(visual_canvas)

    video_writer.release()
    print(f"Visual representation saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()