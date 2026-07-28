# evaluate_run6.py
import os
import glob
import re
import json
import numpy as np
import torch
import cv2
import h5py
import mediapipe as mp
from datetime import datetime

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import DevModel7
from core import transformations

# ==============================================================================
# --- Configuration Parameters ---
# ==============================================================================
HDF5_PATH        = "/work/tuan.tt19010226/Radar-HPE/new_ver/dataset/run6.hdf5"
MP4_PATH         = "/work/tuan.tt19010226/Radar-HPE/new_ver/dataset/run6.mp4"
MODEL_CHECKPOINT = "results/run_20260611_095339/checkpoints/best_model_latest.pt"
OUTPUT_VIDEO     = "run6_comparison.mp4"
BACKGROUND_FILE = "dataset/norm_ds/avg_background.npy"
MEAN_FILE       = "dataset/norm_ds/mean.npy"
STD_FILE        = "dataset/norm_ds/std.npy"
TIME_THRESHOLD_MS = 5.0
EXPECTED_SHAPE  = (8, 64, 64)
BLACK_BACKGROUND = True  # Set to True to draw on a clean black canvas, or False to draw over MP4 frames

# Calibration toggles for physical radar-to-camera 180-degree mounting rotations
FLIP_PRED_X      = True  # Flip predicted X coordinates (x = 1.0 - x)
FLIP_PRED_Y      = True  # Flip predicted Y coordinates (y = 1.0 - y)
# ==============================================================================


def find_log_files(dataset_dir):
    """Searches for matched logging files under standard location patterns."""
    patterns = [
        # 1. Subdirectory logs/
        (os.path.join(dataset_dir, "logs", "radar_run6.txt"), os.path.join(dataset_dir, "logs", "camera_run6.txt")),
        # 2. Root folder
        (os.path.join(dataset_dir, "radar_run6.txt"), os.path.join(dataset_dir, "camera_run6.txt")),
        # 3. logs/ directory up one level
        (os.path.join(dataset_dir, "..", "logs", "radar_run6.txt"), os.path.join(dataset_dir, "..", "logs", "camera_run6.txt")),
    ]
    for r_log, c_log in patterns:
        if os.path.exists(r_log) and os.path.exists(c_log):
            return r_log, c_log
    return None, None


def parse_radar_log(file_path):
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) - radar collecting frame - (\d+)"
    )
    radar_frames = []
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ts_str, frame_idx_str = match.groups()
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                frame_idx = int(frame_idx_str)
                radar_frames.append((ts, frame_idx))
    return radar_frames


def parse_camera_log(file_path):
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) - camera collecting frame - (\d+)"
    )
    camera_frames = []
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ts_str, frame_idx_str = match.groups()
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                frame_idx = int(frame_idx_str)
                camera_frames.append((ts, frame_idx))
    return camera_frames


def process(adcData, Nc=64, Ns=64):
    """Processes raw HDF5 ADC radar frames into 8 receiver channels (Nc, Ns)."""
    adcData = np.reshape(adcData, (8, Nc * Ns), order="F")
    adcData = np.reshape(adcData, (8, Nc, Ns))
    interleaved_indices = [0, 4, 1, 5, 2, 6, 3, 7]
    return adcData[interleaved_indices]


def draw_keypoints(img, landmarks, color, radius=4):
    """Draws keypoints without skeleton lines to meet connection constraints."""
    h, w, _ = img.shape
    for kp in landmarks:
        x = int(kp[0] * w)
        y = int(kp[1] * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # --- Load Preprocessing Files ---
    if not os.path.exists(BACKGROUND_FILE) or not os.path.exists(MEAN_FILE) or not os.path.exists(STD_FILE):
        raise FileNotFoundError(f"Missing background, mean, or std files under configuration paths.")
    
    background = np.load(BACKGROUND_FILE).astype(np.float32)
    mean = np.load(MEAN_FILE).astype(np.float32)
    std = np.load(STD_FILE).astype(np.float32)
    print("Preprocessing arrays successfully loaded.")

    # --- Initialize Preprocessing Transformations ---
    bg_removal = transformations.BackgroundRemoval(background)
    to_tensor = transformations.ToTensor()
    normalize = transformations.Normalize(mean, std)
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # --- Load Network Model ---
    model = DevModel7().to(device)
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_CHECKPOINT}")
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model checkpoint loaded successfully.")

    # --- Load Radar File & Video Stream ---
    if not os.path.exists(HDF5_PATH):
        raise FileNotFoundError(f"HDF5 file not found at: {HDF5_PATH}")
    h5_file = h5py.File(HDF5_PATH, "r")
    available_keys = list(h5_file.keys())
    print(f"HDF5 file opened successfully. Total frames: {len(available_keys)}")

    cap = cv2.VideoCapture(MP4_PATH)
    if not cap.isOpened():
        h5_file.close()
        raise FileNotFoundError(f"Could not open camera video file at: {MP4_PATH}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- Setup Output Video Writer ---
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) if os.path.dirname(OUTPUT_VIDEO) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # --- Discover and Parse Log Files ---
    dataset_dir = os.path.dirname(HDF5_PATH)
    radar_log_path, camera_log_path = find_log_files(dataset_dir)
    
    logs_parsed = False
    radar_logs = []
    camera_logs = []
    
    if radar_log_path and camera_log_path:
        try:
            print(f"Parsing logging metrics:\n  Radar Log: {radar_log_path}\n  Camera Log: {camera_log_path}")
            radar_logs = parse_radar_log(radar_log_path)
            camera_logs = parse_camera_log(camera_log_path)
            if len(radar_logs) > 0 and len(camera_logs) > 0:
                logs_parsed = True
                print(f"Successfully loaded matched logging metrics.")
        except Exception as e:
            print(f"Logging parsing encountered an exception: {e}")

    # --- Matched Core Processing Loop ---
    matched_count = 0

    if logs_parsed:
        # 1. Processing via exact sub-5ms timestamp matches
        print("Starting timestamp-matched extraction loop...")
        threshold_seconds = TIME_THRESHOLD_MS / 1000.0

        for r_ts, r_idx in radar_logs:
            best_match = None
            min_diff = float("inf")

            for c_ts, c_idx in camera_logs:
                diff = abs((r_ts - c_ts).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_match = (c_idx, diff)

            if best_match is not None and min_diff < threshold_seconds:
                c_idx, diff_val = best_match
                diff_ms = diff_val * 1000.0
                
                radar_key = str(r_idx - 1)
                if radar_key not in h5_file:
                    continue

                # Read and shape radar array
                radar_raw = h5_file[radar_key][()]
                radar_cube = process(radar_raw, Nc=64, Ns=64)

                # Extract camera frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, c_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Prepare Canvas
                canvas = np.zeros((height, width, 3), dtype=np.uint8) if BLACK_BACKGROUND else frame.copy()

                # A. Extract Ground Truth using MediaPipe Hands
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = hands_detector.process(img_rgb)
                
                gt_landmarks = []
                if mp_results.multi_hand_landmarks:
                    hand_landmarks = mp_results.multi_hand_landmarks[0]
                    for lm in hand_landmarks.landmark:
                        gt_landmarks.append([lm.x, lm.y])
                    draw_keypoints(canvas, gt_landmarks, color=(0, 255, 0), radius=4)

                # B. Extract Model predictions using the transformation pipeline
                dummy_sample = [radar_cube, np.zeros((21, 2)), np.array([0.0]), np.array([-1.0])]
                sample_subtracted = bg_removal(dummy_sample)
                sample_tensor = to_tensor(sample_subtracted)
                sample_normalized = normalize(sample_tensor)
                
                # Cast the normalized input to float to avoid dtype errors and transfer to device
                input_tensor = sample_normalized[0].float().unsqueeze(0).to(device)

                with torch.no_grad():
                    landmarks_pred, presence_pred, handedness_pred = model(input_tensor)

                pred_landmarks = landmarks_pred[0].cpu().numpy()
                presence_score = presence_pred[0].item()
                handedness_score = handedness_pred[0].item()

                # Apply spatial mirroring flips
                if FLIP_PRED_X:
                    pred_landmarks[:, 0] = 1.0 - pred_landmarks[:, 0]
                if FLIP_PRED_Y:
                    pred_landmarks[:, 1] = 1.0 - pred_landmarks[:, 1]

                if presence_score > 0.5:
                    draw_keypoints(canvas, pred_landmarks, color=(255, 242, 0), radius=4)
                    handedness_str = "Right" if handedness_score > 0.5 else "Left"
                    hand_text = f"Pred: {handedness_str} ({presence_score*100:.1f}%)"
                    text_color = (255, 242, 0)
                else:
                    hand_text = f"Pred: No Hand ({presence_score*100:.1f}%)"
                    text_color = (150, 150, 150)

                # Render details
                cv2.putText(canvas, f"Frame: {c_idx} (Diff: {diff_ms:.2f}ms)", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, hand_text, (20, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
                cv2.putText(canvas, "MP (GT): Green", (width - 160, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(canvas, "Pred: Cyan", (width - 160, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 1, cv2.LINE_AA)

                video_writer.write(canvas)
                matched_count += 1
    else:
        # 2. Fallback execution: sequential index mapping
        print("Logs not loaded. Proceeding with sequential mapping...")
        total_radar_keys = len(available_keys)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(total_radar_keys, total_video_frames)

        for idx in range(max_frames):
            radar_key = str(idx)
            if radar_key not in h5_file:
                continue

            radar_raw = h5_file[radar_key][()]
            radar_cube = process(radar_raw, Nc=64, Ns=64)

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            canvas = np.zeros((height, width, 3), dtype=np.uint8) if BLACK_BACKGROUND else frame.copy()

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = hands_detector.process(img_rgb)
            
            gt_landmarks = []
            if mp_results.multi_hand_landmarks:
                hand_landmarks = mp_results.multi_hand_landmarks[0]
                for lm in hand_landmarks.landmark:
                    gt_landmarks.append([lm.x, lm.y])
                draw_keypoints(canvas, gt_landmarks, color=(0, 255, 0), radius=4)

            # Preprocess frame using the transformation pipeline
            dummy_sample = [radar_cube, np.zeros((21, 2)), np.array([0.0]), np.array([-1.0])]
            sample_subtracted = bg_removal(dummy_sample)
            sample_tensor = to_tensor(sample_subtracted)
            sample_normalized = normalize(sample_tensor)
            
            # Cast the normalized input to float to avoid dtype errors and transfer to device
            input_tensor = sample_normalized[0].float().unsqueeze(0).to(device)

            with torch.no_grad():
                landmarks_pred, presence_pred, handedness_pred = model(input_tensor)

            pred_landmarks = landmarks_pred[0].cpu().numpy()
            presence_score = presence_pred[0].item()
            handedness_score = handedness_pred[0].item()

            # Apply calibration rotation flips
            if FLIP_PRED_X:
                pred_landmarks[:, 0] = 1.0 - pred_landmarks[:, 0]
            if FLIP_PRED_Y:
                pred_landmarks[:, 1] = 1.0 - pred_landmarks[:, 1]

            if presence_score > 0.5:
                draw_keypoints(canvas, pred_landmarks, color=(255, 242, 0), radius=4)
                handedness_str = "Right" if handedness_score > 0.5 else "Left"
                hand_text = f"Pred: {handedness_str} ({presence_score*100:.1f}%)"
                text_color = (255, 242, 0)
            else:
                hand_text = f"Pred: No Hand ({presence_score*100:.1f}%)"
                text_color = (150, 150, 150)

            cv2.putText(canvas, f"Seq Frame: {idx}", (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, hand_text, (20, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            cv2.putText(canvas, "MP (GT): Green", (width - 160, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Pred: Cyan", (width - 160, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 1, cv2.LINE_AA)

            video_writer.write(canvas)
            matched_count += 1

    # --- Cleanup ---
    h5_file.close()
    cap.release()
    video_writer.release()
    hands_detector.close()
    print(f"\nProcessing complete. Successfully generated matched video file: {OUTPUT_VIDEO} ({matched_count} frames)")


if __name__ == "__main__":
    main()