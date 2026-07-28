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

from core.network import RadarHPE3DNet
from core import transformations

run = 9
# ==============================================================================
# --- Configuration Parameters ---
# ==============================================================================
HDF5_PATH         = f"/work/tuan.tt19010226/Radar-HPE/new_ver/dataset/raw_ds/DucVu/radar/run{run}.hdf5"
MP4_PATH          = f"/work/tuan.tt19010226/Radar-HPE/new_ver/dataset/raw_ds/DucVu/camera/run{run}.mp4"
MODEL_CHECKPOINT  = f"results/run_20260614_192327/checkpoints/best_model_latest.pt"
OUTPUT_VIDEO      = f"run{run}_comparison.mp4"
TIME_THRESHOLD_MS = 5.0
BLACK_BACKGROUND  = False  # Set to True to draw on a clean black canvas, or False to draw over MP4 frames

# --- Simulated Lighting Parameters ---
WEAK_LIGHT_ALPHA   = 0.15  # Scale factor for low light (< 1.0)
WEAK_LIGHT_BETA    = 5     # Offset for low light
STRONG_LIGHT_ALPHA = 1.8   # Scale factor for bright light (> 1.0)
STRONG_LIGHT_BETA  = 80    # Offset for bright light
# ==============================================================================


def find_log_files(dataset_dir):
    """Searches for matched logging files under standard location patterns."""
    patterns = [
        # 1. Subdirectory logs/
        (os.path.join(dataset_dir, "logs", f"radar_run{run}.txt"), os.path.join(dataset_dir, "logs", f"camera_run{run}.txt")),
        # 2. Root folder
        (os.path.join(dataset_dir, f"radar_run{run}.txt"), os.path.join(dataset_dir, f"camera_run{run}.txt")),
        # 3. logs/ directory up one level
        (os.path.join(dataset_dir, "..", "logs", f"radar_run{run}.txt"), os.path.join(dataset_dir, "..", "logs", f"camera_run{run}.txt")),
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


def process_and_draw(frame_to_use, radar_pred, radar_presence, radar_handedness, hands_detector, title, black_bg=False, draw_mp=True):
    """Processes a single visual condition pane and overlays predictions."""
    h, w, _ = frame_to_use.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8) if black_bg else frame_to_use.copy()

    mp_detected = False
    if draw_mp:
        # Run MediaPipe hands on the provided frame variant
        img_rgb = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2RGB)
        mp_results = hands_detector.process(img_rgb)
        
        gt_landmarks = []
        if mp_results.multi_hand_landmarks:
            mp_detected = True
            hand_landmarks = mp_results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                gt_landmarks.append([lm.x, lm.y])
            draw_keypoints(canvas, gt_landmarks, color=(0, 255, 0), radius=4)

    # Draw Radar Model Predictions
    if radar_presence > 0.5:
        draw_keypoints(canvas, radar_pred, color=(255, 242, 0), radius=4)
        handedness_str = "Right" if radar_handedness > 0.5 else "Left"
        hand_text = f"Radar: {handedness_str} ({radar_presence*100:.1f}%)"
        text_color = (255, 242, 0)
    else:
        hand_text = f"Radar: No Hand ({radar_presence*100:.1f}%)"
        text_color = (150, 150, 150)

    # Annotations
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, hand_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    
    if draw_mp:
        mp_text = "MP: Detected" if mp_detected else "MP: NOT Detected"
        mp_color = (0, 255, 0) if mp_detected else (0, 0, 255)
        cv2.putText(canvas, mp_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mp_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "MP (GT): Green", (w - 160, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Radar Pred: Cyan", (w - 160, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 242, 0), 1, cv2.LINE_AA)

    return canvas


def render_frame_grid(frame, radar_cube, model, device, hands_detector, to_tensor, width, height):
    """Generates predictions and builds a 2x2 visual comparison grid."""
    
    # 1. Compute Radar Model inference (independent of lighting simulation)
    dummy_sample = [radar_cube, np.zeros((21, 2)), np.array([0.0]), np.array([-1.0])]
    sample_tensor = to_tensor(dummy_sample)
    input_tensor = sample_tensor[0].unsqueeze(0).to(device)

    with torch.no_grad():
        landmarks_pred, presence_pred, handedness_pred = model(input_tensor)

    pred_landmarks = landmarks_pred[0].cpu().numpy()
    presence_score = presence_pred[0].item()
    handedness_score = handedness_pred[0].item()

    # 2. Simulate lighting conditions on the current camera frame
    frame_low = cv2.convertScaleAbs(frame, alpha=WEAK_LIGHT_ALPHA, beta=WEAK_LIGHT_BETA)
    frame_high = cv2.convertScaleAbs(frame, alpha=STRONG_LIGHT_ALPHA, beta=STRONG_LIGHT_BETA)

    # 3. Process and draw each grid pane
    # Top-Left: Normal Light
    panel_normal = process_and_draw(
        frame, pred_landmarks, presence_score, handedness_score, 
        hands_detector, "Normal Light", black_bg=BLACK_BACKGROUND, draw_mp=True
    )
    # Top-Right: Radar Only (Showing independent radar tracking)
    panel_radar = process_and_draw(
        frame, pred_landmarks, presence_score, handedness_score, 
        hands_detector, "Radar Only (No Light Dependency)", black_bg=True, draw_mp=False
    )
    # Bottom-Left: Simulated Weak Light
    panel_low = process_and_draw(
        frame_low, pred_landmarks, presence_score, handedness_score, 
        hands_detector, "Weak Light (Simulated)", black_bg=BLACK_BACKGROUND, draw_mp=True
    )
    # Bottom-Right: Simulated Strong Light
    panel_high = process_and_draw(
        frame_high, pred_landmarks, presence_score, handedness_score, 
        hands_detector, "Strong Light (Simulated)", black_bg=BLACK_BACKGROUND, draw_mp=True
    )

    # Combine into 2x2 structure
    top_row = np.hstack((panel_normal, panel_radar))
    bottom_row = np.hstack((panel_low, panel_high))
    grid = np.vstack((top_row, bottom_row))
    return grid


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # --- Initialize Assets & Loaders ---
    to_tensor = transformations.ToTensor()
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # --- Load Network Model ---
    model = RadarHPE3DNet().to(device)
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
    fps = 20

    # --- Setup Output Video Writer (2x2 Grid has 2*width and 2*height) ---
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) if os.path.dirname(OUTPUT_VIDEO) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (2 * width, 2 * height))

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

                # Render grid with current frame and radar data
                grid = render_frame_grid(
                    frame, radar_cube, model, device, hands_detector, 
                    to_tensor, width, height
                )

                video_writer.write(grid)
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

            # Render grid with current frame and radar data
            grid = render_frame_grid(
                frame, radar_cube, model, device, hands_detector, 
                to_tensor, width, height
            )

            # Annotate current index on the top-left portion of the overall grid frame
            cv2.putText(grid, f"Seq Frame: {idx}", (20, height + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            video_writer.write(grid)
            matched_count += 1

    # --- Cleanup ---
    h5_file.close()
    cap.release()
    video_writer.release()
    hands_detector.close()
    print(f"\nProcessing complete. Successfully generated matched video file: {OUTPUT_VIDEO} ({matched_count} frames)")


if __name__ == "__main__":
    main()