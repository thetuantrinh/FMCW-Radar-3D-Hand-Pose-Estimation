# val.py
import torch
import numpy as np
import glob
import os
from tqdm import tqdm
import torchvision.transforms as transforms

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import RadarHPE3DNet
from core import helper
from core import transformations
from core import dataloader
from core import logger  
from core.parser import parser

def compute_mpjpe(pred, gt, mask):
    """
    Computes Mean Per Joint Position Error (MPJPE).
    pred: (N, 21, 3)
    gt: (N, 21, 3)
    mask: (N,) - whether hand is present
    """
    if mask.sum() == 0:
        return 0.0
    
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    # Euclidean distance per joint: shape (M, 21)
    errors = torch.norm(pred_masked - gt_masked, dim=2)
    return errors.mean().item()

def compute_pck(pred, gt, mask, thresholds=[0.05, 0.1]):
    """
    Computes Percentage of Correct Keypoints (PCK).
    pred: (N, 21, 3)
    gt: (N, 21, 3)
    mask: (N,) - whether hand is present
    thresholds: List of threshold values
    """
    if mask.sum() == 0:
        return {th: 0.0 for th in thresholds}
    
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    
    errors = torch.norm(pred_masked - gt_masked, dim=2)  # (M, 21)
    
    pck_results = {}
    for th in thresholds:
        correct = (errors < th).float()
        pck_results[th] = correct.mean().item() * 100.0  # Percentage
    
    return pck_results

def main():
    args = parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Evaluating on {device}")
    
    composed_transforms = transforms.Compose([
        transformations.ToTensor(),
    ])

    radar_dir = args.radar_dir if os.path.isabs(args.radar_dir) else os.path.join(args.data_dir, args.radar_dir)
    json_dir = args.json_dir if os.path.isabs(args.json_dir) else os.path.join(args.data_dir, args.json_dir)

    dataset = dataloader.RadarHandTrackingDataset(
        radar_dir=radar_dir,
        json_dir=json_dir,
        transforms=composed_transforms,
    )

    if len(dataset) == 0:
        logger.error("No data found for evaluation.")
        return

    # We reuse the same train/val split but only evaluate on validation
    _, val_loader = helper.split_dataset(
        dataset,
        batch_size=args.batch_size,
        train_ratio=0.8,
        shuffle=False,  # Don't shuffle for eval to have deterministic order
        seed=42,
        workers=2,
        pin_memory=True,
    )
    
    model = RadarHPE3DNet().to(device)
    
    # Load model weights
    checkpoint_path = args.checkpoint_path
    if not os.path.isfile(checkpoint_path):
        if not os.path.exists(checkpoint_path) and os.path.exists("results"):
            # Auto-detect latest run in results/
            runs = sorted([d for d in os.listdir("results") if d.startswith("run_")])
            if runs:
                latest_run = runs[-1]
                checkpoint_path = os.path.join("results", latest_run, "checkpoints", "best_model_latest.pt")
                if not os.path.isfile(checkpoint_path):
                    # fallback to just best_model or final_model if latest doesn't exist
                    checkpoints = sorted(glob.glob(os.path.join("results", latest_run, "checkpoints", "*.pt")))
                    if checkpoints:
                        checkpoint_path = checkpoints[-1]
        elif os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(args.checkpoint_path, "best_model_latest.pt")
            
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.error(f"No checkpoint found at '{checkpoint_path}'")
        return

    model.eval()
    
    total_mpjpe = 0.0
    total_hands = 0
    pck_thresholds = [0.05, 0.1, 0.15, 0.2]
    total_pcks = {th: 0.0 for th in pck_thresholds}
    
    pbar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for data_batch in pbar:
            if isinstance(data_batch, dict):
                data = data_batch["radar_tensor"].to(device)
                landmarks_gt = data_batch["keypoints_3d"].to(device)
                hand_presence_gt = data_batch["hand_status"].to(device)
            else:
                data = data_batch[0].to(device)
                landmarks_gt = data_batch[1].to(device)
                hand_presence_gt = data_batch[2].to(device)

            landmarks_pred, _, _ = model(data)
            
            mask = (hand_presence_gt == 1.0).squeeze(1)
            num_hands = mask.sum().item()
            
            if num_hands > 0:
                batch_mpjpe = compute_mpjpe(landmarks_pred, landmarks_gt, mask)
                batch_pck = compute_pck(landmarks_pred, landmarks_gt, mask, thresholds=pck_thresholds)
                
                total_mpjpe += batch_mpjpe * num_hands
                for th in pck_thresholds:
                    total_pcks[th] += batch_pck[th] * num_hands
                total_hands += num_hands
                
    if total_hands > 0:
        final_mpjpe = total_mpjpe / total_hands
        final_pck = {th: total_pcks[th] / total_hands for th in pck_thresholds}
        
        logger.info(f"Evaluation complete over {total_hands} valid hands.")
        logger.info(f"MPJPE: {final_mpjpe:.4f}")
        for th in pck_thresholds:
            logger.info(f"PCK@{th}: {final_pck[th]:.2f}%")
    else:
        logger.warning("No hands found in the validation set to evaluate.")

if __name__ == "__main__":
    main()
