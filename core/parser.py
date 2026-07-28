# parser.py
import argparse

def parser():
    parser = argparse.ArgumentParser(
        description="Training handpose estimation with teacher-student architecture"
    )
    parser.add_argument(
        "--checkpoint-path",
        default="history/model_checkpoints/", type=str)
        
    parser.add_argument(
        "--radar-dir",
        default="matched_radar",
        type=str,
        help="Subdirectory or folder path containing the .npy radar files",
    )
    parser.add_argument(
        "--json-dir",
        default="matched_camera_hand_json_3d",
        type=str,
        help="Subdirectory or folder path containing the .json label files",
    )
    parser.add_argument(
        "--data-dir",
        default="dataset/raw_ds/",
        type=str,
        help="where the image and radar folders are located",
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--saved-model-path", default="path", type=str)
    parser.add_argument(
        "--criterion",
        default="huber",
        type=str,
        help="available: hubber, mse, mae",
    )
    parser.add_argument("--expansion", default=2, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--summary-file", default="history/model_summary.txt", type=str
    )

    return parser.parse_args()