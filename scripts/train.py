# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import os
from torchsummary import summary

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import RadarHPE3DNet
from core import helper
from core import transformations
from core import dataloader
from core.training_engine import Trainer
from core import logger  
from core.parser import parser

torch.manual_seed(42)
np.random.seed(42)


def log_model_summary(model, input_size, device, output_file=None):
    """Prints and saves a model summary."""
    
    def write_summary(f):
        logger.log_section_header("MODEL ARCHITECTURE")
        f.write(f"{model}\n\n")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.log_section_header("MODEL PARAMETERS")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n\n")

        logger.log_section_header("LAYER-WISE PARAMETER COUNT")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: {param.numel():,}\n")
        f.write("\n")

        if input_size:
            try:
                logger.log_section_header("TORCHSUMMARY OUTPUT")
                summary(model.to(device), input_size, print_fn=lambda x: f.write(f"{x}\n"))
            except Exception as e:
                warning_msg = f"Could not generate torchsummary: {e}"
                logger.warning(warning_msg)
                f.write(warning_msg)

    # Print to console
    class ConsoleWriter:
        def write(self, text):
            print(text, end='')
    write_summary(ConsoleWriter())

    # Save to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            write_summary(f)
        logger.info(f"\nModel summary saved to: {output_file}")


def log_system_info():
    """Prints CUDA device and PyTorch version parameters."""
    logger.log_section_header("SYSTEM INFORMATION")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"\nPyTorch Version: {torch.__version__}")
    else:
        print("CUDA is NOT available. Using CPU.")


def setup_environment():
    """Configures environment constraints and initializes PyTorch device."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        device = torch.device("cuda:0")
        logger.info(f"CUDA optimizations enabled. Training on {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info(f"Training on {device}")
    return device


def setup_data_loaders(args):
    composed_transforms = transforms.Compose([
        transformations.ToTensor(),
    ])

    # Resolve radar and JSON source folders
    radar_dir = args.radar_dir if os.path.isabs(args.radar_dir) else os.path.join(args.data_dir, args.radar_dir)
    json_dir = args.json_dir if os.path.isabs(args.json_dir) else os.path.join(args.data_dir, args.json_dir)

    # Load matched pair dataset
    dataset = dataloader.RadarHandTrackingDataset(
        radar_dir=radar_dir,
        json_dir=json_dir,
        transforms=composed_transforms,
    )

    if len(dataset) == 0:
        logger.error(f"No matched pairs found. Ensure matching filenames exist in '{radar_dir}' and '{json_dir}'.")

    # Generate train/val split using 80-20 partition ratio
    train_loader, val_loader = helper.split_dataset(
        dataset,
        batch_size=args.batch_size,
        train_ratio=0.8,
        shuffle=True,
        seed=42,
        workers=2,
        pin_memory=True,
    )
    
    logger.info(f"Total dataset size: {len(dataset)}")
    logger.info(f"Training set: {len(train_loader.sampler)} indices ({len(train_loader)} batches)")
    logger.info(f"Validation set: {len(val_loader.sampler)} indices ({len(val_loader)} batches)")

    return train_loader, val_loader


def main():
    args = parser()
    device = setup_environment()
    
    # --- Data and Model Setup ---
    train_loader, val_loader = setup_data_loaders(args)
    model = RadarHPE3DNet().to(device)

    # --- Logging and Summaries ---
    logger.log_section_header("CONFIGURATION")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    log_system_info()
    log_model_summary(
        model, 
        input_size=(8, 64, 64),  # Shape: (C, H, W)
        device=device,
        output_file=args.summary_file
    )
    
    # --- Training Setup ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.99),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=5, factor=0.1, min_lr=1e-7
    )
    criterions = [nn.L1Loss(), nn.BCELoss(), nn.BCELoss()]

    engine = Trainer(
        model,
        train_loader,
        val_loader,
        criterions,
        optimizer,
        scheduler,
        saved_path=args.saved_model_path,
        device=device,
    )
    
    # --- Start Training ---
    logger.log_section_header("TRAINING START")
    engine.train(start_epoch=args.start_epoch, epochs=args.epochs)


if __name__ == "__main__":
    main()