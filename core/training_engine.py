# training_engine.py
from core import logger
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        saved_path=None,
        loss_plot_path="loss_plots",
        results_dir="results",
        device=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.criterion = criterion
        self.model.to(self.device)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = results_dir
        self.run_dir = os.path.join(results_dir, f"run_{self.timestamp}")
        
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.loss_plot_dir = os.path.join(self.run_dir, "loss_plots")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.loss_plot_dir, exist_ok=True)
        
        if saved_path is None or saved_path == "path":
            saved_path = os.path.join(self.checkpoint_dir, "model.pt")
            logger.info(f"No valid saved_path provided, using default: {saved_path}")
        else:
            filename = os.path.basename(saved_path)
            saved_path = os.path.join(self.checkpoint_dir, filename)
        
        self.saved_path = saved_path
        self.loss_plot_path = self.loss_plot_dir
        
        self.regression_train_loss = []
        self.regression_val_loss = []
        self.min_regression_loss = torch.inf
        self.hand_presence_train_loss = []
        self.hand_presence_val_loss = []
        self.min_presence_loss = torch.inf
        self.handedness_train_loss = []
        self.handedness_val_loss = []
        self.min_handedness_loss = torch.inf
        self.history = []
        self.print_freq = 10
        self.best_checkpoint_path = ""
        torch.autograd.set_detect_anomaly(True)
        
        logger.info(f"Training run directory: {self.run_dir}")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        logger.info(f"Loss plots will be saved to: {self.loss_plot_dir}")

    def train(self, start_epoch, epochs):
        logger.info(
            f"Training model on {self.device} for {epochs} epochs starting at epoch {start_epoch}"
        )
        self.epochs = epochs
        for epoch in range(start_epoch, epochs):
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            
            self.save_loss_plots(is_final=False)
            
            self.history = {
                "train_loss": {
                    "regression_loss": self.regression_train_loss,
                    "handedness_loss": self.handedness_train_loss,
                    "hand_presence_loss": self.hand_presence_train_loss,
                },
                "val_loss": {
                    "regresion_loss": self.regression_val_loss,
                    "handedness_loss": self.handedness_val_loss,
                    "hand_presence_loss": self.hand_presence_val_loss,
                },
                "min_loss": self.min_regression_loss,
            }
        
        self.save_loss_plots(is_final=True)
        self.save_final_checkpoint()
        self.save_training_summary()
        logger.info(f"Training completed! All results saved to: {self.run_dir}")

    def train_epoch(self, epoch):
        self.model.train()
        regression_losses = AverageMeter()
        handedness_losses = AverageMeter()
        hand_presence_losses = AverageMeter()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Train Epoch {epoch+1}/{self.epochs}",
            leave=True
        )
        
        for i, data_batch in pbar:
            if isinstance(data_batch, dict):
                data = data_batch["radar_tensor"].to(self.device)
                landmarks_gt = data_batch["keypoints_3d"].to(self.device)
                hand_presence_gt = data_batch["hand_status"].to(self.device)
                handedness_gt = data_batch["handedness"].to(self.device)
            else:
                data = data_batch[0].to(self.device)
                landmarks_gt = data_batch[1].to(self.device)
                hand_presence_gt = data_batch[2].to(self.device)
                handedness_gt = data_batch[3].to(self.device)

            self.optimizer.zero_grad()
            landmarks_pred, hand_presence_pred, handedness_pred = self.model(data)
            
            # Binary Cross Entropy for overall hand presence
            hand_presence_loss = self.criterion[2](hand_presence_pred, hand_presence_gt)
            
            # Mask to run backpropagation only on samples actually containing a hand.
            # This bypasses empty-frames whose sentinel coordinates and values corrupt learning.
            mask = (hand_presence_gt == 1.0).squeeze(1)
            num_hands = mask.sum().item()
            
            if num_hands > 0:
                regression_loss = self.criterion[0](landmarks_pred[mask], landmarks_gt[mask])
                handedness_loss = self.criterion[1](handedness_pred[mask], handedness_gt[mask])
            else:
                regression_loss = torch.tensor(0.0, device=self.device)
                handedness_loss = torch.tensor(0.0, device=self.device)

            # Consolidated loss for single backward pass
            total_loss = regression_loss + handedness_loss + hand_presence_loss
            total_loss.backward()
            
            self.optimizer.step()

            # Record metrics
            regression_losses.update(regression_loss.item(), max(1, num_hands))
            handedness_losses.update(handedness_loss.item(), max(1, num_hands))
            hand_presence_losses.update(hand_presence_loss.item(), data.size(0))

            pbar.set_postfix({
                'reg_loss': f'{regression_losses.avg:.3f}',
                'hand_presence': f'{hand_presence_losses.avg:.3f}',
                'handedness': f'{handedness_losses.avg:.3f}'
            })

        self.regression_train_loss.append(regression_losses.avg)
        self.hand_presence_train_loss.append(hand_presence_losses.avg)
        self.handedness_train_loss.append(handedness_losses.avg)

    def val_epoch(self, epoch):
        regression_losses = AverageMeter()
        handedness_losses = AverageMeter()
        hand_presence_losses = AverageMeter()
        self.model.eval()
        
        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc=f"Val Epoch {epoch+1}/{self.epochs}",
            leave=True
        )
        
        with torch.no_grad():
            for i, data_batch in pbar:
                if isinstance(data_batch, dict):
                    data = data_batch["radar_tensor"].to(self.device)
                    landmarks_gt = data_batch["keypoints_3d"].to(self.device)
                    hand_presence_gt = data_batch["hand_status"].to(self.device)
                    handedness_gt = data_batch["handedness"].to(self.device)
                else:
                    data = data_batch[0].to(self.device)
                    landmarks_gt = data_batch[1].to(self.device)
                    hand_presence_gt = data_batch[2].to(self.device)
                    handedness_gt = data_batch[3].to(self.device)

                landmarks_pred, hand_presence_pred, handedness_pred = self.model(data)
                
                hand_presence_loss = self.criterion[2](hand_presence_pred, hand_presence_gt)
                
                mask = (hand_presence_gt == 1.0).squeeze(1)
                num_hands = mask.sum().item()
                
                if num_hands > 0:
                    regression_loss = self.criterion[0](landmarks_pred[mask], landmarks_gt[mask])
                    handedness_loss = self.criterion[1](handedness_pred[mask], handedness_gt[mask])
                else:
                    regression_loss = torch.tensor(0.0, device=self.device)
                    handedness_loss = torch.tensor(0.0, device=self.device)

                regression_losses.update(regression_loss.item(), max(1, num_hands))
                handedness_losses.update(handedness_loss.item(), max(1, num_hands))
                hand_presence_losses.update(hand_presence_loss.item(), data.size(0))

                pbar.set_postfix({
                    'reg_val': f'{regression_losses.avg:.3f}',
                    'hand_presence_val': f'{hand_presence_losses.avg:.3f}',
                    'handedness_val': f'{handedness_losses.avg:.3f}'
                })

        logger.info(
            f"Epoch {epoch+1}/{self.epochs} - "
            f"Val Regression Loss: {regression_losses.avg:.4f}, "
            f"Val Hand Presence Loss: {hand_presence_losses.avg:.4f}, "
            f"Val Handedness Loss: {handedness_losses.avg:.4f}"
        )

        self.regression_val_loss.append(regression_losses.avg)
        self.hand_presence_val_loss.append(hand_presence_losses.avg)
        self.handedness_val_loss.append(handedness_losses.avg)

        if self.scheduler:
            self.scheduler.step(self.regression_val_loss[-1])

        self.save_checkpoint(epoch, regression_losses.avg)

    def save_loss_plots(self, is_final=False):
        epochs = range(1, len(self.regression_train_loss) + 1)
        if len(epochs) == 0:
            return
        
        # Combined 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(epochs, self.regression_train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.regression_val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Regression Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.hand_presence_train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.hand_presence_val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Hand Presence Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.handedness_train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[1, 0].plot(epochs, self.handedness_val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Handedness Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.regression_val_loss, 'b-', label='Regression', linewidth=2)
        axes[1, 1].plot(epochs, self.hand_presence_val_loss, 'g-', label='Hand Presence', linewidth=2)
        axes[1, 1].plot(epochs, self.handedness_val_loss, 'r-', label='Handedness', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('All Validation Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        latest_plot_filename = os.path.join(self.loss_plot_path, 'latest_losses.png')
        plt.savefig(latest_plot_filename, dpi=100, bbox_inches='tight')
        
        if is_final:
            final_plot_filename = os.path.join(self.loss_plot_path, 'final_losses.png')
            plt.savefig(final_plot_filename, dpi=100, bbox_inches='tight')
            logger.info(f"Final training curves saved to: {final_plot_filename}")
            
        plt.close(fig)

    def save_checkpoint(self, epoch, loss):
        checkpoint_save_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}_{self.timestamp}.pt"
        )
        
        best_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"best_model_{self.timestamp}.pt"
        )
        self.best_checkpoint_path = best_checkpoint_path
        
        checkpoint_data = {
            "epoch": epoch,
            "timestamp": self.timestamp,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "criterion": str(self.criterion),
            "history": self.history,
            "min_loss": loss,
            "regression_train_loss": self.regression_train_loss,
            "regression_val_loss": self.regression_val_loss,
            "hand_presence_train_loss": self.hand_presence_train_loss,
            "hand_presence_val_loss": self.hand_presence_val_loss,
            "handedness_train_loss": self.handedness_train_loss,
            "handedness_val_loss": self.handedness_val_loss,
        }
        
        torch.save(checkpoint_data, checkpoint_save_path)
        logger.info(f"Checkpoint saved: {checkpoint_save_path}")

        if loss < self.min_regression_loss:
            torch.save(checkpoint_data, best_checkpoint_path)
            logger.info(
                f"Successfully saved best model with smallest loss of {loss:.4f} at {best_checkpoint_path}"
            )
            self.min_regression_loss = loss
            
            latest_best_path = os.path.join(self.checkpoint_dir, "best_model_latest.pt")
            torch.save(checkpoint_data, latest_best_path)

    def save_final_checkpoint(self):
        final_checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"final_model_{self.timestamp}.pt"
        )
        
        checkpoint_data = {
            "epoch": self.epochs - 1,
            "timestamp": self.timestamp,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "criterion": str(self.criterion),
            "history": self.history,
            "min_loss": self.min_regression_loss,
            "regression_train_loss": self.regression_train_loss,
            "regression_val_loss": self.regression_val_loss,
            "hand_presence_train_loss": self.hand_presence_train_loss,
            "hand_presence_val_loss": self.hand_presence_val_loss,
            "handedness_train_loss": self.handedness_train_loss,
            "handedness_val_loss": self.handedness_val_loss,
        }
        
        torch.save(checkpoint_data, final_checkpoint_path)
        logger.info(f"Final model saved: {final_checkpoint_path}")
        
        latest_model_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        torch.save(checkpoint_data, latest_model_path)

    def save_training_summary(self):
        summary_path = os.path.join(self.run_dir, "training_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Epochs: {self.epochs}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Final Losses:\n")
            f.write(f"  Regression Loss: {self.regression_val_loss[-1]:.6f}\n")
            f.write(f"  Hand Presence Loss: {self.hand_presence_val_loss[-1]:.6f}\n")
            f.write(f"  Handedness Loss: {self.handedness_val_loss[-1]:.6f}\n\n")
            
            f.write("Best Losses:\n")
            f.write(f"  Best Regression Loss: {self.min_regression_loss:.6f}\n\n")
            
            f.write("Checkpoint Files:\n")
            f.write(f"  Best Model: {self.best_checkpoint_path}\n")
            f.write(f"  Final Model: final_model_{self.timestamp}.pt\n\n")
            
            f.write("Loss Plot Files:\n")
            f.write(f"  Location: {self.loss_plot_path}\n")
            f.write("  - final_losses.png (Full history)\n")
            f.write("  - latest_losses.png\n")
        
        logger.info(f"Training summary saved to: {summary_path}")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count