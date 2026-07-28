import os
import torch
import torch.nn as nn
import logging
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initializes the DepthEstimator using an offline Depth Anything Model.
        
        Args:
            model_path: The local path to the model directory.
            device: Target device (MUST start with 'cuda').
        """
        self.logger = logging.getLogger("DepthEstimator")
        
        # Enforce GPU Execution
        if not device.startswith("cuda"):
            self.logger.error("CPU execution is strictly forbidden. Device must be cuda.")
            raise ValueError("Device must be cuda. CPU fallback not allowed.")
            
        if not torch.cuda.is_available():
            self.logger.error("CUDA is not available on this system. Terminating immediately.")
            raise RuntimeError("CUDA unavailable.")
            
        if not os.path.exists(model_path):
            self.logger.error(f"Model not found locally at {model_path}. Please download it on hpc-gw first.")
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.device = torch.device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Log GPU Info
        self._log_gpu_info()
        
        # Load model entirely offline
        self.logger.info(f"Loading offline model from {model_path}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_path, 
                local_files_only=True
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_path, 
                local_files_only=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}. Error: {str(e)}")
            raise e

        # Move model to CUDA
        self.model.to(self.device)
        
        # Use DataParallel if multiple GPUs are available and device is 'cuda' (not specific cuda:0)
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and device == "cuda":
            self.logger.info(f"Enabling DataParallel across {gpu_count} GPUs.")
            self.model = nn.DataParallel(self.model)

        self.model.eval()

        # Sanity check: Ensure no parameters are on CPU
        for name, param in self.model.named_parameters():
            if param.device.type == 'cpu':
                self.logger.error(f"Parameter {name} resides on CPU. Aborting.")
                raise RuntimeError("All model parameters must be on CUDA.")

    def _log_gpu_info(self):
        """Logs information about the available GPUs."""
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            self.logger.info(f"GPU {i}: {gpu_name} ({gpu_mem:.2f} GB) - CUDA: {torch.version.cuda}")

    @torch.no_grad()
    def predict_depth(self, images):
        """
        Predicts depth maps for a batch of images.
        
        Args:
            images: A list of PIL Images or numpy arrays (batch).
            
        Returns:
            predicted_depth: Tensor of shape (batch_size, height, width)
        """
        # Prepare inputs and move to device
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Use Mixed Precision for faster inference
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(**inputs)
            
        # The output contains predicted_depth
        predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        # Assuming all images in batch have same size
        # Original size is (height, width)
        import torch.nn.functional as F
        if isinstance(images, list):
            target_size = images[0].size[::-1] # PIL size is (width, height), we need (height, width)
        elif hasattr(images, 'shape'):
            target_size = images.shape[1:3] # numpy array (batch, height, width, channels)
        else:
            target_size = (predicted_depth.shape[1], predicted_depth.shape[2])
            
        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        return predicted_depth
