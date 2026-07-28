import os
import argparse
import time
import logging
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Internal modules
from depth_engine import DepthEstimator
from annotation_updater import process_annotation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("GenerateDepth")

class RGBDataset(Dataset):
    def __init__(self, image_dir: str, hand_json_dir: str, output_dir: str, recursive: bool = False, overwrite: bool = False):
        self.image_dir = Path(image_dir)
        self.hand_json_dir = Path(hand_json_dir)
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        
        self.image_paths = []
        
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        
        logger.info(f"Scanning for images in {self.image_dir} (Recursive: {recursive})...")
        for ext in extensions:
            if recursive:
                self.image_paths.extend(self.image_dir.rglob(ext))
            else:
                self.image_paths.extend(self.image_dir.glob(ext))
                
        self.image_paths = sorted(self.image_paths)
        logger.info(f"Found {len(self.image_paths)} images.")
        
        # Pre-filter files to skip processed ones if not overwriting
        self.valid_samples = []
        for img_path in self.image_paths:
            rel_path = img_path.relative_to(self.image_dir)
            json_name = rel_path.with_suffix('.json')
            
            input_json_path = self.hand_json_dir / json_name
            output_json_path = self.output_dir / json_name
            
            if not input_json_path.exists():
                # Missing original annotation, we skip but don't count as failure yet
                continue
                
            if not self.overwrite and output_json_path.exists():
                # Already processed
                continue
                
            self.valid_samples.append({
                "img_path": str(img_path),
                "input_json_path": str(input_json_path),
                "output_json_path": str(output_json_path)
            })
            
        logger.info(f"Found {len(self.valid_samples)} samples to process after checking overwrite and existing JSONs.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        img_path = sample["img_path"]
        
        try:
            image = Image.open(img_path).convert("RGB")
            return {
                "image": image, # PIL Image will be batched by a custom collate_fn if size differs, but we assume same size
                "input_json_path": sample["input_json_path"],
                "output_json_path": sample["output_json_path"],
                "status": "success"
            }
        except Exception as e:
            return {
                "image": None,
                "input_json_path": sample["input_json_path"],
                "output_json_path": sample["output_json_path"],
                "status": f"failed: {str(e)}"
            }

def custom_collate(batch):
    images = []
    input_json_paths = []
    output_json_paths = []
    statuses = []
    
    for item in batch:
        images.append(item["image"])
        input_json_paths.append(item["input_json_path"])
        output_json_paths.append(item["output_json_path"])
        statuses.append(item["status"])
        
    return {
        "images": images,
        "input_json_paths": input_json_paths,
        "output_json_paths": output_json_paths,
        "statuses": statuses
    }

def main():
    parser = argparse.ArgumentParser(description="Generate 3D depth annotations using Depth Anything.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--hand_json_dir", type=str, required=True, help="Directory containing original 2D JSONs")
    parser.add_argument("--radar_dir", type=str, default="", help="Directory containing radar data (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output 3D JSONs")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to Depth Anything Model")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (must be cuda)")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    
    args = parser.parse_args()
    
    # 1. Initialize DepthEstimator
    logger.info("Initializing Depth Estimator...")
    try:
        depth_estimator = DepthEstimator(model_path=args.model_path, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize Depth Estimator: {e}")
        return
        
    # 2. Setup Dataset & DataLoader
    dataset = RGBDataset(
        image_dir=args.image_dir,
        hand_json_dir=args.hand_json_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite
    )
    
    if len(dataset) == 0:
        logger.info("No samples to process. Exiting.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=True if args.device.startswith("cuda") else False
    )
    
    # 3. Processing Loop
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()
    
    logger.info(f"Starting inference with batch size {args.batch_size}...")
    
    pbar = tqdm(total=len(dataset), desc="Generating Depth")
    
    for batch in dataloader:
        images = batch["images"]
        input_paths = batch["input_json_paths"]
        output_paths = batch["output_json_paths"]
        statuses = batch["statuses"]
        
        valid_indices = [i for i, s in enumerate(statuses) if s == "success"]
        failed_indices = [i for i, s in enumerate(statuses) if s != "success"]
        
        for idx in failed_indices:
            logger.error(f"Failed to load image for {input_paths[idx]}: {statuses[idx]}")
            failed_count += 1
            pbar.update(1)
            
        if not valid_indices:
            continue
            
        valid_images = [images[i] for i in valid_indices]
        valid_input = [input_paths[i] for i in valid_indices]
        valid_output = [output_paths[i] for i in valid_indices]
        
        # Predict Depth
        try:
            depth_maps = depth_estimator.predict_depth(valid_images) # Returns (batch, H, W)
            
            # Process and save JSONs
            for i in range(len(valid_images)):
                try:
                    process_annotation(valid_input[i], valid_output[i], depth_maps[i])
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to process annotation for {valid_input[i]}: {e}")
                    failed_count += 1
                    
        except Exception as e:
            logger.error(f"Inference failed for batch: {e}")
            failed_count += len(valid_images)
            
        pbar.update(len(valid_indices))
        
        # Monitor Peak GPU Memory
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            pbar.set_postfix({"Peak VRAM (GB)": f"{mem_allocated:.2f}"})

    pbar.close()
    
    # 4. Final Report
    end_time = time.time()
    total_runtime = end_time - start_time
    avg_inference_time = total_runtime / max(1, processed_count)
    
    logger.info("=" * 60)
    logger.info("FINAL EXECUTION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total images requested : {len(dataset)}")
    logger.info(f"Successfully processed : {processed_count}")
    logger.info(f"Skipped                : {skipped_count} (Excluding already processed/missing)")
    logger.info(f"Failed                 : {failed_count}")
    logger.info("-" * 60)
    logger.info(f"Total runtime          : {total_runtime:.2f} seconds")
    logger.info(f"Avg time per image     : {avg_inference_time:.4f} seconds")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            peak_mem = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            logger.info(f"Peak GPU {i} Memory     : {peak_mem:.2f} GB")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Prevent multiprocessing issues with PyTorch in some environments
    multiprocessing.set_start_method('spawn', force=True)
    main()
