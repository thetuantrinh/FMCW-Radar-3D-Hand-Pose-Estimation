import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model for offline use.")
    parser.add_argument("--model", type=str, required=True, help="Model ID on HuggingFace")
    parser.add_argument("--save_dir", type=str, required=True, help="Local directory to save the model")
    
    args = parser.parse_args()
    
    print(f"Downloading model '{args.model}'...")
    print(f"This might take a while depending on your internet connection.")
    
    # We load the model via the pipeline (or auto classes) and save it.
    # For Depth Anything V2:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model)
    
    processor.save_pretrained(args.save_dir)
    model.save_pretrained(args.save_dir)
    
    print(f"Successfully downloaded and saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
