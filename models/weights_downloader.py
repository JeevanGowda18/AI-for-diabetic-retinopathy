"""
Model Weights Downloader and Integrator for DR Vision AI
Provides clear documentation and utility methods for medical institutions,
data scientists, and engineers to integrate fine-tuned ResNet-50 weights
trained on APTOS 2019 or EyePACS datasets.
"""

import sys
import os
import argparse
from pathlib import Path
import torch

from models.dr_model import DRResNet50
from config import Config

def verify_and_integrate_weights(weights_file_path):
    """
    Validates that a provided .pth file matches the DRResNet50 5-class architecture
    and installs it into the models directory.
    """
    path = Path(weights_file_path)
    if not path.exists():
        print(f"[Error] Weights file not found at: {path}")
        return False

    print(f"[Verification] Inspecting weights file: {path.name}")
    try:
        state_dict = torch.load(path, map_location='cpu')
        
        # Test loading into architecture
        model = DRResNet50(num_classes=5, pretrained_backbone=False)
        model.load_state_dict(state_dict)
        print("[Success] Weights state_dict matches DRResNet50 architecture (5 classes).")
        
        target_path = Path(Config.MODEL_WEIGHTS_PATH)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, target_path)
        print(f"[Installed] Production weights deployed to: {target_path}")
        return True
    except Exception as e:
        print(f"[Error] Invalid weights file: {e}")
        return False

def download_weights_from_url(url, target_path=None):
    """
    Downloads model weights from an authorized HTTP/HTTPS endpoint with progress.
    """
    import urllib.request
    if target_path is None:
        target_path = Path(Config.MODEL_WEIGHTS_PATH)
    else:
        target_path = Path(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download = target_path.with_suffix('.tmp')

    print(f"[Download] Fetching model weights from: {url}")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(100, (blocknum * blocksize * 100) / totalsize)
                print(f"\rDownloading: {percent:.1f}% ({blocknum * blocksize / (1024*1024):.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, str(temp_download), reporthook=reporthook)
        print("\n[Download] Complete. Verifying state dictionary...")
        
        if verify_and_integrate_weights(str(temp_download)):
            if temp_download.exists() and temp_download != target_path:
                temp_download.unlink()
            return True
        else:
            if temp_download.exists():
                temp_download.unlink()
            return False
    except Exception as e:
        if temp_download.exists():
            temp_download.unlink()
        print(f"\n[Download Error] Failed to retrieve weights: {e}")
        return False

def print_dataset_instructions():
    """Prints instructions for accessing APTOS 2019 and EyePACS datasets."""
    instructions = """
================================================================================
DR VISION AI - RETINAL MODEL WEIGHTS INTEGRATION GUIDE
================================================================================

To deploy this application with clinically validated production weights:

1. RECOMMENDED DATASETS:
   - APTOS 2019 Blindness Detection (Kaggle / Aravind Eye Hospital):
     https://www.kaggle.com/c/aptos2019-blindness-detection
     Contains 3,662 high-resolution retinal images rated by ophthalmologists.
   - EyePACS (Diabetic Retinopathy Detection):
     https://www.kaggle.com/c/diabetic-retinopathy-detection
     Over 35,000 images collected from rural and urban clinics.

2. MODEL TRAINING SPECIFICATIONS:
   - Backbone: ResNet-50 (ImageNet initial weights)
   - Input size: 224x224 (or 512x512 with adaptive pooling)
   - Loss Function: Quadratic Weighted Kappa (QWK) Loss or Smooth L1 Loss / Focal Loss
   - Optimizer: AdamW with Cosine Annealing learning rate schedule
   - Number of output classes: 5 (Grades 0 to 4)

3. INTEGRATION COMMAND:
   python models/weights_downloader.py --integrate /path/to/trained_dr_model.pth

4. AUTOMATIC DETECTION:
   Placing the weights file at:
   models/dr_resnet50_weights.pth
   will automatically activate Production Mode across the application.
================================================================================
"""
    print(instructions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DR Vision AI Model Integrator")
    parser.add_argument('--integrate', type=str, help="Path to trained .pth weights file")
    parser.add_argument('--download', type=str, help="Direct URL to trained .pth weights file")
    parser.add_argument('--info', action='store_true', help="Show dataset training instructions")
    args = parser.parse_args()

    if args.integrate:
        verify_and_integrate_weights(args.integrate)
    elif args.download:
        download_weights_from_url(args.download)
    else:
        print_dataset_instructions()
