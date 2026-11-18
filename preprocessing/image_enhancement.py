"""
Image Enhancement Module for Retinal Fundus Photography
Implements:
1. Aspect-ratio preserved crop and circular mask extraction
2. LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE)
3. Ben Graham Color Normalization & local contrast adjustment
4. Mild micro-vascular sharpening without artifact fabrication
5. Tensor transformation for PyTorch CNN inference
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def crop_retina_circle(image_bgr, tolerance=15):
    """
    Crops uninformative black borders around the retinal fundus circular field.
    Preserves exact aspect ratio and avoids cropping actual retinal lesions.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    
    # Check if any significant retina was found
    if not np.any(mask):
        return image_bgr, None

    # Get bounding coordinates of non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Ensure valid bounding box
    if rmax > rmin and cmax > cmin:
        cropped = image_bgr[rmin:rmax+1, cmin:cmax+1]
        return cropped, (rmin, rmax, cmin, cmax)
    return image_bgr, None

def apply_clahe_lab(image_bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    exclusively on the Lightness (L) channel in CIE LAB color space.
    This improves contrast of microaneurysms and hemorrhages without altering hue or saturation.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_merged = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def apply_ben_graham_enhancement(image_bgr, target_size=(512, 512), sigma=10):
    """
    Applies Ben Graham's method (standard in Kaggle/APTOS DR competitions):
    Local color normalization: img = 4 * img - 4 * blur + 128
    Reduces variations between different camera models and illumination conditions across rural PHCs.
    """
    # Resize to standardized dimensions
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    
    # Gaussian blur for background illumination field estimation
    blur = cv2.GaussianBlur(resized, (0, 0), sigma)
    
    # Weighted difference + 128 midpoint
    normalized = cv2.addWeighted(resized, 4.0, blur, -4.0, 128)
    
    # Add a circular mask to clean outer boundary
    h, w = target_size
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.48)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Smooth mask border
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    mask_3c = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)
    
    enhanced = (normalized * mask_3c + 128 * (1 - mask_3c)).astype(np.uint8)
    return enhanced

def apply_mild_sharpening(image_bgr, amount=0.4):
    """
    Applies controlled unsharp masking to enhance fine vascular edges without creating noise.
    """
    gaussian = cv2.GaussianBlur(image_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, gaussian, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def enhance_retinal_image(image_bgr, target_size=(512, 512)):
    """
    Full clinical preprocessing pipeline for rural screening:
    1. Border cropping
    2. Resize preserving aspect ratio / pad
    3. LAB CLAHE enhancement
    4. Mild micro-sharpening
    Returns enhanced BGR image ready for display and feature extraction.
    """
    # 1. Crop uninformative dark margins
    cropped, _ = crop_retina_circle(image_bgr)
    
    # 2. Resize to target dimension
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. LAB CLAHE
    enhanced_clahe = apply_clahe_lab(resized, clip_limit=2.5, tile_grid_size=(8, 8))
    
    # 4. Mild vascular sharpening
    final_enhanced = apply_mild_sharpening(enhanced_clahe, amount=0.35)
    
    return final_enhanced

def prepare_tensor_for_model(image_bgr, target_size=(224, 224), device='cpu'):
    """
    Converts enhanced BGR image to normalized PyTorch tensor ready for CNN classification.
    Uses standard ImageNet normalization:
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    return tensor
