"""
Image Enhancement Module for Retinal Fundus Photography
Implements:
1. Fundus circle detection + 1:1 square black-padding (minEnclosingCircle)
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


def detect_and_crop_fundus_circle(img_bgr, target_size=(512, 512)):
    """
    Detects the retinal fundus circular field using contour analysis and
    minEnclosingCircle, then extracts a 1:1 square crop with black padding
    so the retina is never stretched. Applies a soft circular mask to remove
    camera vignette artifacts at the boundary.

    Args:
        img_bgr: Input BGR image (any aspect ratio)
        target_size: (height, width) tuple for final output

    Returns:
        result_img: Cropped, padded, and masked BGR image of shape target_size
        circle_info: Tuple (cx, cy, radius) of detected fundus circle in original coords
    """
    h, w = img_bgr.shape[:2]

    # --- Step 1: Threshold + morphology to isolate retinal disk ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    # Morphological close+open to fill gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- Step 2: Find largest contour → minEnclosingCircle ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour by area → most likely the retinal disk
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cx, cy, radius = int(cx), int(cy), int(radius)
    else:
        # Fallback: assume centered fundus
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 2

    # Sanity check: reject degenerate detections (< 25% of image min dim)
    min_expected_radius = min(h, w) * 0.25
    if radius < min_expected_radius:
        cx, cy = w // 2, h // 2
        radius = min(h, w) // 2

    # --- Step 3: 1:1 square crop with black padding ---
    r = int(radius * 1.02)   # 2% margin to avoid clipping retinal edges
    side = 2 * r
    square = np.zeros((side, side, 3), dtype=np.uint8)

    # Source ROI (clamped to image boundaries)
    src_x1 = max(0, cx - r)
    src_y1 = max(0, cy - r)
    src_x2 = min(w, cx + r)
    src_y2 = min(h, cy + r)

    # Destination offsets in the square canvas
    dst_x1 = r - (cx - src_x1)
    dst_y1 = r - (cy - src_y1)

    square[dst_y1:dst_y1 + (src_y2 - src_y1),
           dst_x1:dst_x1 + (src_x2 - src_x1)] = img_bgr[src_y1:src_y2, src_x1:src_x2]

    # --- Step 4: Resize to target_size ---
    resized = cv2.resize(square, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_AREA)

    # --- Step 5: Apply smooth circular mask (removes vignette / camera border) ---
    th, tw = target_size
    tcx, tcy = tw // 2, th // 2
    tr = int(min(th, tw) * 0.485)

    circ_mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.circle(circ_mask, (tcx, tcy), tr, 255, -1)
    # Gaussian feather for smooth edge blend
    circ_mask_smooth = cv2.GaussianBlur(circ_mask, (11, 11), 0) / 255.0
    circ_mask_3c = np.repeat(circ_mask_smooth[:, :, np.newaxis], 3, axis=2)

    result_img = (resized * circ_mask_3c).astype(np.uint8)

    return result_img, (cx, cy, radius)


def crop_retina_circle(image_bgr, tolerance=15):
    """
    Legacy function: crops uninformative black borders.
    Kept for backward compatibility. New code should call detect_and_crop_fundus_circle.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance

    if not np.any(mask):
        return image_bgr, None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if rmax > rmin and cmax > cmin:
        cropped = image_bgr[rmin:rmax + 1, cmin:cmax + 1]
        return cropped, (rmin, rmax, cmin, cmax)
    return image_bgr, None


def apply_clahe_lab(image_bgr, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies CLAHE exclusively on the L channel in CIE LAB color space.
    Enhances microaneurysm/hemorrhage contrast without altering hue.
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
    Ben Graham normalization: 4*img - 4*blur + 128.
    Reduces camera-to-camera illumination variation across rural PHC cameras.
    """
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(resized, (0, 0), sigma)
    normalized = cv2.addWeighted(resized, 4.0, blur, -4.0, 128)

    h, w = target_size
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.48)
    cv2.circle(mask, center, radius, 255, -1)

    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    mask_3c = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)

    enhanced = (normalized * mask_3c + 128 * (1 - mask_3c)).astype(np.uint8)
    return enhanced


def apply_mild_sharpening(image_bgr, amount=0.4):
    """
    Unsharp masking to enhance fine vascular edges without noise amplification.
    """
    gaussian = cv2.GaussianBlur(image_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, gaussian, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_retinal_image(image_bgr, target_size=(512, 512)):
    """
    Full clinical preprocessing pipeline for rural DR screening.

    Pipeline:
      1. Fundus circle detection → 1:1 square black-padding (new: replaces legacy crop)
      2. LAB CLAHE contrast enhancement
      3. Mild micro-vascular unsharp sharpening

    Returns:
        enhanced_bgr: BGR image of shape target_size, ready for display and CV analysis
    """
    # --- Step 1: Detect fundus circle, crop to square, resize, mask ---
    cropped, circle_info = detect_and_crop_fundus_circle(image_bgr, target_size=target_size)

    if circle_info is not None:
        cx, cy, r = circle_info
        print(f"[Preprocessing] Fundus circle detected: center=({cx},{cy}), radius={r}px")
    else:
        print("[Preprocessing] Fundus circle detection fallback: using image center")

    # --- Step 2: LAB CLAHE ---
    enhanced_clahe = apply_clahe_lab(cropped, clip_limit=2.5, tile_grid_size=(8, 8))

    # --- Step 3: Mild vascular sharpening ---
    final_enhanced = apply_mild_sharpening(enhanced_clahe, amount=0.35)

    return final_enhanced


def prepare_tensor_for_model(image_bgr, target_size=(224, 224), device='cpu'):
    """
    Converts enhanced BGR image to normalized PyTorch tensor for CNN inference.

    Uses standard ImageNet normalization:
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Args:
        image_bgr: Enhanced BGR image (output of enhance_retinal_image)
        target_size: (H, W) for model input (224x224 for ResNet/EfficientNet)
        device: 'cpu' or 'cuda'

    Returns:
        tensor: 1×3×H×W float32 tensor on `device`
    """
    # Convert BGR → RGB
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
