"""
Image Quality Assessment Module for Retinal Fundus Photographs
Designed for Primary Health Centres (PHC) in Rural India.

Checks for:
1. Focus / Blur (Laplacian Variance)
2. Illumination (Mean luminosity, overexposure, underexposure, quadrant uniformity)
3. Field of View (Retinal circle coverage, excessive border ratio)
"""

import cv2
import numpy as np

def assess_image_quality(image_bgr, thresholds=None):
    """
    Evaluates retinal image quality prior to clinical AI inference.
    Returns:
        dict: {
            'status': 'Pass' | 'Warning' | 'Fail',
            'score': float (0-100),
            'metrics': dict of individual numerical metrics,
            'issues': list of detected problems,
            'recommendations': list of actionable steps for rural healthcare worker
        }
    """
    default_thresholds = {
        'min_laplacian_var': 25.0,
        'warning_laplacian_var': 60.0,
        'min_brightness': 30.0,
        'max_brightness': 235.0,
        'min_fov_ratio': 0.20,
        'max_quadrant_imbalance': 3.0
    }
    if thresholds is None:
        thresholds = default_thresholds
    else:
        merged = default_thresholds.copy()
        merged.update(thresholds)
        thresholds = merged

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    issues = []
    recommendations = []
    scores = []

    # -------------------------------------------------------------
    # 1. Field of View (FOV) & Retinal Mask Extraction
    # -------------------------------------------------------------
    # Retinal fundus is usually a brightly colored circle on dark background
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Clean mask with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    retinal_pixels = cv2.countNonZero(mask_cleaned)
    total_pixels = h * w
    fov_ratio = retinal_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    if fov_ratio < 0.15:
        issues.append("Retinal field too small or obstructed (FOV < 15%)")
        recommendations.append("Position the portable fundus camera closer to the patient's pupil.")
        fov_score = 15.0
    elif fov_ratio < thresholds['min_fov_ratio']:
        issues.append("Excessive black border or decentered retinal view.")
        recommendations.append("Center the fundus camera lens on the optic axis.")
        fov_score = 55.0
    else:
        fov_score = min(100.0, 50.0 + (fov_ratio * 70.0))
    scores.append(fov_score)

    # -------------------------------------------------------------
    # 2. Focus & Blur Assessment (Laplacian Variance inside retina)
    # -------------------------------------------------------------
    # Only calculate blur within the actual retinal region to avoid border edge artifacts
    if retinal_pixels > 100:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Mask out background border
        laplacian_retina = laplacian[mask_cleaned > 0]
        laplacian_var = float(laplacian_retina.var()) if len(laplacian_retina) > 0 else 0.0
    else:
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if laplacian_var < thresholds['min_laplacian_var']:
        issues.append("Image is severely blurred (low focal clarity).")
        recommendations.append("Adjust camera diopter focus ring or ask patient to fixate on the internal target.")
        blur_score = max(10.0, (laplacian_var / thresholds['min_laplacian_var']) * 50.0)
    elif laplacian_var < thresholds['warning_laplacian_var']:
        issues.append("Moderate blur detected; minor loss of vessel sharpness.")
        recommendations.append("Stabilize camera and ensure patient remains still during flash.")
        blur_score = 65.0 + (laplacian_var - thresholds['min_laplacian_var']) / (thresholds['warning_laplacian_var'] - thresholds['min_laplacian_var']) * 25.0
    else:
        blur_score = min(100.0, 85.0 + min(15.0, (laplacian_var - thresholds['warning_laplacian_var']) / 50.0))
    scores.append(blur_score)

    # -------------------------------------------------------------
    # 3. Illumination & Exposure Assessment
    # -------------------------------------------------------------
    if retinal_pixels > 100:
        mean_brightness = float(np.mean(gray[mask_cleaned > 0]))
        std_brightness = float(np.std(gray[mask_cleaned > 0]))
    else:
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

    illum_score = 100.0
    if mean_brightness < thresholds['min_brightness']:
        issues.append("Image is severely underexposed (too dark to discern lesions).")
        recommendations.append("Increase illumination intensity or check if pupil dilation (mydriasis) is required.")
        illum_score -= 50.0
    elif mean_brightness < 55.0:
        issues.append("Mild underexposure detected.")
        recommendations.append("Ensure ambient room lights are dimmed at the PHC.")
        illum_score -= 20.0
    elif mean_brightness > thresholds['max_brightness']:
        issues.append("Image is overexposed/flash washed out.")
        recommendations.append("Reduce camera flash power or prevent direct specular reflection.")
        illum_score -= 50.0
    elif mean_brightness > 195.0:
        issues.append("High flash glare detected.")
        recommendations.append("Re-angle camera slightly to avoid central corneal reflection.")
        illum_score -= 20.0

    # 4. Illumination Uniformity (Quadrant Comparison)
    h_mid, w_mid = h // 2, w // 2
    quadrants = [
        gray[0:h_mid, 0:w_mid],
        gray[0:h_mid, w_mid:w],
        gray[h_mid:h, 0:w_mid],
        gray[h_mid:h, w_mid:w]
    ]
    q_means = [float(np.mean(q)) for q in quadrants]
    min_q = max(min(q_means), 1.0)
    max_q = max(q_means)
    quadrant_ratio = max_q / min_q

    if quadrant_ratio > thresholds['max_quadrant_imbalance']:
        issues.append("Severe uneven illumination across quadrants (shadowing/vignetting).")
        recommendations.append("Align camera squarely with the optical axis of the eye.")
        illum_score -= 25.0

    scores.append(max(10.0, illum_score))

    # -------------------------------------------------------------
    # Overall Quality Scoring & Verdict
    # -------------------------------------------------------------
    # Weighted score: Blur (40%), Illumination (35%), FOV (25%)
    overall_score = round((blur_score * 0.40) + (illum_score * 0.35) + (fov_score * 0.25), 1)

    if laplacian_var < thresholds['min_laplacian_var'] or mean_brightness < thresholds['min_brightness'] or fov_ratio < 0.15:
        status = 'Fail'
        if not recommendations:
            recommendations.append("Capture a fresh image following proper positioning protocols.")
    elif overall_score < 70.0 or issues:
        status = 'Warning'
    else:
        status = 'Pass'

    metrics = {
        'laplacian_var': round(laplacian_var, 2),
        'blur_score': round(blur_score, 1),
        'mean_brightness': round(mean_brightness, 2),
        'std_brightness': round(std_brightness, 2),
        'fov_ratio': round(fov_ratio, 3),
        'quadrant_ratio': round(quadrant_ratio, 2)
    }

    return {
        'status': status,
        'score': overall_score,
        'metrics': metrics,
        'issues': issues,
        'recommendations': recommendations
    }
