"""
Retinal Structure and Lesion Candidate Analysis (Computer Vision)
Provides auxiliary visual aids:
- Optic disc localization
- Estimated fovea macula center
- Retinal blood vessel segmentation
- Bright lesion candidates (possible exudates)
- Dark lesion candidates (possible microaneurysms / hemorrhages)

IMPORTANT CLINICAL DISCLAIMER:
These detections are experimental computer vision heuristic markers intended for
visual orientation and educational context. They are NOT certified clinical diagnoses.
Final clinical assessment must be made by a certified ophthalmologist.
"""

import cv2
import numpy as np

def analyze_retinal_structures(image_bgr):
    """
    Performs classical computer vision analysis on retinal fundus image.
    Returns:
        annotated_bgr: image with structure annotations overlaid
        vessels_mask: binary vessel segmentation
        stats: dictionary of candidate counts and detected landmarks
    """
    h, w = image_bgr.shape[:2]
    annotated = image_bgr.copy()
    
    # Work primarily on Green channel for maximum retinal contrast
    b, g, r = cv2.split(image_bgr)
    
    # Retinal Field of View mask
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, fov_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel_fov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, kernel_fov)
    fov_mask = cv2.erode(fov_mask, kernel_fov, iterations=2) # Inset slightly to avoid border artifacts

    # -------------------------------------------------------------
    # 1. Retinal Blood Vessel Segmentation (Morphological Top-Hat)
    # -------------------------------------------------------------
    # Invert green channel so vessels are bright
    inv_green = cv2.bitwise_not(g)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    inv_g_clahe = clahe.apply(inv_green)
    
    # Multi-scale morphological closing and opening
    kernel_vessel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top_hat = cv2.morphologyEx(inv_g_clahe, cv2.MORPH_TOPHAT, kernel_vessel)
    
    # Adaptive thresholding for vessels
    vessels = cv2.adaptiveThreshold(
        top_hat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    vessels = cv2.bitwise_and(vessels, vessels, mask=fov_mask)
    # Clean noise
    vessels = cv2.medianBlur(vessels, 3)

    # -------------------------------------------------------------
    # 2. Optic Disc Localization
    # -------------------------------------------------------------
    # Optic disc is the brightest compact circular region in the retina
    # Blur heavily to suppress vessels and small lesions
    blurred_g = cv2.GaussianBlur(g, (35, 35), 0)
    blurred_g_masked = cv2.bitwise_and(blurred_g, blurred_g, mask=fov_mask)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_g_masked)
    od_center = max_loc
    od_radius = int(min(h, w) * 0.08)  # Approximate OD radius

    # Create Optic Disc exclusion mask for lesion search
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, od_center, int(od_radius * 1.5), 255, -1)
    
    # -------------------------------------------------------------
    # 3. Estimated Foveal / Macular Center
    # -------------------------------------------------------------
    # Macula is located approximately 2.5 OD diameters temporally
    # In fundus photography: if OD is on right (left eye), fovea is to the left;
    # if OD is on left (right eye), fovea is to the right.
    temporal_direction = 1 if od_center[0] < (w // 2) else -1
    fovea_offset_x = int(temporal_direction * od_radius * 2.8)
    fovea_x = max(10, min(w - 10, od_center[0] + fovea_offset_x))
    fovea_y = max(10, min(h - 10, od_center[1] + int(od_radius * 0.2)))
    fovea_center = (fovea_x, fovea_y)

    # -------------------------------------------------------------
    # 4. Bright Lesion Candidates (Hard Exudates / Cotton Wool Spots)
    # -------------------------------------------------------------
    # High intensity in L channel and high in Green, but excluding Optic Disc
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_chan, _, _ = cv2.split(lab)
    
    # Threshold bright spots
    l_mean, l_std = cv2.meanStdDev(l_chan, mask=fov_mask)
    bright_thresh = int(l_mean[0][0] + 1.6 * l_std[0][0])
    _, bright_mask = cv2.threshold(l_chan, max(160, bright_thresh), 255, cv2.THRESH_BINARY)
    
    # Exclude OD region and outside FOV
    bright_mask = cv2.bitwise_and(bright_mask, fov_mask)
    bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(od_mask))
    
    # Filter small contours
    contours_bright, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exudate_candidates = []
    for cnt in contours_bright:
        area = cv2.contourArea(cnt)
        if 8 < area < 1500: # Exclude tiny noise and huge flash glare
            x, y, cw, ch = cv2.boundingRect(cnt)
            exudate_candidates.append((x, y, cw, ch))

    # -------------------------------------------------------------
    # 5. Dark Lesion Candidates (Microaneurysms & Hemorrhages)
    # -------------------------------------------------------------
    # Dark in green channel, not part of major vessel tree
    g_mean, g_std = cv2.meanStdDev(g, mask=fov_mask)
    dark_thresh = int(g_mean[0][0] - 1.2 * g_std[0][0])
    _, dark_mask = cv2.threshold(g, max(20, dark_thresh), 255, cv2.THRESH_BINARY_INV)
    
    dark_mask = cv2.bitwise_and(dark_mask, fov_mask)
    # Remove blood vessels to isolate focal spots
    dark_mask_no_vessels = cv2.bitwise_and(dark_mask, cv2.bitwise_not(vessels))
    
    contours_dark, _ = cv2.findContours(dark_mask_no_vessels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_lesion_candidates = []
    for cnt in contours_dark:
        area = cv2.contourArea(cnt)
        if 6 < area < 800:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Circularity check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.35: # reasonably round/focal
                    red_lesion_candidates.append((x, y, cw, ch))

    # -------------------------------------------------------------
    # Render Overlay Visualizations
    # -------------------------------------------------------------
    # 1. Vessel overlay in subtle semi-transparent cyan
    vessel_color = np.zeros_like(annotated)
    vessel_color[vessels > 0] = [200, 255, 100] # BGR light green-cyan
    annotated = cv2.addWeighted(annotated, 0.82, vessel_color, 0.40, 0)

    # 2. Optic Disc ring (Bright Cyan)
    cv2.circle(annotated, od_center, od_radius, (255, 255, 0), 2)
    cv2.putText(annotated, "Optic Disc", (od_center[0] - 35, od_center[1] - od_radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # 3. Fovea Center (Yellow crosshair)
    cv2.drawMarker(annotated, fovea_center, (0, 255, 255), markerType=cv2.MARKER_CROSS,
                   markerSize=22, thickness=2, line_type=cv2.LINE_AA)
    cv2.putText(annotated, "Macula / Fovea", (fovea_center[0] - 45, fovea_center[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. Mark candidate exudates (Bright Yellow/Orange squares)
    for (x, y, cw, ch) in exudate_candidates[:30]:
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 215, 255), 1)

    # 5. Mark candidate red lesions / hemorrhages (Red circles)
    for (x, y, cw, ch) in red_lesion_candidates[:40]:
        center_l = (x + cw // 2, y + ch // 2)
        r_l = max(3, int(max(cw, ch) / 1.8))
        cv2.circle(annotated, center_l, r_l, (0, 0, 255), 1)

    # Add legend at bottom
    leg_h = 32
    overlay_bar = annotated.copy()
    cv2.rectangle(overlay_bar, (0, h - leg_h), (w, h), (20, 24, 33), -1)
    annotated = cv2.addWeighted(annotated, 0.25, overlay_bar, 0.75, 0)
    
    cv2.putText(annotated, "CV Cand: Yellow=Exudates | Red=Red Lesions | Cyan=Optic Disc | Green=Vessels",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

    stats = {
        'optic_disc_center': od_center,
        'optic_disc_radius': od_radius,
        'fovea_center': fovea_center,
        'exudate_candidates_count': len(exudate_candidates),
        'red_lesion_candidates_count': len(red_lesion_candidates),
        'vessel_density_percent': round((cv2.countNonZero(vessels) / float(max(1, cv2.countNonZero(fov_mask)))) * 100, 1)
    }

    return annotated, vessels, stats
