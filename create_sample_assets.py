"""
Generates realistic medical retinal fundus representations for demonstration,
runs the enhancement, Grad-CAM and PDF generator, ensuring complete end-to-end assets.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json

from preprocessing.image_quality import assess_image_quality
from preprocessing.image_enhancement import enhance_retinal_image, prepare_tensor_for_model
from preprocessing.retinal_analysis import analyze_retinal_structures
from models.model_loader import ModelService
from explainability.gradcam import GradCAM
from reports.pdf_generator import generate_screening_pdf
from config import Config

def create_synthetic_fundus(grade=0, width=512, height=512):
    """
    Creates an anatomically accurate synthetic retinal fundus image:
    - Circular retinal field with dark boundary
    - Orange-red retinal pigment epithelium gradient
    - Optic disc (yellowish oval with cup)
    - Retinal blood vessels (superior and inferior vascular arcades)
    - Macula / Fovea centralis (darker avascular zone)
    - Grade-specific diabetic retinopathy lesions:
      Grade 0: Normal retina
      Grade 1: Isolated microaneurysms
      Grade 2: Multiple microaneurysms, hard exudates
      Grade 3: Extensive blot hemorrhages, cotton wool spots
      Grade 4: Neovascularization and proliferative changes
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.46)

    # 1. Base retinal background gradient (Orange-Red / Choroidal)
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    retina_mask = dist_from_center <= radius

    # Radial gradient from warm amber center to deep brick-red periphery
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    r_chan = np.clip(220 - norm_dist * 80 + np.random.normal(0, 3, (height, width)), 0, 255).astype(np.uint8)
    g_chan = np.clip(95 - norm_dist * 50 + np.random.normal(0, 2, (height, width)), 0, 255).astype(np.uint8)
    b_chan = np.clip(25 - norm_dist * 18 + np.random.normal(0, 1, (height, width)), 0, 255).astype(np.uint8)

    img[:, :, 2] = np.where(retina_mask, r_chan, 0)
    img[:, :, 1] = np.where(retina_mask, g_chan, 0)
    img[:, :, 0] = np.where(retina_mask, b_chan, 0)

    # 2. Optic Disc (Nasal side)
    od_x = center[0] - int(radius * 0.45)
    od_y = center[1]
    od_r = int(radius * 0.16)
    
    cv2.circle(img, (od_x, od_y), od_r, (70, 180, 245), -1) # Pale yellow disc
    cv2.circle(img, (od_x, od_y), int(od_r * 0.55), (100, 210, 255), -1) # Physiologic cup

    # 3. Retinal Blood Vessels (Superior & Inferior Arcades)
    vessel_color = (15, 30, 120) # Deep dark red/brown
    for branch_y in [-1, 1]:
        # Main temporal arcade
        pts = [
            (od_x, od_y),
            (od_x + 40, od_y + branch_y * 45),
            (od_x + 110, od_y + branch_y * 95),
            (od_x + 200, od_y + branch_y * 115),
            (od_x + 280, od_y + branch_y * 85),
            (od_x + 330, od_y + branch_y * 35)
        ]
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, vessel_color, thickness=4, lineType=cv2.LINE_AA)
        
        # Secondary branches
        for i in range(1, len(pts)-1):
            start = tuple(pts[i][0])
            end = (start[0] + np.random.randint(15, 40), start[1] + branch_y * np.random.randint(15, 45))
            cv2.line(img, start, end, vessel_color, thickness=2, lineType=cv2.LINE_AA)

    # 4. Macula / Fovea Centralis (Darker avascular area)
    macula_x = center[0] + int(radius * 0.15)
    macula_y = center[1]
    cv2.circle(img, (macula_x, macula_y), int(radius * 0.12), (18, 45, 140), -1)
    cv2.circle(img, (macula_x, macula_y), int(radius * 0.04), (10, 30, 110), -1) # Fovea pit

    # 5. Inject DR Pathological Lesions based on Severity Grade
    np.random.seed(42 + grade)

    if grade >= 1: # Mild NPDR: isolated microaneurysms (tiny dark red dots)
        for _ in range(8):
            rx = int(macula_x + np.random.randint(-70, 70))
            ry = int(macula_y + np.random.randint(-60, 60))
            cv2.circle(img, (rx, ry), 2, (10, 15, 75), -1)

    if grade >= 2: # Moderate NPDR: microaneurysms + hard exudates (bright yellowish waxy deposits)
        for _ in range(18):
            rx = int(center[0] + np.random.randint(-120, 140))
            ry = int(center[1] + np.random.randint(-120, 120))
            # Hard exudate clusters
            cv2.circle(img, (rx, ry), np.random.randint(2, 5), (120, 230, 255), -1)
        # Small blot hemorrhages
        for _ in range(12):
            rx = int(center[0] + np.random.randint(-140, 140))
            ry = int(center[1] + np.random.randint(-130, 130))
            cv2.ellipse(img, (rx, ry), (np.random.randint(4, 9), np.random.randint(3, 7)),
                        np.random.randint(0, 180), 0, 360, (5, 10, 85), -1)

    if grade >= 3: # Severe NPDR: extensive blot hemorrhages in multiple quadrants
        for _ in range(35):
            rx = int(center[0] + np.random.randint(-160, 160))
            ry = int(center[1] + np.random.randint(-160, 160))
            if np.sqrt((rx - center[0])**2 + (ry - center[1])**2) < (radius * 0.88):
                cv2.ellipse(img, (rx, ry), (np.random.randint(6, 14), np.random.randint(4, 10)),
                            np.random.randint(0, 180), 0, 360, (5, 8, 80), -1)
        # Cotton wool spots (fluffy white lesions)
        for _ in range(6):
            rx = int(center[0] + np.random.randint(-100, 100))
            ry = int(center[1] + np.random.randint(-100, 100))
            cv2.circle(img, (rx, ry), np.random.randint(8, 14), (180, 210, 225), -1)

    if grade >= 4: # Proliferative DR: neovascularization fronds + preretinal vitreous hemorrhage
        # Neovascular fronds near optic disc
        for _ in range(25):
            start = (od_x + np.random.randint(-15, 25), od_y + np.random.randint(-25, 25))
            end = (start[0] + np.random.randint(-35, 45), start[1] + np.random.randint(-35, 45))
            cv2.line(img, start, end, (10, 25, 130), 1, cv2.LINE_AA)
        # Large preretinal boat-shaped hemorrhage
        cv2.ellipse(img, (center[0] + 40, center[1] + 60), (45, 18), 15, 0, 360, (5, 5, 70), -1)

    # Mild gaussian smooth to simulate optical camera MTF
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img

def generate_all_samples():
    """Generates sample fundus images, processes them, runs Grad-CAM and creates PDFs."""
    upload_dir = Path(Config.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ('sample_normal', 0, 'SCR-2026-0103', 'DR-2026-0003'),
        ('sample_mild', 1, 'SCR-2026-0101', 'DR-2026-0001'),
        ('sample_mod', 2, 'SCR-2026-0104', 'DR-2026-0004'),
        ('sample_severe', 3, 'SCR-2026-0102', 'DR-2026-0002'),
        ('sample_pdr', 4, 'SCR-2026-0105', 'DR-2026-0005')
    ]

    model_service = ModelService.get_instance()
    gradcam = GradCAM(model_service.model)

    for name, grade, scr_id, pat_id in samples:
        orig_path = upload_dir / f"{name}.jpg"
        proc_path = upload_dir / f"{name}_proc.jpg"
        cam_path = upload_dir / f"{name}_gradcam.jpg"
        cv_path = upload_dir / f"{name}_cv.jpg"
        pdf_path = upload_dir / f"{scr_id}_report.pdf"

        # 1. Create fundus
        fundus = create_synthetic_fundus(grade=grade)
        cv2.imwrite(str(orig_path), fundus)

        # 2. Enhance image
        enhanced = enhance_retinal_image(fundus)
        cv2.imwrite(str(proc_path), enhanced)

        # 3. CV analysis
        annotated_cv, vessels, cv_stats = analyze_retinal_structures(enhanced)
        cv2.imwrite(str(cv_path), annotated_cv)

        # 4. Grad-CAM
        tensor = prepare_tensor_for_model(enhanced, device=model_service.device)
        heatmap, _ = gradcam.generate_heatmap(tensor, target_class=grade)
        overlay, _ = gradcam.overlay_on_image(enhanced, heatmap)
        cv2.imwrite(str(cam_path), overlay)

        # 5. Generate PDF report
        screening_data = {
            'screening_id': scr_id,
            'eye_side': 'Right' if grade != 3 else 'Left',
            'original_image_path': str(orig_path),
            'processed_image_path': str(proc_path),
            'gradcam_image_path': str(cam_path),
            'quality_status': 'Pass',
            'quality_score': 94.5,
            'predicted_grade': grade,
            'confidence_score': 0.88 + (grade * 0.02),
            'referral_urgency': Config.DR_CLASSES[grade]['referral_urgency'],
            'clinical_recommendation': Config.DR_CLASSES[grade]['recommendation_en'],
            'model_mode': 'demo_calibrated',
            'class_probabilities': [0.0] * 5
        }
        screening_data['class_probabilities'][grade] = 0.88 + (grade * 0.02)
        # Distribute remaining probability
        rem = (1.0 - screening_data['class_probabilities'][grade]) / 4
        for k in range(5):
            if k != grade:
                screening_data['class_probabilities'][k] = round(rem, 3)

        patient_data = {
            'patient_id': pat_id,
            'name': f"Sample Patient ({Config.DR_CLASSES[grade]['name']})",
            'age': 55 + grade * 3,
            'gender': 'Male' if grade % 2 == 0 else 'Female',
            'mobile': '+91 98451 0000' + str(grade),
            'village': 'Shirur PHC Area',
            'district': 'Kalaburagi',
            'diabetes_history': 'Type 2',
            'duration_diabetes_years': 4.0 + (grade * 3.5),
            'blood_glucose_level': 140.0 + (grade * 25.0)
        }

        generate_screening_pdf(screening_data, patient_data, str(pdf_path))
        print(f"[Sample Created] {name}.jpg, {name}_proc.jpg, {name}_gradcam.jpg, {name}_cv.jpg, {scr_id}_report.pdf")

if __name__ == '__main__':
    generate_all_samples()
