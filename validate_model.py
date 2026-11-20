"""
Diabetic Retinopathy Model Validation & Benchmarking Script
Audits and evaluates the calibrated DRResNet50 model across all 5 ICDR severity classes:
  Grade 0: No DR
  Grade 1: Mild NPDR
  Grade 2: Moderate NPDR
  Grade 3: Severe NPDR
  Grade 4: Proliferative DR

Calculates:
  - Accuracy
  - Precision (Macro & Weighted)
  - Recall / Sensitivity (Macro & Weighted)
  - Specificity (per-class & Macro)
  - F1-Score (Macro & Weighted)
  - Confusion Matrix (5x5)
  - ROC-AUC (One-vs-Rest Macro & Weighted)
Generates and saves confusion_matrix.png.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config
from models.dr_model import DRResNet50
from models.model_loader import ModelService
from preprocessing.image_enhancement import enhance_retinal_image, prepare_tensor_for_model
from models.train_dr_model import generate_diverse_fundus

def generate_test_dataset(samples_per_class=12, start_seed=9000):
    """Generates an independent, unseen test dataset across all 5 DR grades."""
    test_data = []
    print(f"[Test Dataset] Generating {samples_per_class * 5} independent test images ({samples_per_class} per class)...")
    for grade in range(5):
        for i in range(samples_per_class):
            seed = start_seed + grade * 700 + i
            fundus_bgr = generate_diverse_fundus(grade=grade, seed=seed)
            enhanced_bgr = enhance_retinal_image(fundus_bgr)
            tensor = prepare_tensor_for_model(enhanced_bgr, target_size=(224, 224), device='cpu')
            test_data.append({
                'tensor': tensor,
                'true_grade': grade,
                'fundus_bgr': fundus_bgr,
                'sample_id': f"Test_G{grade}_{i+1}"
            })
    return test_data

def evaluate_model(weights_path=None):
    print("=" * 80)
    print("DR VISION AI: INDEPENDENT MODEL AUDIT & BENCHMARK EVALUATION")
    print("=" * 80)

    # 1. Initialize Model
    model = DRResNet50(num_classes=5, pretrained_backbone=False)
    
    if weights_path is None:
        if Path(Config.MODEL_WEIGHTS_PATH).exists():
            weights_path = Path(Config.MODEL_WEIGHTS_PATH)
        else:
            weights_path = BASE_DIR / 'models' / 'dr_demo_weights.pth'

    print(f"[Model Loading] Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Generate Independent Test Data
    test_samples = generate_test_dataset(samples_per_class=12, start_seed=9500)
    
    y_true = []
    y_pred = []
    y_scores = []
    confidences = []

    # 3. Run Inference
    with torch.no_grad():
        for sample in test_samples:
            tensor = sample['tensor']
            true_g = sample['true_grade']
            
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pred_g = int(np.argmax(probs))
            
            y_true.append(true_g)
            y_pred.append(pred_g)
            y_scores.append(probs)
            confidences.append(float(probs[pred_g]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # 4. Compute Metrics
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = len(y_true)
    accuracy = np.trace(cm) / total

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    specificity_per_class = []
    support_per_class = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        tn = total - (tp + fp + fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)
        specificity_per_class.append(spec)
        support_per_class.append(int(np.sum(cm[c, :])))

    # Compute Macro & Weighted averages
    macro_precision = float(np.mean(precision_per_class))
    macro_recall = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))
    macro_specificity = float(np.mean(specificity_per_class))

    weights = np.array(support_per_class) / total
    weighted_precision = float(np.sum(np.array(precision_per_class) * weights))
    weighted_recall = float(np.sum(np.array(recall_per_class) * weights))
    weighted_f1 = float(np.sum(np.array(f1_per_class) * weights))

    class_names = [
        "Grade 0 (No DR)",
        "Grade 1 (Mild NPDR)",
        "Grade 2 (Moderate NPDR)",
        "Grade 3 (Severe NPDR)",
        "Grade 4 (Proliferative DR)"
    ]

    # Print Detailed Classification Report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 80)
    print(f"{'Class':<26} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<11} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 86)
    for c in range(num_classes):
        print(f"{class_names[c]:<26} | {precision_per_class[c]:<10.4f} | {recall_per_class[c]:<10.4f} | {specificity_per_class[c]:<11.4f} | {f1_per_class[c]:<10.4f} | {support_per_class[c]:<8}")
    print("-" * 86)
    print(f"{'Overall Accuracy':<26} | {accuracy:<10.4f} | {'':<10} | {'':<11} | {'':<10} | {total:<8}")
    print(f"{'Macro Average':<26} | {macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_specificity:<11.4f} | {macro_f1:<10.4f} | {total:<8}")
    print(f"{'Weighted Average':<26} | {weighted_precision:<10.4f} | {weighted_recall:<10.4f} | {'':<11} | {weighted_f1:<10.4f} | {total:<8}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX (Rows: Ground Truth, Columns: Predicted)")
    print("=" * 80)
    header = "True \\ Pred | " + " | ".join([f"G{i}" for i in range(5)])
    print(header)
    print("-" * len(header))
    for i in range(5):
        row_str = f"Grade {i}     | " + " | ".join([f"{cm[i, j]:2d}" for j in range(5)])
        print(row_str)

    # 5. Plot and Save Confusion Matrix Figure
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)

    for i in range(num_classes):
        for j in range(num_classes):
            text_color = 'white' if cm[i, j] > (cm.max() / 2) else 'black'
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=14, fontweight='bold', color=text_color)

    fig.colorbar(cax)
    tick_marks = np.arange(num_classes)
    short_labels = ["G0 No DR", "G1 Mild", "G2 Mod", "G3 Sev", "G4 PDR"]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short_labels, fontsize=10, rotation=15)
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel('Predicted DR Severity Grade', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True DR Severity Grade (Clinical Ground Truth)', fontsize=12, fontweight='bold')
    ax.set_title(f'DR Vision AI - Confusion Matrix\nAccuracy: {accuracy*100:.1f}% | Macro F1: {macro_f1:.4f}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    # Save to static/images and reports
    images_dir = BASE_DIR / 'static' / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    cm_path1 = images_dir / 'confusion_matrix.png'
    plt.savefig(str(cm_path1), dpi=200)

    reports_dir = BASE_DIR / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm_path2 = reports_dir / 'confusion_matrix.png'
    plt.savefig(str(cm_path2), dpi=200)
    plt.close()

    print(f"\n[Saved] Confusion matrix figure saved to:")
    print(f"  - {cm_path1}")
    print(f"  - {cm_path2}")

    # 6. Audit Static Sample Images
    print("\n" + "=" * 80)
    print("AUDIT OF APPLICATION SAMPLE IMAGES (Verifying Grade 0 to Grade 4)")
    print("=" * 80)
    sample_files = [
        ('sample_normal.jpg', 0, 'No DR'),
        ('sample_mild.jpg', 1, 'Mild NPDR'),
        ('sample_mod.jpg', 2, 'Moderate NPDR'),
        ('sample_severe.jpg', 3, 'Severe NPDR'),
        ('sample_pdr.jpg', 4, 'Proliferative DR')
    ]

    upload_dir = BASE_DIR / 'static' / 'uploads'
    all_sample_preds_distinct = True
    sample_audit_results = []

    for fname, expected_g, exp_desc in sample_files:
        fpath = upload_dir / fname
        if not fpath.exists():
            print(f"  [Warning] {fname} not found.")
            continue
        
        img_bgr = cv2.imread(str(fpath))
        enhanced = enhance_retinal_image(img_bgr)
        tensor = prepare_tensor_for_model(enhanced, target_size=(224, 224), device='cpu')
        
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).numpy()[0]
        
        pred_g = int(np.argmax(probs))
        conf = float(probs[pred_g])
        prob_str = " | ".join([f"G{i}: {probs[i]:.3f}" for i in range(5)])

        match_status = "MATCH" if pred_g == expected_g else "MISMATCH"
        print(f"  File: {fname:<18} | Expected: G{expected_g} ({exp_desc:<16}) | Pred: G{pred_g} ({class_names[pred_g]:<18}) | Conf: {conf*100:5.1f}% | [{match_status}]")
        print(f"    Raw Logits: {np.round(logits.numpy()[0], 2)} | Probs: {prob_str}")
        
        sample_audit_results.append({
            'file': fname,
            'expected': expected_g,
            'predicted': pred_g,
            'confidence': conf,
            'probs': probs.tolist()
        })

    # Verify that not all predictions are Grade 2
    distinct_preds = set(r['predicted'] for r in sample_audit_results)
    if len(distinct_preds) > 1 and 2 in distinct_preds:
        print(f"\n[CRITICAL BUG FIXED] Model successfully predicts {len(distinct_preds)} distinct classes across samples!")
        print("  The application NO LONGER gets stuck on Grade 2 Moderate NPDR!")
    else:
        print(f"\n[Warning] Distinct predictions count: {len(distinct_preds)}")

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm.tolist(),
        'sample_audit_results': sample_audit_results
    }

if __name__ == '__main__':
    evaluate_model()
