"""Quick inference audit - run to check current model status."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from models.dr_model import DRResNet50
from preprocessing.image_enhancement import enhance_retinal_image, prepare_tensor_for_model
from config import Config

def run_audit():
    print("=" * 70)
    print("PHASE 1 AUDIT: CURRENT MODEL INFERENCE PIPELINE")
    print("=" * 70)

    # 1. Check weight files
    prod = Config.MODEL_WEIGHTS_PATH
    demo = prod.parent / 'dr_demo_weights.pth'
    print(f"\n[Weights] Production path: {prod} | Exists: {prod.exists()}")
    print(f"[Weights] Demo path:       {demo}  | Exists: {demo.exists()}")

    # 2. Load model
    model = DRResNet50(num_classes=5, pretrained_backbone=False)
    weights_path = prod if prod.exists() else demo if demo.exists() else None
    if weights_path:
        try:
            sd = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(sd)
            print(f"[Model Load] SUCCESS from {weights_path.name}")
        except Exception as e:
            print(f"[Model Load] FAILED: {e}")
    else:
        print("[Model Load] NO WEIGHTS FOUND - random initialization")
    model.eval()

    # 3. Inspect final FC layer
    fc5 = model.backbone.fc[5]
    print(f"\n[FC Layer] Weight shape: {fc5.weight.shape}")
    print(f"[FC Layer] Weight mean: {fc5.weight.mean().item():.4f}, std: {fc5.weight.std().item():.4f}")
    print(f"[FC Layer] Bias: {fc5.bias.data.numpy().round(3)}")

    # 4. Run on each sample
    print("\n[Inference Check] Running on sample images:")
    samples = [
        ('sample_normal.jpg', 0, 'No DR'),
        ('sample_mild.jpg',   1, 'Mild NPDR'),
        ('sample_mod.jpg',    2, 'Moderate NPDR'),
        ('sample_severe.jpg', 3, 'Severe NPDR'),
        ('sample_pdr.jpg',    4, 'Proliferative DR'),
    ]
    upload_dir = Config.UPLOAD_FOLDER

    for fname, expected, label in samples:
        fpath = upload_dir / fname
        if not fpath.exists():
            print(f"  {fname}: NOT FOUND (skipping)")
            continue
        img = cv2.imread(str(fpath))
        enh = enhance_retinal_image(img)
        t = prepare_tensor_for_model(enh)
        with torch.no_grad():
            logits = model(t)
            probs = F.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        match = "MATCH" if pred == expected else "MISMATCH"
        logit_str = "[" + ", ".join(f"{v:.2f}" for v in logits.numpy()[0]) + "]"
        prob_str  = "[" + ", ".join(f"{v:.3f}" for v in probs) + "]"
        print(f"  {fname:<18} Expected G{expected} | Pred G{pred} {match}")
        print(f"    Logits: {logit_str}")
        print(f"    Probs:  {prob_str}")

if __name__ == '__main__':
    run_audit()
