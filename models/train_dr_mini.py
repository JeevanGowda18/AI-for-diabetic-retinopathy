"""
DR-Mini AI (EfficientNet-B0) Training & Calibration Script (Phase 3)
Trains the lightweight 5-class DR classifier independently on synthetic fundus data.
Saves calibrated weights to models/dr_mini_weights.pth.
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.dr_mini_model import DRMiniAI
from models.train_dr_model import create_retinal_dataset


def train_dr_mini(epochs=18, batch_size=8, lr=3e-4):
    print("=" * 70)
    print("DR-MINI AI (EFFICIENTNET-B0) INDEPENDENT TRAINING & CALIBRATION")
    print("=" * 70)

    device = torch.device('cpu')  # Ensure CPU calibration
    print(f"[Device] Using {device} for training")

    # 1. Dataset Generation
    print("[Data] Generating diverse fundus dataset for DR-Mini...")
    X_train, y_train = create_retinal_dataset(num_per_class=14, start_seed=5000)
    print(f"[Data Ready] Training set: {X_train.shape[0]} images, shape: {X_train.shape}")

    # 2. Model Initialization
    model = DRMiniAI(num_classes=5)
    model.to(device)

    # Stable weight initialization on classifier head
    for m in model.backbone.classifier.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    num_samples = X_train.shape[0]
    indices = list(range(num_samples))
    start_time = time.time()

    print(f"\n[Training] Running {epochs} epochs (batch size: {batch_size})...")
    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        correct = 0
        total = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            b_X = X_train[batch_indices].to(device)
            b_y = y_train[batch_indices].to(device)

            optimizer.zero_grad()
            outputs = model(b_X)
            loss = criterion(outputs, b_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_indices)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == b_y).sum().item()
            total += len(batch_indices)

        scheduler.step()
        avg_loss = epoch_loss / total
        acc = (correct / total) * 100.0
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Accuracy: {acc:5.1f}% | Elapsed: {elapsed:.1f}s")

    model.eval()

    # 3. Validation on Samples
    print("\n[Logits Verification] Verifying DR-Mini evaluation mode behavior...")
    with torch.no_grad():
        test_sample = X_train[:5].to(device)
        eval_logits = model(test_sample)
        eval_probs = F.softmax(eval_logits, dim=1).cpu().numpy()
        for i in range(5):
            print(f"  Sample {i} (True Grade: {y_train[i]}): Logits: {np.round(eval_logits[i].cpu().numpy(), 2)} -> Probs: {np.round(eval_probs[i], 3)}")

    # 4. Save Checkpoint
    weights_path = BASE_DIR / 'models' / 'dr_mini_weights.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"\n[Success] DR-Mini AI weights saved successfully to:\n  - {weights_path}")
    return model


if __name__ == '__main__':
    train_dr_mini()
