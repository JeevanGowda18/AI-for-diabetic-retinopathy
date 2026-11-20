"""
DR-Mini AI Model Loader & Inference Service
EfficientNet-B0 based DR classifier — independent from the ResNet-50 baseline model.
Singleton pattern for efficient reuse across Flask requests.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from models.dr_mini_model import DRMiniAI


class DRMiniService:
    """Singleton service for DR-Mini AI (EfficientNet-B0) inference."""

    _instance = None

    # Canonical weight path
    WEIGHTS_PATH = Path(__file__).resolve().parent / 'dr_mini_weights.pth'

    def __init__(self):
        self.device = torch.device('cpu')   # CPU-only deployment (8 GB RAM PHC laptop)
        self.model: DRMiniAI | None = None
        self.status = 'uninitialized'
        self.status_message = 'DR-Mini AI not yet loaded'
        self.is_available = False
        self._load()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> 'DRMiniService':
        if cls._instance is None:
            cls._instance = DRMiniService()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Force reload on next get_instance() call (useful in tests)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        self.model = DRMiniAI(num_classes=5)
        self.model.to(self.device)

        if self.WEIGHTS_PATH.exists():
            try:
                state = torch.load(str(self.WEIGHTS_PATH), map_location=self.device)
                # Support checkpoints that wrap state_dict inside a dict
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                self.model.load_state_dict(state)
                self.model.eval()
                self.status = 'trained'
                self.status_message = f'DR-Mini AI weights loaded ({self.WEIGHTS_PATH.name})'
                self.is_available = True
                print(f'[DR-Mini AI] Weights loaded from {self.WEIGHTS_PATH}')
            except Exception as exc:
                self.status = 'load_error'
                self.status_message = f'DR-Mini AI weight load failed: {exc}'
                self.is_available = False
                print(f'[DR-Mini AI] ERROR loading weights: {exc}')
        else:
            self.status = 'weights_missing'
            self.status_message = (
                f'DR-Mini AI weights not found at {self.WEIGHTS_PATH}. '
                'Run: python models/train_dr_mini.py'
            )
            self.is_available = False
            print(f'[DR-Mini AI] Weights not found at {self.WEIGHTS_PATH}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, tensor: torch.Tensor) -> dict:
        """
        Run 5-class inference on a pre-processed 1×3×224×224 tensor.

        Returns dict with keys:
            available, model_name, predicted_grade, confidence,
            class_probabilities, logits, inference_time_ms, status, status_message
        """
        if not self.is_available or self.model is None:
            return {
                'available': False,
                'model_name': 'DR-Mini AI (EfficientNet-B0)',
                'status': self.status,
                'status_message': self.status_message,
                'predicted_grade': None,
                'confidence': None,
                'class_probabilities': None,
                'logits': None,
                'inference_time_ms': 0.0,
            }

        t0 = time.perf_counter()
        tensor = tensor.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(tensor)

            raw_np = logits.cpu().numpy()[0]
            # Guard against logit explosion (untrained / corrupted weights)
            if np.max(np.abs(raw_np)) > 100.0:
                print(f'[DR-Mini AI] Logit explosion detected: {raw_np}. Clamping.')
                logits = torch.clamp(logits, -20.0, 20.0)
                raw_np = logits.cpu().numpy()[0]

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        predicted_grade = int(np.argmax(probs))
        confidence = float(probs[predicted_grade])
        probabilities = [round(float(p), 4) for p in probs]
        logit_list = [round(float(l), 3) for l in raw_np]

        # Always print 5-class distribution for debugging
        print(
            f'[DR-Mini AI] Probs → '
            f'G0:{probabilities[0]:.3f}  G1:{probabilities[1]:.3f}  '
            f'G2:{probabilities[2]:.3f}  G3:{probabilities[3]:.3f}  '
            f'G4:{probabilities[4]:.3f}  → Pred Grade {predicted_grade} '
            f'({confidence*100:.1f}%)  [{elapsed_ms:.1f} ms]'
        )

        return {
            'available': True,
            'model_name': 'DR-Mini AI (EfficientNet-B0)',
            'predicted_grade': predicted_grade,
            'confidence': round(confidence, 4),
            'class_probabilities': probabilities,
            'logits': logit_list,
            'inference_time_ms': round(elapsed_ms, 1),
            'status': self.status,
            'status_message': self.status_message,
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        return {
            'model_name': 'DR-Mini AI (EfficientNet-B0)',
            'status': self.status,
            'status_message': self.status_message,
            'is_available': self.is_available,
            'weights_path': str(self.WEIGHTS_PATH),
            'weights_exist': self.WEIGHTS_PATH.exists(),
            'device': str(self.device),
        }
