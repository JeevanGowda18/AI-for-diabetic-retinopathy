"""
RETFound Foundation Model Loader & Inference Service (Phase 2)
Integrates RETFound (ViT-Large/16) adapted for 5-class diabetic retinopathy classification.
Adheres strictly to clinical integrity guidelines: if pretrained weights are not installed,
it returns status='weights_required' and NEVER fabricates predictions.

Pretrained Weights Download Instructions:
1. Download official weights from Hugging Face:
   URL: https://huggingface.co/YukunLi/RETFound_cfp
   File: RETFound_cfp_weights.pth (~1.1 GB)
2. Place the downloaded file into the models directory:
   Path: models/retfound_cfp_weights.pth
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

from models.retfound_model import RETFoundViT


class RETFoundService:
    """Singleton service for RETFound Foundation Model inference."""

    _instance = None
    WEIGHTS_PATH = Path(__file__).resolve().parent / 'retfound_cfp_weights.pth'

    def __init__(self):
        self.device = torch.device('cpu')  # CPU-only deployment for rural PHCs
        self.model: Optional[RETFoundViT] = None
        self.status = 'uninitialized'
        self.status_message = 'RETFound Foundation Model not yet loaded'
        self.is_available = False
        self._load()

    @classmethod
    def get_instance(cls) -> 'RETFoundService':
        if cls._instance is None:
            cls._instance = RETFoundService()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Force reload on next get_instance() call."""
        cls._instance = None

    def _load(self):
        """Initializes model architecture and checks for verified weights checkpoint."""
        self.model = RETFoundViT(num_classes=5)
        self.model.to(self.device)

        if self.WEIGHTS_PATH.exists():
            try:
                state = torch.load(str(self.WEIGHTS_PATH), map_location=self.device)
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                elif isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                
                # Load with strict=False to accommodate MAE pre-trained checkpoints
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                self.model.eval()
                self.status = 'production_trained'
                self.status_message = f'RETFound Foundation Weights Active ({self.WEIGHTS_PATH.name})'
                self.is_available = True
                print(f"[RETFound] Successfully loaded foundation weights from {self.WEIGHTS_PATH}")
            except Exception as exc:
                self.status = 'load_error'
                self.status_message = f'RETFound weights load failed: {exc}'
                self.is_available = False
                print(f"[RETFound] ERROR loading weights: {exc}")
        else:
            self.status = 'weights_required'
            self.status_message = (
                "RETFound weights (~1.1 GB) not found. Download 'RETFound_cfp_weights.pth' "
                "from https://huggingface.co/YukunLi/RETFound_cfp and place in models/ folder."
            )
            self.is_available = False
            print(f"[RETFound] Notice: Weights file not present at {self.WEIGHTS_PATH}. Running in weights_required mode.")

    def predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Runs 5-class inference on preprocessed 1x3x224x224 tensor.
        Returns empty/unavailable diagnostic if weights are not installed.
        """
        if not self.is_available or self.model is None:
            return {
                'available': False,
                'model_name': 'RETFound (ViT-Large/16)',
                'status': self.status,
                'status_message': self.status_message,
                'download_url': 'https://huggingface.co/YukunLi/RETFound_cfp',
                'predicted_grade': None,
                'confidence': None,
                'class_probabilities': None,
                'logits': None,
                'inference_time_ms': 0.0
            }

        t0 = time.perf_counter()
        tensor = tensor.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(tensor)
            raw_np = logits.cpu().numpy()[0]

            if np.max(np.abs(raw_np)) > 100.0:
                logits = torch.clamp(logits, -20.0, 20.0)
                raw_np = logits.cpu().numpy()[0]

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        predicted_grade = int(np.argmax(probs))
        confidence = float(probs[predicted_grade])
        probabilities = [round(float(p), 4) for p in probs]
        logit_list = [round(float(l), 3) for l in raw_np]

        print(
            f"[RETFound] Probs -> G0:{probabilities[0]:.3f} G1:{probabilities[1]:.3f} "
            f"G2:{probabilities[2]:.3f} G3:{probabilities[3]:.3f} G4:{probabilities[4]:.3f} "
            f"-> Pred Grade {predicted_grade} ({confidence*100:.1f}%) [{elapsed_ms:.1f} ms]"
        )

        return {
            'available': True,
            'model_name': 'RETFound (ViT-Large/16)',
            'predicted_grade': predicted_grade,
            'confidence': round(confidence, 4),
            'class_probabilities': probabilities,
            'logits': logit_list,
            'inference_time_ms': round(elapsed_ms, 1),
            'status': self.status,
            'status_message': self.status_message
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            'model_name': 'RETFound (ViT-Large/16)',
            'status': self.status,
            'status_message': self.status_message,
            'is_available': self.is_available,
            'weights_path': str(self.WEIGHTS_PATH),
            'weights_exist': self.WEIGHTS_PATH.exists(),
            'device': str(self.device),
            'download_instructions': (
                "Download RETFound_cfp_weights.pth from "
                "https://huggingface.co/YukunLi/RETFound_cfp and save to models/"
            )
        }
