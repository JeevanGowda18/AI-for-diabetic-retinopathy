"""
Hybrid Prediction Engine for DR Vision AI (Phase 4 & F-01/F-04)
Integrates and fuses predictions across:
1. Primary Clinical Model (ResNet-50 / DRClassifier)
2. DR-Mini AI (Lightweight EfficientNet-B0)
3. RETFound Retinal Foundation Model (ViT-Large/16)

Applies probabilistic fusion, disagreement resolution, and Top-2 margin quality gating.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch

from models.model_loader import ModelService
from models.dr_mini_loader import DRMiniService
from models.retfound_loader import RETFoundService
from preprocessing.image_quality import assess_prediction_quality_gate
from config import Config


class HybridPredictionEngine:
    """Coordinates multi-model diabetic retinopathy inference and consensus fusion."""

    _instance = None

    def __init__(self):
        self.primary_service = ModelService.get_instance()
        self.dr_mini_service = DRMiniService.get_instance()
        self.retfound_service = RETFoundService.get_instance()

    @classmethod
    def get_instance(cls) -> 'HybridPredictionEngine':
        if cls._instance is None:
            cls._instance = HybridPredictionEngine()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def predict(self, tensor: torch.Tensor, enhanced_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Runs inference across available models and performs probabilistic consensus fusion.

        Args:
            tensor: 1x3x224x224 normalized PyTorch tensor
            enhanced_bgr: Optional preprocessed image for auxiliary checks

        Returns:
            dict containing individual model outputs, hybrid fused diagnosis,
            agreement status, and Top-2 margin quality gate evaluation.
        """
        # 1. Primary Model (ResNet-50 / DRClassifier)
        if not self.primary_service.is_available or self.primary_service.model is None:
            raise RuntimeError(
                "Primary model weights are not loaded. Cannot run inference without valid weights."
            )
        primary_res = self.primary_service.predict(tensor)

        # 2. DR-Mini AI (EfficientNet-B0)
        dr_mini_res = self.dr_mini_service.predict(tensor)

        # 3. RETFound Foundation Model (ViT-Large/16)
        retfound_res = self.retfound_service.predict(tensor)

        # --- Fusion Strategy ---
        # Case A: Both RETFound and DR-Mini AI are genuinely available
        if retfound_res.get('available') and dr_mini_res.get('available'):
            r_probs = np.array(retfound_res['class_probabilities'], dtype=float)
            m_probs = np.array(dr_mini_res['class_probabilities'], dtype=float)
            fused_probs = (0.5 * r_probs + 0.5 * m_probs).tolist()

            fused_grade = int(np.argmax(fused_probs))
            fused_conf = float(fused_probs[fused_grade])
            models_agree = (retfound_res['predicted_grade'] == dr_mini_res['predicted_grade'])

            if models_agree:
                agreement_status = "Consensus: Both Foundation & Specialist Models Agree"
            else:
                agreement_status = (
                    f"Discrepancy: RETFound (Grade {retfound_res['predicted_grade']}) vs "
                    f"DR-Mini AI (Grade {dr_mini_res['predicted_grade']}) - Specialist Review Recommended"
                )
            fusion_method = "dual_model_average (RETFound + DR-Mini)"

        # Case B: DR-Mini AI is available, RETFound weights pending download
        elif dr_mini_res.get('available'):
            # Combine DR-Mini AI with Primary DRClassifier for reliable two-model validation
            p_probs = np.array(primary_res['probabilities'], dtype=float)
            m_probs = np.array(dr_mini_res['class_probabilities'], dtype=float)
            fused_probs = (0.5 * p_probs + 0.5 * m_probs).tolist()

            fused_grade = int(np.argmax(fused_probs))
            fused_conf = float(fused_probs[fused_grade])
            models_agree = (primary_res['predicted_grade'] == dr_mini_res['predicted_grade'])

            if models_agree:
                agreement_status = "Consensus: Primary ResNet-50 & DR-Mini AI Agree"
            else:
                agreement_status = (
                    f"Discrepancy: Primary ResNet-50 (Grade {primary_res['predicted_grade']}) vs "
                    f"DR-Mini AI (Grade {dr_mini_res['predicted_grade']}) - Specialist Review Recommended"
                )
            fusion_method = "hybrid_validated (Primary ResNet-50 + DR-Mini AI; RETFound weights pending)"

        # Case C: Primary Model fallback
        else:
            fused_probs = primary_res['probabilities']
            fused_grade = primary_res['predicted_grade']
            fused_conf = primary_res['confidence']
            models_agree = True
            agreement_status = "Single Model Active (Primary ResNet-50 / DRClassifier)"
            fusion_method = "primary_only"

        # Apply Top-2 Margin Quality Gate (F-04)
        gate_eval = assess_prediction_quality_gate(
            fused_probs,
            min_confidence=getattr(Config, 'CONFIDENCE_THRESHOLD', 0.50),
            min_top2_margin=getattr(Config, 'TOP2_MARGIN_THRESHOLD', 0.15)
        )

        grade_meta = Config.DR_CLASSES.get(fused_grade, Config.DR_CLASSES[0])
        is_referable = 1 if fused_grade >= 2 else 0

        hybrid_summary = {
            'predicted_grade': fused_grade,
            'predicted_class_name': grade_meta['name'],
            'confidence': round(fused_conf, 4),
            'class_probabilities': [round(float(p), 4) for p in fused_probs],
            'is_referable': is_referable,
            'models_agree': models_agree,
            'agreement_status': agreement_status,
            'fusion_method': fusion_method,
            'quality_gate': gate_eval,
            'quality_gate_passed': 1 if gate_eval['gate_passed'] else 0,
            'review_required': gate_eval['review_required'] or not models_agree,
            'referral_urgency': grade_meta['referral_urgency'],
            'clinical_recommendation': grade_meta['recommendation_en']
        }

        return {
            'primary_model': primary_res,
            'dr_mini': dr_mini_res,
            'retfound': retfound_res,
            'hybrid_result': hybrid_summary
        }
