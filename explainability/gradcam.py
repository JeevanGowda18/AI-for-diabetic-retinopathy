"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module
Generates visual explanations for Diabetic Retinopathy deep learning classifications.
Highlights discriminative retinal regions that guided the model's prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Initializes Grad-CAM on the model.
        Args:
            model: PyTorch DRResNet50 model instance
            target_layer: The target convolutional layer (defaults to layer4[-1])
        """
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layer = self.model.get_target_layer_for_cam()
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._fwd_handle = None
        self._bwd_handle = None
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.remove_hooks()
        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Safely removes registered PyTorch forward and backward hooks."""
        if self._fwd_handle is not None:
            try:
                self._fwd_handle.remove()
            except Exception:
                pass
            self._fwd_handle = None
        if self._bwd_handle is not None:
            try:
                self._bwd_handle.remove()
            except Exception:
                pass
            self._bwd_handle = None

    def __del__(self):
        self.remove_hooks()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM activation map for the given input tensor.
        Args:
            input_tensor: 1x3xHxW normalized PyTorch tensor
            target_class: int (0 to 4), class to explain. If None, uses top predicted class.
        Returns:
            heatmap_norm: 2D numpy array [0.0, 1.0]
            target_class: int of class explained
        """
        # Ensure gradients are enabled for Grad-CAM computation
        input_tensor.requires_grad = True
        self.model.zero_grad()

        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # Target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: [1, C, H, W]
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0] # [C, H, W]

        # Weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Channel-wise sum and ReLU
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU

        # Normalize between 0 and 1
        max_val = float(np.max(heatmap))
        if max_val > 1e-7:
            heatmap = heatmap / max_val
            self.last_heatmap_valid = True
        else:
            heatmap = np.zeros_like(heatmap)
            self.last_heatmap_valid = False

        return heatmap, target_class

    def overlay_on_image(self, original_bgr, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
        """
        Resizes heatmap to match original image and overlays it with alpha blending.
        Returns:
            overlay_bgr: image with color overlay
            heatmap_bgr: standalone heatmap visualization
        """
        h, w = original_bgr.shape[:2]
        
        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Convert [0, 1] to uint8 [0, 255]
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap (JET: Red = high activation, Blue = low activation)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Alpha blend overlay
        overlay_bgr = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        
        return overlay_bgr, heatmap_color

    def analyze_activated_regions(self, heatmap, image_shape):
        """
        Analyzes the spatial distribution of the highest activation zones.
        Identifies quadrants (Superior-Temporal, Superior-Nasal, Inferior-Temporal, Inferior-Nasal).
        """
        h, w = heatmap.shape
        h_mid, w_mid = h // 2, w // 2
        
        quadrant_scores = {
            'Superior-Nasal': float(np.mean(heatmap[0:h_mid, 0:w_mid])),
            'Superior-Temporal': float(np.mean(heatmap[0:h_mid, w_mid:w])),
            'Inferior-Nasal': float(np.mean(heatmap[h_mid:h, 0:w_mid])),
            'Inferior-Temporal': float(np.mean(heatmap[h_mid:h, w_mid:w]))
        }
        
        dominant_quadrant = max(quadrant_scores, key=quadrant_scores.get)
        peak_intensity = float(np.max(heatmap))
        is_valid = peak_intensity > 1e-7

        if not is_valid:
            return {
                'dominant_quadrant': 'Indeterminate',
                'quadrant_scores': {
                    'Superior-Nasal': 0.0,
                    'Superior-Temporal': 0.0,
                    'Inferior-Nasal': 0.0,
                    'Inferior-Temporal': 0.0
                },
                'is_central': False,
                'is_valid': False,
                'text_explanation': 'Grad-CAM did not detect discriminative convolutional saliency (flat activation map). Visual explanation unavailable.'
            }

        # Center proximity (Macular / Peripapillary region)
        center_region = heatmap[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        center_mean = float(np.mean(center_region)) if center_region.size > 0 else 0.0

        is_central = center_mean > (peak_intensity * 0.45)

        explanation = (
            f"The neural network focused predominantly on the {dominant_quadrant} quadrant "
            f"(relative activation: {quadrant_scores[dominant_quadrant]:.2f}). "
        )
        if is_central:
            explanation += "Significant attention is directed toward the central retinal/macular arcade region, where microvascular anomalies and exudates commonly accumulate in diabetic retinopathy."
        else:
            explanation += "Primary attention is observed along the peripheral vascular branches."

        return {
            'dominant_quadrant': dominant_quadrant,
            'quadrant_scores': quadrant_scores,
            'is_central': is_central,
            'text_explanation': explanation
        }
