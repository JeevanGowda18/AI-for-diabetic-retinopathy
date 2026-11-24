# SYSTEM TECHNICAL SPECIFICATION: DR VISION AI ARCHITECTURE & END-TO-END PIPELINE

**Project Title:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Problem Statement ID:** 26038  
**Application Name:** DR Vision AI  
**Deployment Target:** Rural Primary Health Centres (PHCs) & Community Health Centres (CHCs)  

---

## 1. System Architecture Overview

DR Vision AI is an end-to-end clinical triage decision support system architected specifically for resource-constrained primary care clinics. The software stack integrates computer vision, deep convolutional neural networks, model explainability, multilingual localization, and strict clinical gating into a unified, responsive web platform:

```
[ Retinal Image Capture (Mobile / Desktop Fundus Camera) ]
                           │
                           ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 1: Pre-Inference Automated Guardrail  │
     │  - Focus / Blur: Laplacian Variance >= 60.0  │
     │  - Illumination: Mean Intensity [40, 215]    │
     │  - Field of View (FOV): Area Mask >= 20.0%   │
     └──────────────────────┬───────────────────────┘
                            │ Pass
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 2: Standardized Retinal Enhancement   │
     │  - Gemini Vision AI / OpenCV Contour Crop    │
     │  - Circular FOV Masking & Square Letterbox   │
     │  - LAB Color Space CLAHE (L-Channel eq.)     │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 3: Multi-Model Hybrid Inference       │
     │  - Model 1: ResNet-50 (Macroscopic Features) │
     │  - Model 2: DR-Mini AI (Micro-Lesions)       │
     │  - Model 3: RETFound ViT-Large/16 (Wrapper)  │
     │  - Softmax Probability Fusion Engine         │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 4: Confidence & Quality Gate          │
     │  - Model Confidence >= 0.50                  │
     │  - Top-2 Margin: (P_top1 - P_top2) >= 0.15   │
     │  - If Failed: Flag for Mandatory Doctor Rev. │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 5: Explainable AI & Feature Saliency  │
     │  - PyTorch Hook-Based Grad-CAM Heatmap       │
     │  - Thermal JET Overlay on Original Fundus    │
     │  - 4-Quadrant Saliency Breakdown (ST,SN,IT,IN│
     │  - CV Lesion Markers (MA, Hemorrhage, Exudate│
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 6: Clinical Sign-Off & Verification   │
     │  - Doctor Review Queue (Verify/Modify/Reject)│
     │  - Immutable Audit Logging in SQLite DB      │
     │  - Independent OD (Right) & OS (Left) Record │
     └──────────────────────┬───────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Stage 7: Report Generation & Patient Triage │
     │  - ReportLab PDF: Digital (Color) & Print    │
     │  - Trilingual Explanations: EN, KN, HI       │
     │  - Multilingual Speech Synthesis (TTS)       │
     │  - Secure SMTP Dispatch / PHC Local Audit    │
     └──────────────────────────────────────────────┘
```

---

## 2. Technical Component Breakdown

### A. Preprocessing & Quality Assurance (`preprocessing/`)
1. **`image_quality.py`:**
   - Focus calculation using the variance of the 2D Laplacian operator:
     $$\text{Focus Score} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum (L(x, y) - \mu_L)^2$$
   - Luminance and glare check across image quadrants.
   - Retinal mask segmentation ensuring the circular FOV occupies at least 20% of the camera frame.
2. **`gemini_vision_crop.py` (Strictly Preserved):**
   - Connects to Google Gemini Vision AI (`gemini-1.5-flash`) via the official API to detect the optical circle bounding box.
   - If the API key is not present or internet is down, automatically invokes the certified OpenCV adaptive contour circle detector (`detect_and_crop_fundus_circle`) with zero runtime errors.
3. **`image_enhancement.py`:**
   - Converts BGR to CIE $L^*a^*b^*$ color space.
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) with `clipLimit=2.5, tileGridSize=(8, 8)` strictly on the Lightness ($L^*$) channel, preserving physiological pigmentation while accentuating capillary microaneurysms and hard exudates.
   - Circular mask bounding box extraction and square aspect-ratio padding with black pixels.

### B. Machine Learning Architecture (`models/`)
1. **`dr_model.py` (`DRResNet50` / `DRClassifier`):**
   - ResNet-50 backbone with fine-tuned bottleneck stages and calibrated 5-class linear classifier head.
2. **`dr_mini_model.py` (`DRMiniAI`):**
   - EfficientNet-B0 inverted bottleneck convolutional network optimized for real-time edge CPU triage.
3. **`retfound_model.py` & `retfound_loader.py`:**
   - ViT-Large/16 Masked Autoencoder architecture wrapper for the RETFound retinal foundation model. Operating in transparent `weights_required` mode.
4. **`hybrid_prediction.py`:**
   - Singleton service executing simultaneous inference across available models.
   - Evaluates inter-model consensus and passes results to the Top-2 Margin Quality Gate.

### C. Explainability & Clinical NLP (`explainability/`)
1. **`gradcam.py`:**
   - Dynamically registers forward and backward gradient hooks on convolutional layers.
   - Computes channel-weighted activation maps and normalizes activations using min-max scaling.
   - Blends JET colormap with 60% opacity over the enhanced fundus image.
2. **`clinical_nlp.py`:**
   - Divides the retinal coordinate space into four quadrants: Superior-Temporal (ST), Superior-Nasal (SN), Inferior-Temporal (IT), and Inferior-Nasal (IN), plus central macular zone.
   - Translates activation percentages into plain-language clinical notes for rural community health officers.

### D. Relational Database Schema (`database/`)
- Engine: SQLite3 (`instance/dr_vision.db`).
- **`patients` table:** `id`, `patient_id` (`PAT-YYYY-XXXX`), `name`, `age`, `gender`, `mobile`, `village`, `district`, `diabetes_history`, `duration_diabetes_years`, `blood_glucose_level`, `created_at`.
- **`screenings` table:** `screening_id` (`SCR-YYYY-XXXX`), `patient_id`, `eye_side` (`Right` / `Left`), `image_paths`, `quality_score`, `predicted_grade`, `confidence_score`, `class_probabilities`, `verification_status` (`Pending`, `Verified`, `Modified`, `Rejected`), `verified_by`, `doctor_notes`, `doctor_assigned_grade`, `dr_mini_grade`, `retfound_grade`.
- **`audit_logs` table:** Records administrative, screening, email, and clinical override events with timestamps and user attribution.

### E. Reporting & Multilingual Tele-Ophthalmology (`reports/`, `translations/`)
1. **`pdf_generator.py`:**
   - Powered by ReportLab Flowables. Generates two A4 formats:
     - **Digital Soft-Copy:** Full-color, high-resolution diagnostic report with Grad-CAM overlays, probability charts, and referral guidance.
     - **Printable Hard-Copy:** High-contrast, ink-saving monochrome format with clinical checkboxes and doctor signature boxes for paper filing.
2. **`routes/tts.py` & `translations.py`:**
   - Full localization into English, Kannada (`kn`), and Hindi (`hi`).
   - Server-side `gTTS` speech generation with client-side browser Web Speech API fallback.
