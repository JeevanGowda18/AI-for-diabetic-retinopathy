# Explainable AI for Diabetic Retinopathy Screening in Rural India
### Application Name: **DR Vision AI** (Rural Tele-Ophthalmology Decision Support)

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Framework](https://img.shields.io/badge/Flask-3.1-teal)
![Deep Learning](https://img.shields.io/badge/PyTorch-ResNet50-orange)
![Computer Vision](https://img.shields.io/badge/OpenCV-Image%20Quality%20%26%20CLAHE-green)
![Explainability](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-red)
![PDF Reports](https://img.shields.io/badge/Reports-ReportLab-navy)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20%E0%B2%95%E0%B2%A8%E0%B3%8D%E0%B2%A8%E0%B2%A1%20%7C%20%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80-purple)

---

## Executive Summary & Rural Healthcare Context

In India, over **101 million people live with diabetes**, with an additional 136 million identified as pre-diabetic. Over **70% of India's population resides in rural villages and agricultural taluks**, yet **over 80% of all certified ophthalmologists and retinal surgeons practice in metropolitan tier-1/tier-2 hospitals**.

Diabetic Retinopathy (DR) is a microvascular complication of chronic diabetes that damages retinal capillaries, causing microaneurysms, hemorrhages, hard lipid exudates, and abnormal neovascularization. Because early-stage DR (Grades 1 and 2) is largely asymptomatic, rural patients rarely seek hospital eye care until severe vision loss or vitreous detachment occurs.

**DR Vision AI** was designed to bridge this clinical gap. Tailored specifically for **Primary Health Centres (PHCs)**, Community Health Centres (CHCs), and rural sub-centres, this platform equips Community Health Officers (CHOs), staff nurses, and medical officers with point-of-care tele-ophthalmology triage support.

---

## Key Clinical & Technical Features

### 1. Automated Pre-Inference Quality Guardrails
Retinal images captured with handheld or desktop cameras in rural clinics often suffer from focal blur, flash glare, or eyelid obstruction. Rather than producing unreliable diagnoses, DR Vision AI subjects every image to rigorous automated validation:
* **Focus & Blur Detection:** Measures Laplacian variance inside the retinal field. Flags blurry acquisitions below focal threshold.
* **Illumination & Glare Check:** Evaluates mean luminance and quadrant uniformity. Identifies overexposed flash washouts and dark underexposed captures.
* **Field of View (FOV) Extraction:** Segments the circular retinal mask and measures the usable area percentage relative to the image frame.

### 2. Standardized Medical Preprocessing
* **LAB Color Space CLAHE:** Applies Contrast Limited Adaptive Histogram Equalization exclusively on the Lightness ($L$) channel. Sharpens microaneurysms and intraretinal hemorrhages without altering physiological color balance.
* **Circular Border Cropping:** Removes uninformative peripheral black borders while strictly preserving aspect ratio.
* **Vascular Edge Clarification:** Mild unsharp masking calibrated to avoid artificial lesion artifacts.

## The Eight Core Features of DR Vision AI

### F-01: Five-Stage Diabetic Retinopathy Classification
Adapted deep learning architecture classifying retinal fundus images according to the International Clinical Diabetic Retinopathy (ICDR) scale:
* **Grade 0: No DR** &mdash; Healthy retina baseline; annual screening recommended.
* **Grade 1: Mild NPDR** &mdash; Isolated microaneurysms; 6-12 month follow-up.
* **Grade 2: Moderate NPDR** &mdash; Microaneurysms, blot hemorrhages, hard exudates; specialist referral within 2-3 months.
* **Grade 3: Severe NPDR** &mdash; 4-2-1 criteria (extensive multi-quadrant hemorrhages); urgent referral within 2-4 weeks.
* **Grade 4: Proliferative DR (PDR)** &mdash; Neovascularization (NVD/NVE), preretinal/vitreous hemorrhage; emergency referral within 48-72 hours.
* **Hybrid Multi-Model Consensus Engine:** Predicts using a fusion of Primary ResNet-50 (`models/dr_resnet50_weights.pth`) and specialized lightweight DR-Mini AI EfficientNet-B0 (`models/dr_mini_weights.pth`), with architectural readiness for RETFound ViT-Large/16.

### F-02: Transparent Explainable AI via Grad-CAM
* Registers PyTorch forward and backward gradient hooks on the final convolutional bottleneck layer (`layer4[-1]`).
* Computes weighted activation maps and projects thermal colormap overlays (JET) directly onto original fundus photographs.
* Automatically analyzes spatial attention across retinal quadrants (Superior-Temporal, Superior-Nasal, Inferior-Temporal, Inferior-Nasal) and central macular zones.
* **Ethical Transparency:** Explicitly reminds healthcare workers that Grad-CAM depicts deep neural saliency, not definitive histological pathology.

### F-03: Automated Eye Detection & Cropping (Gemini Vision AI)
* Automatically detects the retinal circular field of view and crops peripheral black borders.
* Powered by Google Gemini Vision AI (`gemini-1.5-flash` / `gemini-pro-vision`) when `GEMINI_API_KEY` is configured.
* Certified zero-failure OpenCV contour circle detector fallback when operating in offline rural health posts without internet access.

### F-04: Model Confidence and Top-2 Margin Quality Gate
* Computes model confidence and the numerical margin between the top two predicted class probabilities ($P_{top1} - P_{top2}$).
* If confidence falls below 50% or the top-2 margin is narrower than 15%, the case is automatically flagged as uncertain and quarantined for clinical doctor review.

### F-05: Multilingual Medical Explanations and Text-to-Speech (TTS)
* Full localization across three languages:
  * **English (EN)**
  * **Kannada - ಕನ್ನಡ (KN)** (Tailored for Karnataka rural health posts)
  * **Hindi - हिन्दी (HI)**
* Spoken voice explanations powered by `gTTS` with client-side browser Web Speech API fallback for offline rural clinics.

### F-06: Strict Clinical Gating & Doctor-in-the-Loop Verification
* Every AI screening is logged as `Pending` until formally signed off by medical personnel.
* Primary Care Medical Officers and consulting ophthalmologists can verify, modify (assign clinical grade 0-4), or reject ungradable scans with clinical notes and audit timestamps.
* Distinguishes between AI-assisted predictions and doctor-verified outcomes on all digital and printed records.

### F-07: Patient Registration & Longitudinal Visit History
* Tracks patients with unique auto-generated IDs (`DR-YYYY-XXXX`).
* Rigorously isolates Left Eye (OS) and Right Eye (OD) clinical records to avoid accidental cross-contamination.
* Comprehensive longitudinal progression timeline displays past screening dates, grades, visual comparisons, and clinical notes.

### F-08: Dual PDF Report Generation & Email Dispatch
* **Printable Hard-Copy Report:** High-contrast, black-and-white ink-saving layout optimized for standard PHC thermal or laser printers.
* **Digital Soft-Copy Report:** Full-color, high-fidelity report featuring original fundus, CLAHE-enhanced image, Grad-CAM attention heatmap, computer vision anatomical structures, 5-class probability chart, multilingual summary, and audit stamp.
* **Email Dispatch:** Directly delivers PDF clinical summaries to district hospital eye desks or patients, with offline PHC audit trail logging.

---

## Directory Structure

```
dr_vision_ai/
│
├── app.py                      # Flask Application factory & server entry point
├── config.py                   # Central configuration & clinical thresholds
├── requirements.txt            # Locked, stable dependency list
├── README.md                   # System documentation & deployment guide
├── test_app.py                 # Comprehensive automated test suite
├── create_sample_assets.py     # Synthetic fundus generator for demo cohort
│
├── database/
│   ├── db.py                   # SQLite connection, transaction & seeding manager
│   └── schema.sql              # Relational schema (users, patients, screenings, audit)
│
├── models/
│   ├── dr_model.py             # ResNet-50 5-class neural network architecture
│   ├── model_loader.py         # Model lifecycle, device management (CPU/CUDA), demo mode
│   ├── dr_demo_weights.pth     # (Optional local cache, excluded from Git)
│   └── weights_downloader.py   # CLI weights validator, downloader & deployment utility
│
├── preprocessing/
│   ├── image_quality.py        # Focus (Laplacian), illumination, FOV quality guard
│   ├── image_enhancement.py    # LAB CLAHE, Ben Graham color normalization, sharpening
│   └── retinal_analysis.py     # Optic disc, vessel tree, and candidate lesion mapping
│
├── explainability/
│   └── gradcam.py              # PyTorch hook-based Grad-CAM generator & quadrant analyzer
│
├── reports/
│   └── pdf_generator.py        # Medical-grade ReportLab PDF report generation
│
├── routes/
│   ├── auth.py                 # Login, session management & role-based access
│   ├── dashboard.py            # Dynamic DB analytics, Chart.js metrics
│   ├── patients.py             # Patient registration, search & longitudinal records
│   ├── screening.py            # Upload, quality guard, inference, Grad-CAM, results
│   ├── reports.py              # PDF download, preview & archive
│   ├── about.py                # Public health context & explainability educational guide
│   └── settings.py             # PHC configuration & model diagnostics
│
├── templates/
│   ├── base.html               # Healthcare dashboard layout, sidebar & language switch
│   ├── login.html              # Medical personnel sign in
│   ├── dashboard.html          # Dynamic metrics, Chart.js charts, recent activity
│   ├── screening.html          # 4-step screening workflow & camera integration
│   ├── results.html            # Diagnostic verdict, 4-quadrant views, Grad-CAM, referral
│   ├── patients.html           # Patient directory & registration modal
│   ├── patient_details.html    # Patient profile with past screenings timeline
│   ├── history.html            # Historical screening audit log with grade filters
│   ├── reports.html            # Clinical PDF report repository
│   ├── explainability.html     # Interactive Grad-CAM educational explorer
│   ├── about.html              # Public health background & architecture diagram
│   └── settings.html           # PHC operational settings & model diagnostics
│
├── static/
│   ├── css/styles.css          # Medical styling, teal & dark navy accents
│   ├── js/main.js              # Drag-drop, live quality check, camera trigger
│   ├── js/charts.js            # Chart.js visualizations
│   ├── images/logo.svg         # Official DR Vision AI vector logo
│   └── uploads/                # Fundus captures, Grad-CAMs, and PDF reports
│
├── translations/
│   ├── en.json                 # English strings
│   ├── kn.json                 # Kannada strings (ಕನ್ನಡ)
│   └── hi.json                 # Hindi strings (हिन्दी)
│
└── instance/
    └── dr_vision.db            # SQLite local database
```

---

## Quickstart & Local Execution Instructions

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Windows, Linux, or macOS

### 1. Installation

Clone or extract the repository and install dependencies:
```bash
cd "AI for diabetic retinopathy"
pip install -r requirements.txt
```

### 2. Generate Initial Assets & Database
Initialize the database and generate sample retinal cases:
```bash
python create_sample_assets.py
```

### 3. Run Automated Tests
Verify all 8 test suites (Authentication, Dashboard, Patients, Quality Guard, AI Pipeline, Grad-CAM, PDF Reports, Multilingual):
```bash
python test_app.py
```

### 4. Launch the Web Application
```bash
python app.py
```

Access the application in any web browser:
```
http://127.0.0.1:5000
```

---

## Default Login Credentials

| Role | Username | Password | Purpose |
| :--- | :--- | :--- | :--- |
| **Community Health Officer / Nurse** | `phc_worker` | `worker123` | Patient intake, fundus upload, screening triage |
| **Primary Care Medical Officer** | `dr_patil` | `doctor123` | Clinical review, referral authorization, report sign-off |
| **System Administrator** | `admin` | `admin123` | PHC settings, model weights deployment |

*(Quick-fill credential buttons are provided on the login page for rapid demonstration).*

---

## Integrating Trained Production Model Weights

DR Vision AI supports any PyTorch ResNet-50 weights fine-tuned on standard benchmarks such as **APTOS 2019 Blindness Detection** or **EyePACS**:

1. Obtain your trained weights file (`.pth`).
2. Deploy directly via the CLI validator:
   ```bash
   python models/weights_downloader.py --integrate /path/to/dr_resnet50_weights.pth
   ```
3. Or place the weights file at `models/dr_resnet50_weights.pth`. The application will automatically detect the weights and switch from Demonstration Mode to Certified Production Mode.

---

## Statutory Medical & Ethical Disclaimer

> **IMPORTANT CLINICAL NOTICE:** DR Vision AI is a computer-assisted decision support system engineered for clinical triage in Primary Health Centres (PHCs). It is **NOT** a standalone medical device and does not produce a legally binding or definitive medical diagnosis. Heatmaps generated via Grad-CAM show neural network feature attention and must not be interpreted as confirmed histological pathology. Final diagnosis, pupillary dilation, and treatment decisions must be made by a certified ophthalmologist.
# AI-for-diabetic-retinopathy
