"""
Configuration module for DR Vision AI
Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dr-vision-ai-rural-phc-secret-key-2026')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')

    # Instance & Database
    INSTANCE_DIR = BASE_DIR / 'instance'
    DATABASE_PATH = INSTANCE_DIR / 'dr_vision.db'

    # File uploads
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    # AI Model Configuration
    MODEL_WEIGHTS_PATH = BASE_DIR / 'models' / 'dr_resnet50_weights.pth'
    MODEL_ARCH = 'resnet50'
    DEVICE = 'cuda' if os.environ.get('USE_CUDA', '1') == '1' else 'cpu'
    
    # 5-Class Diabetic Retinopathy Severity Grading (ICDR Classification)
    DR_CLASSES = {
        0: {
            'name': 'No DR',
            'code': 'GRADE_0',
            'severity': 'Normal',
            'badge': 'bg-success',
            'referral_urgency': 'Routine',
            'recommendation_en': 'No diabetic retinopathy detected. Schedule annual follow-up screening and maintain optimal glycemic control.',
            'recommendation_kn': 'ಮಧುಮೇಹ ರೆಟಿನೋಪತಿ ಕಂಡುಬಂದಿಲ್ಲ. ವಾರ್ಷಿಕ ಕಣ್ಣಿನ ತಪಾಸಣೆಯನ್ನು ನಿಗದಿಪಡಿಸಿ ಮತ್ತು ರಕ್ತದ ಸಕ್ಕರೆ ಮಟ್ಟವನ್ನು ನಿಯಂತ್ರಣದಲ್ಲಿಡಿ.',
            'recommendation_hi': 'डायबिटिक रेटिनोपैथी का कोई लक्षण नहीं मिला। वार्षिक अनुवर्ती जांच निर्धारित करें और रक्त शर्करा को नियंत्रित रखें।'
        },
        1: {
            'name': 'Mild NPDR',
            'code': 'GRADE_1',
            'severity': 'Mild Non-Proliferative',
            'badge': 'bg-info',
            'referral_urgency': 'Low',
            'recommendation_en': 'Mild non-proliferative DR (isolated microaneurysms). Schedule follow-up dilated eye exam in 6-12 months.',
            'recommendation_kn': 'ಸೌಮ್ಯ ಪ್ರಮಾಣದ ಮಧುಮೇಹ ರೆಟಿನೋಪತಿ (ಸಣ್ಣ ರಕ್ತನಾಳ ಊತ). 6-12 ತಿಂಗಳಲ್ಲಿ ಕಣ್ಣಿನ ತಜ್ಞರಿಂದ ಮರುಪರೀಕ್ಷೆ ಮಾಡಿಸಿ.',
            'recommendation_hi': 'हल्की नॉन-प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी। 6-12 महीनों में नेत्र रोग विशेषज्ञ से अनुवर्ती जांच कराएं।'
        },
        2: {
            'name': 'Moderate NPDR',
            'code': 'GRADE_2',
            'severity': 'Moderate Non-Proliferative',
            'badge': 'bg-warning',
            'referral_urgency': 'Moderate (Referral)',
            'recommendation_en': 'Moderate non-proliferative DR (multiple microaneurysms, hemorrhages, hard exudates). Refer to ophthalmologist within 2-3 months.',
            'recommendation_kn': 'ಮಧ್ಯಮ ಪ್ರಮಾಣದ ನಾನ್-ಪ್ರೊಲಿಫೆರೇಟಿವ್ ರೆಟಿನೋಪತಿ. 2-3 ತಿಂಗಳೊಳಗೆ ನೇತ್ರತಜ್ಞರಿಗೆ ಕಡ್ಡಾಯವಾಗಿ ತೋರಿಸಿ.',
            'recommendation_hi': 'मध्यम नॉन-प्रोलिफेरेटिव रेटिनोपैथी (रक्तस्राव, एक्सयूडेट्स)। 2-3 महीने के भीतर नेत्र रोग विशेषज्ञ से मिलें।'
        },
        3: {
            'name': 'Severe NPDR',
            'code': 'GRADE_3',
            'severity': 'Severe Non-Proliferative',
            'badge': 'bg-danger',
            'referral_urgency': 'Urgent (Referral)',
            'recommendation_en': 'Severe non-proliferative DR (4-2-1 rule criteria, extensive intraretinal hemorrhages). Urgent ophthalmology referral within 2-4 weeks.',
            'recommendation_kn': 'ತೀವ್ರ ಪ್ರಮಾಣದ ನಾನ್-ಪ್ರೊಲಿಫೆರೇಟಿವ್ ರೆಟಿನೋಪತಿ. 2-4 ವಾರಗಳೊಳಗೆ ತುರ್ತಾಗಿ ನೇತ್ರ ಆಸ್ಪತ್ರೆಗೆ ಕಳುಹಿಸಿ.',
            'recommendation_hi': 'गंभीर नॉन-प्रोलिफेरेटिव रेटिनोपैथी। 2-4 सप्ताह के भीतर तत्काल नेत्र विशेषज्ञ रेफरल आवश्यक है।'
        },
        4: {
            'name': 'Proliferative DR',
            'code': 'GRADE_4',
            'severity': 'Proliferative DR (Sight-Threatening)',
            'badge': 'bg-dark',
            'referral_urgency': 'Critical / Emergency',
            'recommendation_en': 'Proliferative diabetic retinopathy (neovascularization, vitreous hemorrhage risk). Emergency referral within 48-72 hours to prevent permanent vision loss.',
            'recommendation_kn': 'ಪ್ರೊಲಿಫೆರೇಟಿವ್ ರೆಟಿನೋಪತಿ (ಹೊಸ ಅಸಹಜ ರಕ್ತನಾಳಗಳು/ದೃಷ್ಟಿಹಾನಿಯ ತೀವ್ರ ಅಪಾಯ). ತಕ್ಷಣವೇ (48-72 ಗಂಟೆಗಳಲ್ಲಿ) ಜಿಲ್ಲಾ ನೇತ್ರ ಆಸ್ಪತ್ರೆಗೆ ದಾಖಲಿಸಿ.',
            'recommendation_hi': 'प्रोलिफेरेटिव डायबिटिक रेटिनोपैथी (दृष्टि जाने का गंभीर खतरा)। 48-72 घंटों के भीतर आपातकालीन नेत्र अस्पताल रेफरल आवश्यक।'
        }
    }

    # Image Quality Thresholds
    QUALITY_THRESHOLDS = {
        'min_laplacian_var': 25.0,     # Blur cutoff
        'warning_laplacian_var': 60.0, # Acceptable but borderline
        'min_brightness': 30.0,        # Underexposed cutoff
        'max_brightness': 235.0,       # Overexposed cutoff
        'min_fov_ratio': 0.20,         # Retinal area vs total frame
        'min_contrast_rms': 15.0,      # RMS contrast minimum
        'max_quadrant_imbalance': 3.0  # Quadrant uniformity cutoff
    }

    # PHC / Center Default Settings
    PHC_DEFAULT = {
        'center_name': 'Shirur Primary Health Centre (PHC)',
        'district': 'Kalaburagi',
        'state': 'Karnataka',
        'medical_officer': 'Dr. Ramesh Patil, MBBS',
        'operator_role': 'Community Health Officer (CHO)'
    }
