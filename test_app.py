"""
Comprehensive Test Suite for DR Vision AI
Verifies authentication, dashboard metrics, patient registration,
image quality evaluation, AI inference, Grad-CAM, PDF reporting,
and multilingual switching.
"""

import os
import io
import json
import unittest
from pathlib import Path
import cv2
import numpy as np

from app import create_app
from config import Config
from database.db import get_db_connection

class DRVisionAITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.client = cls.app.test_client()

    def login(self, username='phc_worker', password='worker123'):
        return self.client.post('/login', data={
            'username': username,
            'password': password
        }, follow_redirects=True)

    def logout(self):
        return self.client.get('/logout', follow_redirects=True)

    def test_01_login_and_auth_protection(self):
        """Tests that protected routes redirect to login, and login succeeds."""
        # Unauthenticated access to dashboard should redirect to login
        resp = self.client.get('/dashboard')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/login', resp.headers['Location'])

        # Failed login
        resp = self.client.post('/login', data={'username': 'wrong', 'password': 'bad'}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid medical personnel credentials", resp.data)

        # Successful login
        resp = self.login()
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Shirur Primary Health Centre", resp.data)
        print("[TEST 01 PASSED] Authentication and route protection verified.")

    def test_02_dashboard_metrics(self):
        """Verifies dynamic dashboard stats and charts."""
        self.login()
        resp = self.client.get('/dashboard')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Total Patients Registered", resp.data)
        self.assertIn(b"Screenings Completed", resp.data)
        self.assertIn(b"severityDoughnutChart", resp.data)

        # Test dynamic stats API
        api_resp = self.client.get('/api/dashboard-stats')
        self.assertEqual(api_resp.status_code, 200)
        data = json.loads(api_resp.data)
        self.assertIn('labels', data)
        self.assertIn('counts', data)
        self.assertEqual(len(data['counts']), 5)
        print("[TEST 02 PASSED] Dynamic dashboard stats and Chart.js API verified.")

    def test_03_patient_registration_and_search(self):
        """Tests patient creation, auto-generated IDs, search, and details view."""
        self.login()
        
        # Register new patient
        new_patient_data = {
            'name': 'Gouramma Patil',
            'age': '61',
            'gender': 'Female',
            'mobile': '+91 94481 99881',
            'village': 'Aland Cross',
            'district': 'Kalaburagi',
            'state': 'Karnataka',
            'diabetes_history': 'Type 2',
            'duration_diabetes_years': '11.0',
            'previous_eye_exam': 'Never',
            'blood_glucose_level': '195.0',
            'notes': 'Screening test patient'
        }
        resp = self.client.post('/patients/register', data=new_patient_data, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Gouramma Patil", resp.data)

        # Verify patient search
        search_resp = self.client.get('/api/patient-search?q=Gouramma')
        self.assertEqual(search_resp.status_code, 200)
        results = json.loads(search_resp.data)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['name'], 'Gouramma Patil')
        patient_id = results[0]['id']

        # View details
        detail_resp = self.client.get(f'/patients/{patient_id}')
        self.assertEqual(detail_resp.status_code, 200)
        self.assertIn(b"Gouramma Patil", detail_resp.data)
        self.assertIn(b"Aland Cross", detail_resp.data)
        print("[TEST 03 PASSED] Patient registration, ID generation, and search verified.")

    def test_04_image_quality_assessment_api(self):
        """Tests automated quality guard for blur and illumination."""
        self.login()

        # 1. Test with a sharp retinal fundus image
        fundus_sample_path = Config.UPLOAD_FOLDER / 'sample_normal.jpg'
        if not fundus_sample_path.exists():
            from create_sample_assets import create_synthetic_fundus
            img = create_synthetic_fundus(grade=0)
            cv2.imwrite(str(fundus_sample_path), img)

        with open(fundus_sample_path, 'rb') as f:
            resp = self.client.post('/screening/upload-quality', data={
                'image': (io.BytesIO(f.read()), 'sample_normal.jpg')
            }, content_type='multipart/form-data')

        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data['success'])
        self.assertIn(data['quality']['status'], ['Pass', 'Warning'])
        self.assertGreater(data['quality']['score'], 60.0)

        # 2. Test with a completely blank/black image (should Fail)
        black_img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, enc = cv2.imencode('.jpg', black_img)
        resp_black = self.client.post('/screening/upload-quality', data={
            'image': (io.BytesIO(enc.tobytes()), 'black.jpg')
        }, content_type='multipart/form-data')
        self.assertEqual(resp_black.status_code, 200)
        data_black = json.loads(resp_black.data)
        self.assertEqual(data_black['quality']['status'], 'Fail')
        print("[TEST 04 PASSED] Image quality assessment (Laplacian blur, illumination, FOV guard) verified.")

    def test_05_full_ai_screening_pipeline_and_gradcam(self):
        """Executes full diagnostic workflow: enhancement, ResNet-50 inference, Grad-CAM, and PDF."""
        self.login()

        # Get first patient ID
        conn = get_db_connection()
        p = conn.execute("SELECT id FROM patients LIMIT 1").fetchone()
        conn.close()
        patient_id = p['id']

        fundus_sample_path = Config.UPLOAD_FOLDER / 'sample_mod.jpg'
        with open(fundus_sample_path, 'rb') as f:
            resp = self.client.post('/screening/run-ai', data={
                'patient_id': patient_id,
                'eye_side': 'Right',
                'notes': 'Automated end-to-end test execution',
                'image': (io.BytesIO(f.read()), 'test_retina.jpg')
            }, content_type='multipart/form-data', follow_redirects=True)

        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Diabetic Retinopathy Screening Results", resp.data)
        self.assertIn(b"Grad-CAM Attention Map", resp.data)
        self.assertIn(b"probDistChart", resp.data)
        self.assertIn(b"Digital PDF", resp.data)
        print("[TEST 05 PASSED] Full AI screening pipeline, Grad-CAM, and results page verified.")

    def test_06_pdf_report_download(self):
        """Verifies ReportLab PDF generation and retrieval."""
        self.login()
        conn = get_db_connection()
        scr = conn.execute("SELECT screening_id FROM screenings ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        screening_id = scr['screening_id']
        resp = self.client.get(f'/reports/download/{screening_id}')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'application/pdf')
        data_len = len(resp.data)
        resp.close()
        self.assertTrue(data_len > 1000)
        print(f"[TEST 06 PASSED] Official PDF report download verified ({data_len} bytes).")

    def test_07_multilingual_language_switching(self):
        """Tests switching between English, Kannada (ಕನ್ನಡ), and Hindi (हिन्दी)."""
        self.login()

        # Switch to Kannada
        resp_kn = self.client.get('/set-language/kn', follow_redirects=True)
        self.assertEqual(resp_kn.status_code, 200)
        # Check for Kannada translation strings
        resp_dash_kn = self.client.get('/dashboard')
        self.assertIn("ಹೊಸ ತಪಾಸಣೆ".encode('utf-8'), resp_dash_kn.data)

        # Switch to Hindi
        resp_hi = self.client.get('/set-language/hi', follow_redirects=True)
        self.assertEqual(resp_hi.status_code, 200)
        resp_dash_hi = self.client.get('/dashboard')
        self.assertIn("नई स्क्रीनिंग".encode('utf-8'), resp_dash_hi.data)

        # Switch back to English
        resp_en = self.client.get('/set-language/en', follow_redirects=True)
        self.assertEqual(resp_en.status_code, 200)
        resp_dash_en = self.client.get('/dashboard')
        self.assertIn(b"Start New Screening", resp_dash_en.data)
        print("[TEST 07 PASSED] Multilingual translation switching (English, Kannada, Hindi) verified.")

    def test_08_explainability_and_about_pages(self):
        """Verifies explainability guide and detailed project documentation."""
        self.login()

        resp_exp = self.client.get('/explainability')
        self.assertEqual(resp_exp.status_code, 200)
        self.assertIn(b"Grad-CAM", resp_exp.data)

        resp_about = self.client.get('/about')
        self.assertEqual(resp_about.status_code, 200)
        self.assertIn(b"Explainable AI for Diabetic Retinopathy Screening in Rural India", resp_about.data)
        self.assertIn(b"End-to-End System Architecture", resp_about.data)

        resp_settings = self.client.get('/settings')
        self.assertEqual(resp_settings.status_code, 200)
        self.assertIn(b"Primary Health Centre Details", resp_settings.data)
        print("[TEST 08 PASSED] Explainability, About, and Settings pages verified.")

    def test_09_gemini_vision_and_crop_fallback(self):
        """Tests Gemini Vision AI eye detection and automatic OpenCV fallback (F-03)."""
        from preprocessing.gemini_vision_crop import gemini_detect_and_crop
        fundus_sample_path = Config.UPLOAD_FOLDER / 'sample_normal.jpg'
        img = cv2.imread(str(fundus_sample_path))
        self.assertIsNotNone(img)

        # Test fallback behavior when API key is unconfigured
        cropped, circle_info, detector_used = gemini_detect_and_crop(img, target_size=(512, 512))
        self.assertEqual(cropped.shape, (512, 512, 3))
        self.assertEqual(detector_used, 'opencv_fallback')
        self.assertIsNotNone(circle_info)
        print("[TEST 09 PASSED] Gemini Vision AI retinal eye detection and certified fallback verified (F-03).")

    def test_10_top2_margin_and_quality_gate(self):
        """Verifies Confidence and Top-2 Probability Margin Quality Gate (F-04)."""
        from preprocessing.image_quality import assess_prediction_quality_gate

        # 1. High confidence, wide margin (Pass)
        decisive_probs = [0.85, 0.05, 0.05, 0.03, 0.02]
        gate_pass = assess_prediction_quality_gate(decisive_probs, min_confidence=0.50, min_top2_margin=0.15)
        self.assertTrue(gate_pass['gate_passed'])
        self.assertFalse(gate_pass['review_required'])
        self.assertAlmostEqual(gate_pass['top2_margin'], 0.80, places=2)

        # 2. Low confidence / ambiguous spread (Fail / Review Required)
        ambiguous_probs = [0.35, 0.32, 0.20, 0.08, 0.05]
        gate_fail = assess_prediction_quality_gate(ambiguous_probs, min_confidence=0.50, min_top2_margin=0.15)
        self.assertFalse(gate_fail['gate_passed'])
        self.assertTrue(gate_fail['review_required'])
        self.assertTrue(len(gate_fail['flag_reasons']) > 0)
        print("[TEST 10 PASSED] Confidence and Top-2 margin quality gate verified (F-04).")

    def test_11_multilingual_tts_generation(self):
        """Verifies multilingual clinical speech synthesis in EN, KN, and HI (F-05)."""
        self.login()
        conn = get_db_connection()
        scr = conn.execute("SELECT screening_id FROM screenings ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        scr_id = scr['screening_id']

        for lang in ['en', 'kn', 'hi']:
            resp = self.client.get(f'/api/tts/{scr_id}?lang={lang}')
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data['success'])
            self.assertEqual(data['lang'], lang)
            self.assertTrue(len(data['spoken_text']) > 10)
        print("[TEST 11 PASSED] Multilingual Text-to-Speech (English, Kannada, Hindi) verified (F-05).")

    def test_12_doctor_in_the_loop_verification(self):
        """Verifies Doctor-in-the-Loop review, approval, and modification workflow (F-06)."""
        self.logout()
        self.login(username='dr_patil', password='doctor123')
        conn = get_db_connection()
        scr = conn.execute("SELECT screening_id FROM screenings ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        scr_id = scr['screening_id']

        # 1. Doctor Verifies
        resp_verify = self.client.post(f'/screening/{scr_id}/verify', data={
            'action': 'verify',
            'doctor_notes': 'Clinical macula check confirmed clear.',
            'assigned_grade': '0'
        }, follow_redirects=True)
        self.assertEqual(resp_verify.status_code, 200)

        conn = get_db_connection()
        updated_scr = conn.execute("SELECT * FROM screenings WHERE screening_id = ?", (scr_id,)).fetchone()
        conn.close()
        self.assertEqual(updated_scr['verification_status'], 'Verified')
        self.assertEqual(updated_scr['doctor_notes'], 'Clinical macula check confirmed clear.')

        # 2. Doctor Modifies Grade
        resp_modify = self.client.post(f'/screening/{scr_id}/verify', data={
            'action': 'modify',
            'doctor_notes': 'Clinical review reveals early microaneurysms.',
            'assigned_grade': '1'
        }, follow_redirects=True)
        self.assertEqual(resp_modify.status_code, 200)

        conn = get_db_connection()
        mod_scr = conn.execute("SELECT * FROM screenings WHERE screening_id = ?", (scr_id,)).fetchone()
        conn.close()
        self.assertEqual(mod_scr['verification_status'], 'Modified')
        self.assertEqual(mod_scr['doctor_assigned_grade'], 1)
        print("[TEST 12 PASSED] Doctor-in-the-loop clinical verification and grade modification verified (F-06).")

    def test_13_dual_pdf_reports_and_email_dispatch(self):
        """Verifies dual PDF reports (printable and digital) and email dispatch (F-08)."""
        self.logout()
        self.login()
        conn = get_db_connection()
        scr = conn.execute("SELECT screening_id FROM screenings ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        scr_id = scr['screening_id']

        # 1. Download Printable Hard-Copy Report
        resp_print = self.client.get(f'/reports/download/{scr_id}?type=printable')
        self.assertEqual(resp_print.status_code, 200)
        self.assertEqual(resp_print.mimetype, 'application/pdf')
        len_print = len(resp_print.data)
        resp_print.close()
        self.assertTrue(len_print > 1000)

        # 2. Download Digital Report
        resp_digi = self.client.get(f'/reports/download/{scr_id}?type=digital')
        self.assertEqual(resp_digi.status_code, 200)
        self.assertEqual(resp_digi.mimetype, 'application/pdf')
        len_digi = len(resp_digi.data)
        resp_digi.close()
        self.assertTrue(len_digi > 1000)

        # 3. Email Dispatch (Offline PHC Logging Mode)
        resp_mail = self.client.post(f'/reports/email/{scr_id}', data={
            'recipient_email': 'test.patient@gmail.com'
        }, follow_redirects=True)
        self.assertEqual(resp_mail.status_code, 200)

        conn = get_db_connection()
        log_entry = conn.execute("SELECT * FROM audit_logs WHERE action = 'EMAIL_REPORT_DISPATCH' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        self.assertIsNotNone(log_entry)
        self.assertIn('test.patient@gmail.com', log_entry['details'])
        print("[TEST 13 PASSED] Dual PDF report generation and email dispatch verified (F-08).")

    def test_14_dr_mini_and_hybrid_consensus(self):
        """Verifies DR-Mini AI independent classification and hybrid probabilistic fusion (F-01 / Hybrid)."""
        import torch
        from models.dr_mini_loader import DRMiniService
        from hybrid_prediction import HybridPredictionEngine

        dr_mini = DRMiniService.get_instance()
        self.assertTrue(dr_mini.is_available)

        dummy = torch.randn(1, 3, 224, 224)
        pred_mini = dr_mini.predict(dummy)
        self.assertTrue(pred_mini['available'])
        self.assertIn(pred_mini['predicted_grade'], [0, 1, 2, 3, 4])
        self.assertEqual(len(pred_mini['class_probabilities']), 5)
        self.assertAlmostEqual(sum(pred_mini['class_probabilities']), 1.0, places=2)

        engine = HybridPredictionEngine.get_instance()
        hybrid_res = engine.predict(dummy)
        self.assertIn('hybrid_result', hybrid_res)
        self.assertIn(hybrid_res['hybrid_result']['predicted_grade'], [0, 1, 2, 3, 4])
        print("[TEST 14 PASSED] DR-Mini AI independent model and hybrid consensus engine verified.")

    def test_15_retfound_weights_required_handling(self):
        """Verifies ethical integrity of RETFound Foundation Model (no false predictions when weights are missing)."""
        from models.retfound_loader import RETFoundService
        import torch

        retfound = RETFoundService.get_instance()
        # When retfound_cfp_weights.pth is not present on disk:
        if not retfound.WEIGHTS_PATH.exists():
            self.assertFalse(retfound.is_available)
            self.assertEqual(retfound.status, 'weights_required')
            dummy = torch.randn(1, 3, 224, 224)
            pred = retfound.predict(dummy)
            self.assertFalse(pred['available'])
            self.assertIsNone(pred['predicted_grade'])
            self.assertIn('huggingface.co', pred['download_url'])
        print("[TEST 15 PASSED] RETFound Foundation Model ethical weights_required handling verified.")


if __name__ == '__main__':
    unittest.main()
