"""
Screening Workflow and AI Diagnostics Pipeline
Orchestrates:
1. Patient assignment
2. Image quality assessment (Laplacian blur, illumination, FOV)
3. LAB CLAHE enhancement
4. PyTorch ResNet-50 inference (5 classes)
5. Grad-CAM explainability overlay
6. Retinal structure and lesion candidate mapping (CV)
7. ReportLab PDF report generation
8. Database persistence & results presentation
"""

import os
import cv2
import json
import uuid
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session

from database.db import get_db_connection
from routes.auth import login_required
from config import Config
from preprocessing.image_quality import assess_image_quality
from preprocessing.image_enhancement import enhance_retinal_image, prepare_tensor_for_model
from preprocessing.retinal_analysis import analyze_retinal_structures
from models.model_loader import ModelService
from explainability.gradcam import GradCAM
from reports.pdf_generator import generate_screening_pdf

screening_bp = Blueprint('screening', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_screening_id(conn):
    year = datetime.now().year
    cursor = conn.cursor()
    cursor.execute("SELECT screening_id FROM screenings WHERE screening_id LIKE ? ORDER BY id DESC LIMIT 1", (f"SCR-{year}-%",))
    row = cursor.fetchone()
    if row:
        last_id = row['screening_id']
        try:
            seq = int(last_id.split('-')[-1]) + 1
        except Exception:
            seq = 1
    else:
        seq = 101
    return f"SCR-{year}-{seq:04d}"

@screening_bp.route('/screening/new')
@login_required
def new_screening():
    patient_id = request.args.get('patient_id')
    conn = get_db_connection()
    cursor = conn.cursor()

    selected_patient = None
    if patient_id:
        cursor.execute("SELECT * FROM patients WHERE id = ? OR patient_id = ?", (patient_id, patient_id))
        selected_patient = cursor.fetchone()

    cursor.execute("SELECT id, patient_id, name, age, gender, village FROM patients ORDER BY id DESC LIMIT 30")
    patients_list = cursor.fetchall()
    conn.close()

    model_service = ModelService.get_instance()
    model_info = model_service.get_info()

    return render_template(
        'screening.html',
        selected_patient=selected_patient,
        patients=patients_list,
        model_info=model_info,
        dr_classes=Config.DR_CLASSES
    )

@screening_bp.route('/screening/upload-quality', methods=['POST'])
@login_required
def check_quality_async():
    """
    AJAX endpoint for instant feedback on image quality
    before committing to full AI inference.
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid image format. Supported formats: JPG, JPEG, PNG.'}), 400

    # Save to temp directory
    upload_dir = Path(Config.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_name = f"temp_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    temp_path = upload_dir / temp_name
    file.save(str(temp_path))

    # Read image with OpenCV
    img_bgr = cv2.imread(str(temp_path))
    if img_bgr is None:
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({'success': False, 'error': 'Failed to decode image file.'}), 400

    # Evaluate image quality
    quality_result = assess_image_quality(img_bgr, Config.QUALITY_THRESHOLDS)

    return jsonify({
        'success': True,
        'temp_filename': temp_name,
        'temp_url': url_for('static', filename=f'uploads/{temp_name}'),
        'quality': quality_result
    })

@screening_bp.route('/screening/run-ai', methods=['POST'])
@login_required
def run_ai():
    """Executes full diagnostic pipeline."""
    patient_id = request.form.get('patient_id')
    eye_side = request.form.get('eye_side', 'Right')
    temp_filename = request.form.get('temp_filename')
    screening_notes = request.form.get('notes', '')

    if not patient_id:
        flash("Please select a registered patient for screening.", "danger")
        return redirect(url_for('screening.new_screening'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ? OR patient_id = ?", (patient_id, patient_id))
    patient = cursor.fetchone()
    if not patient:
        conn.close()
        flash("Selected patient record not found.", "danger")
        return redirect(url_for('screening.new_screening'))

    upload_dir = Path(Config.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Acquire image either from temp file or direct file upload
    if temp_filename and (upload_dir / temp_filename).exists():
        src_path = upload_dir / temp_filename
        img_bgr = cv2.imread(str(src_path))
    elif 'image' in request.files and request.files['image'].filename:
        file = request.files['image']
        if not allowed_file(file.filename):
            conn.close()
            flash("Invalid image format. Please upload JPG or PNG.", "danger")
            return redirect(url_for('screening.new_screening'))
        fname = f"orig_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        src_path = upload_dir / fname
        file.save(str(src_path))
        img_bgr = cv2.imread(str(src_path))
    else:
        conn.close()
        flash("No retinal image provided. Please upload or capture an image.", "danger")
        return redirect(url_for('screening.new_screening'))

    if img_bgr is None:
        conn.close()
        flash("Failed to read image. Image may be corrupt.", "danger")
        return redirect(url_for('screening.new_screening'))

    # 1. Quality Check
    quality = assess_image_quality(img_bgr, Config.QUALITY_THRESHOLDS)
    if quality['status'] == 'Fail':
        conn.close()
        flash("Screening blocked: Retinal image quality failed clinical focus/illumination thresholds. Please re-capture image.", "danger")
        return redirect(url_for('screening.new_screening', patient_id=patient['id']))

    # Generate permanent screening identifiers
    screening_code = generate_screening_id(conn)
    base_name = f"{screening_code}_{patient['patient_id']}"
    
    orig_save_name = f"{base_name}_orig.jpg"
    proc_save_name = f"{base_name}_proc.jpg"
    cam_save_name = f"{base_name}_gradcam.jpg"
    cv_save_name = f"{base_name}_cv.jpg"
    pdf_save_name = f"{screening_code}_report.pdf"

    orig_full_path = upload_dir / orig_save_name
    proc_full_path = upload_dir / proc_save_name
    cam_full_path = upload_dir / cam_save_name
    cv_full_path = upload_dir / cv_save_name
    pdf_full_path = upload_dir / pdf_save_name

    # Save original image
    cv2.imwrite(str(orig_full_path), img_bgr)

    # 2. Image Enhancement (LAB CLAHE & Ben Graham / Sharpening)
    enhanced_bgr = enhance_retinal_image(img_bgr)
    cv2.imwrite(str(proc_full_path), enhanced_bgr)

    # 3. Retinal Structure & Candidate Lesion Detection (CV Heuristics)
    cv_annotated, vessels, cv_stats = analyze_retinal_structures(enhanced_bgr)
    cv2.imwrite(str(cv_full_path), cv_annotated)

    # 4. PyTorch AI Model Inference
    model_service = ModelService.get_instance()
    tensor = prepare_tensor_for_model(enhanced_bgr, device=model_service.device)
    prediction = model_service.predict(tensor)

    pred_grade = prediction['predicted_grade']
    confidence = prediction['confidence']
    probabilities = prediction['probabilities']
    is_demo = prediction['is_demo_mode']
    model_mode = 'demo_calibrated' if is_demo else 'production_trained'

    grade_meta = Config.DR_CLASSES.get(pred_grade, Config.DR_CLASSES[0])
    is_referable = 1 if pred_grade >= 2 else 0

    # 5. Grad-CAM Explainability Map
    gradcam = GradCAM(model_service.model)
    heatmap, _ = gradcam.generate_heatmap(tensor, target_class=pred_grade)
    overlay_cam, _ = gradcam.overlay_on_image(enhanced_bgr, heatmap)
    cv2.imwrite(str(cam_full_path), overlay_cam)
    cam_analysis = gradcam.analyze_activated_regions(heatmap, enhanced_bgr.shape)

    # 6. Generate Official PDF Report
    screening_record = {
        'screening_id': screening_code,
        'eye_side': eye_side,
        'original_image_path': str(orig_full_path),
        'processed_image_path': str(proc_full_path),
        'gradcam_image_path': str(cam_full_path),
        'cv_analysis_path': str(cv_full_path),
        'quality_status': quality['status'],
        'quality_score': quality['score'],
        'predicted_grade': pred_grade,
        'predicted_class_name': grade_meta['name'],
        'confidence_score': confidence,
        'class_probabilities': probabilities,
        'is_referable': is_referable,
        'referral_urgency': grade_meta['referral_urgency'],
        'clinical_recommendation': grade_meta['recommendation_en'],
        'model_mode': model_mode,
        'screening_notes': screening_notes or cam_analysis['text_explanation']
    }

    patient_dict = dict(patient)
    generate_screening_pdf(screening_record, patient_dict, str(pdf_full_path))

    # 7. Persist to Database
    cursor.execute("""
        INSERT INTO screenings (
            screening_id, patient_id, user_id, eye_side,
            original_image_path, processed_image_path, gradcam_image_path, cv_analysis_path,
            quality_status, quality_score, quality_details,
            predicted_grade, predicted_class_name, confidence_score, class_probabilities,
            is_referable, referral_urgency, clinical_recommendation,
            model_mode, screening_notes, pdf_report_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        screening_code, patient['id'], session.get('user_id'), eye_side,
        f"static/uploads/{orig_save_name}",
        f"static/uploads/{proc_save_name}",
        f"static/uploads/{cam_save_name}",
        f"static/uploads/{cv_save_name}",
        quality['status'], quality['score'], json.dumps(quality['metrics']),
        pred_grade, grade_meta['name'], confidence, json.dumps(probabilities),
        is_referable, grade_meta['referral_urgency'], grade_meta['recommendation_en'],
        model_mode, screening_notes, f"static/uploads/{pdf_save_name}"
    ))
    conn.commit()
    conn.close()

    flash(f"Screening {screening_code} completed successfully. AI Diagnosis generated.", "success")
    return redirect(url_for('screening.results', screening_id=screening_code))

@screening_bp.route('/screening/results/<screening_id>')
@login_required
def results(screening_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as patient_code, p.age, p.gender,
               p.village, p.district, p.mobile, p.diabetes_history, p.duration_diabetes_years,
               p.blood_glucose_level, p.previous_eye_exam
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.screening_id = ?
    """, (screening_id,))
    record = cursor.fetchone()
    conn.close()

    if not record:
        flash(f"Screening record {screening_id} not found.", "warning")
        return redirect(url_for('dashboard.index'))

    probs = json.loads(record['class_probabilities']) if record['class_probabilities'] else [0]*5
    grade = record['predicted_grade'] if record['predicted_grade'] is not None else 0
    grade_meta = Config.DR_CLASSES.get(grade, Config.DR_CLASSES[0])

    return render_template(
        'results.html',
        scr=record,
        grade_meta=grade_meta,
        probabilities=probs,
        dr_classes=Config.DR_CLASSES
    )

@screening_bp.route('/history')
@login_required
def history():
    grade_filter = request.args.get('grade')
    referral_filter = request.args.get('referral')
    search_query = request.args.get('q', '').strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
        SELECT s.*, p.name as patient_name, p.patient_id as patient_code, p.village, p.mobile
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        WHERE 1=1
    """
    params = []

    if search_query:
        sql += " AND (p.name LIKE ? OR p.patient_id LIKE ? OR s.screening_id LIKE ?)"
        term = f"%{search_query}%"
        params.extend([term, term, term])

    if grade_filter is not None and grade_filter != '':
        sql += " AND s.predicted_grade = ?"
        params.append(int(grade_filter))

    if referral_filter == '1':
        sql += " AND s.is_referable = 1"

    sql += " ORDER BY s.created_at DESC"
    cursor.execute(sql, params)
    screenings = cursor.fetchall()
    conn.close()

    return render_template(
        'history.html',
        screenings=screenings,
        dr_classes=Config.DR_CLASSES,
        selected_grade=grade_filter,
        selected_referral=referral_filter,
        query=search_query
    )
