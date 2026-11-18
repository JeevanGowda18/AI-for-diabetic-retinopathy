"""
Reports Management Routes
Provides clinical PDF report downloading, on-the-fly regeneration,
and repository auditing for rural Primary Health Centres.
"""

import os
from pathlib import Path
from flask import Blueprint, render_template, send_file, flash, redirect, url_for, request
from database.db import get_db_connection
from routes.auth import login_required
from config import Config
from reports.pdf_generator import generate_screening_pdf

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/reports')
@login_required
def index():
    query = request.args.get('q', '').strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
        SELECT s.id, s.screening_id, s.eye_side, s.predicted_grade, s.confidence_score,
               s.is_referable, s.pdf_report_path, s.created_at,
               p.name as patient_name, p.patient_id as patient_code, p.village
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        WHERE 1=1
    """
    params = []
    if query:
        sql += " AND (p.name LIKE ? OR p.patient_id LIKE ? OR s.screening_id LIKE ?)"
        term = f"%{query}%"
        params.extend([term, term, term])

    sql += " ORDER BY s.created_at DESC"
    cursor.execute(sql, params)
    reports = cursor.fetchall()
    conn.close()

    return render_template('reports.html', reports=reports, dr_classes=Config.DR_CLASSES, query=query)

@reports_bp.route('/reports/download/<screening_id>')
@login_required
def download_pdf(screening_id):
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
    screening = cursor.fetchone()
    conn.close()

    if not screening:
        flash("Screening record not found.", "warning")
        return redirect(url_for('reports.index'))

    pdf_rel_path = screening['pdf_report_path']
    pdf_full_path = Config.BASE_DIR / pdf_rel_path if pdf_rel_path else None

    # If PDF does not exist on disk, regenerate dynamically
    if not pdf_full_path or not pdf_full_path.exists():
        pdf_name = f"{screening_id}_report.pdf"
        target_path = Path(Config.UPLOAD_FOLDER) / pdf_name
        
        screening_dict = dict(screening)
        patient_dict = {
            'patient_id': screening['patient_code'],
            'name': screening['patient_name'],
            'age': screening['age'],
            'gender': screening['gender'],
            'mobile': screening['mobile'],
            'village': screening['village'],
            'district': screening['district'],
            'diabetes_history': screening['diabetes_history'],
            'duration_diabetes_years': screening['duration_diabetes_years'],
            'blood_glucose_level': screening['blood_glucose_level']
        }
        
        generate_screening_pdf(screening_dict, patient_dict, str(target_path))
        pdf_full_path = target_path

    return send_file(
        str(pdf_full_path),
        mimetype='application/pdf',
        as_attachment=False,
        download_name=f"DR_Screening_Report_{screening_id}.pdf"
    )
