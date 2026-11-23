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
from reports.pdf_generator import generate_screening_pdf, send_screening_email

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
    report_type = request.args.get('type', 'digital')  # 'digital' or 'printable'

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

    pdf_suffix = "_printable.pdf" if report_type == 'printable' else "_report.pdf"
    pdf_name = f"{screening_id}{pdf_suffix}"
    target_path = Path(Config.UPLOAD_FOLDER) / pdf_name

    # If requested variant does not exist on disk, generate dynamically
    if not target_path.exists():
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
        generate_screening_pdf(screening_dict, patient_dict, str(target_path), report_type=report_type)

    return send_file(
        str(target_path),
        mimetype='application/pdf',
        as_attachment=False,
        download_name=f"DR_Screening_{screening_id}_{report_type}.pdf"
    )

@reports_bp.route('/reports/email/<screening_id>', methods=['POST'])
@login_required
def email_report(screening_id):
    """Dispatches report via email (F-08) with offline PHC simulation fallback."""
    recipient_email = request.form.get('recipient_email', '').strip()
    if not recipient_email or '@' not in recipient_email:
        flash("Please enter a valid recipient email address.", "danger")
        return redirect(request.referrer or url_for('reports.index'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as patient_code
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
    pdf_full_path = Config.BASE_DIR / pdf_rel_path if pdf_rel_path else Config.UPLOAD_FOLDER / f"{screening_id}_report.pdf"

    result = send_screening_email(
        dict(screening),
        {'name': screening['patient_name'], 'patient_id': screening['patient_code']},
        recipient_email,
        str(pdf_full_path)
    )

    if result.get('mode') == 'smtp_dispatched':
        flash(f"Screening report successfully emailed to {recipient_email}.", "success")
    else:
        flash(f"Report queued and logged in audit log for dispatch to {recipient_email} (Rural PHC Offline Mode).", "info")

    return redirect(request.referrer or url_for('screening.results', screening_id=screening_id))

