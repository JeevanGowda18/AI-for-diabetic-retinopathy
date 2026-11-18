"""
Patient Management Routes for DR Vision AI
Handles rural cohort registration, input validation, patient search,
and longitudinal screening progression tracking.
"""

from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from database.db import get_db_connection
from routes.auth import login_required
from config import Config

patients_bp = Blueprint('patients', __name__)

def generate_patient_id(conn):
    """Generates unique sequential patient ID: DR-YYYY-XXXX."""
    year = datetime.now().year
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id FROM patients WHERE patient_id LIKE ? ORDER BY id DESC LIMIT 1", (f"DR-{year}-%",))
    row = cursor.fetchone()
    if row:
        last_id = row['patient_id']
        try:
            seq = int(last_id.split('-')[-1]) + 1
        except Exception:
            seq = 1
    else:
        seq = 1
    return f"DR-{year}-{seq:04d}"

@patients_bp.route('/patients')
@login_required
def index():
    query = request.args.get('q', '').strip()
    village = request.args.get('village', '').strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    sql = "SELECT * FROM patients WHERE 1=1"
    params = []

    if query:
        sql += " AND (name LIKE ? OR patient_id LIKE ? OR mobile LIKE ?)"
        term = f"%{query}%"
        params.extend([term, term, term])

    if village:
        sql += " AND village LIKE ?"
        params.append(f"%{village}%")

    sql += " ORDER BY id DESC"
    cursor.execute(sql, params)
    patients = cursor.fetchall()

    # Get distinct villages for filter dropdown
    cursor.execute("SELECT DISTINCT village FROM patients WHERE village IS NOT NULL ORDER BY village ASC")
    villages = [r['village'] for r in cursor.fetchall()]

    conn.close()
    return render_template('patients.html', patients=patients, query=query, selected_village=village, villages=villages)

@patients_bp.route('/patients/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        mobile = request.form.get('mobile', '').strip()
        village = request.form.get('village', '').strip()
        district = request.form.get('district', Config.PHC_DEFAULT['district']).strip()
        state = request.form.get('state', Config.PHC_DEFAULT['state']).strip()
        diabetes_history = request.form.get('diabetes_history', 'Type 2').strip()
        duration_diabetes = request.form.get('duration_diabetes_years', '0').strip()
        previous_eye_exam = request.form.get('previous_eye_exam', 'Never').strip()
        blood_glucose = request.form.get('blood_glucose_level', '').strip()
        notes = request.form.get('notes', '').strip()

        # Validation
        errors = []
        if not name or len(name) < 2:
            errors.append("Patient name is required (at least 2 characters).")
        try:
            age_int = int(age)
            if age_int < 1 or age_int > 120:
                errors.append("Please enter a valid age between 1 and 120.")
        except ValueError:
            errors.append("Age must be a valid number.")

        if not mobile or len(mobile) < 8:
            errors.append("A valid contact/mobile number is required.")

        if not village:
            errors.append("Village or PHC Sub-Centre is required.")

        if errors:
            for err in errors:
                flash(err, "danger")
            return redirect(url_for('patients.index'))

        conn = get_db_connection()
        patient_code = generate_patient_id(conn)

        try:
            duration_val = float(duration_diabetes) if duration_diabetes else 0.0
            glucose_val = float(blood_glucose) if blood_glucose else None

            conn.execute(
                """INSERT INTO patients (
                    patient_id, name, age, gender, mobile, village, district, state,
                    diabetes_history, duration_diabetes_years, previous_eye_exam, blood_glucose_level, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (patient_code, name, int(age), gender, mobile, village, district, state,
                 diabetes_history, duration_val, previous_eye_exam, glucose_val, notes)
            )
            conn.commit()
            flash(f"Patient registered successfully: {name} (ID: {patient_code})", "success")
        except Exception as e:
            conn.rollback()
            flash(f"Database error registering patient: {e}", "danger")
        finally:
            conn.close()

        return redirect(url_for('patients.index'))

    return redirect(url_for('patients.index'))

@patients_bp.route('/patients/<int:patient_id>')
@login_required
def details(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()

    if not patient:
        conn.close()
        flash("Patient record not found.", "warning")
        return redirect(url_for('patients.index'))

    # Retrieve all screenings for this patient in chronological order
    cursor.execute("""
        SELECT * FROM screenings 
        WHERE patient_id = ? 
        ORDER BY created_at DESC
    """, (patient_id,))
    screenings = cursor.fetchall()
    conn.close()

    return render_template(
        'patient_details.html',
        patient=patient,
        screenings=screenings,
        dr_classes=Config.DR_CLASSES
    )

@patients_bp.route('/api/patient-search')
@login_required
def api_search():
    term = request.args.get('q', '').strip()
    if not term:
        return jsonify([])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, patient_id, name, age, gender, village, mobile 
        FROM patients 
        WHERE name LIKE ? OR patient_id LIKE ? OR mobile LIKE ?
        LIMIT 10
    """, (f"%{term}%", f"%{term}%", f"%{term}%"))
    results = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify(results)
