"""
Database helper module for DR Vision AI
Manages SQLite database connections, schema migrations, and initial seeding.
"""

import sqlite3
import json
import os
from pathlib import Path
from werkzeug.security import generate_password_hash
from config import Config

def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    Config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(Config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes the database schema and creates default user accounts."""
    Config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    schema_path = Path(__file__).resolve().parent / 'schema.sql'
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    conn = get_db_connection()
    try:
        conn.executescript(schema_sql)
        conn.commit()
        
        # Seed default users if table is empty
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            default_users = [
                (
                    'admin',
                    generate_password_hash('admin123'),
                    'Dr. Suresh Hegde',
                    'admin',
                    'District Health Office - Kalaburagi',
                    'Kalaburagi'
                ),
                (
                    'phc_worker',
                    generate_password_hash('worker123'),
                    'Pooja Sharanappa',
                    'health_worker',
                    'Shirur Primary Health Centre (PHC)',
                    'Kalaburagi'
                ),
                (
                    'dr_patil',
                    generate_password_hash('doctor123'),
                    'Dr. Ramesh Patil, MBBS',
                    'doctor',
                    'Shirur Primary Health Centre (PHC)',
                    'Kalaburagi'
                )
            ]
            cursor.executemany(
                """INSERT INTO users (username, password_hash, full_name, role, center_name, district)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                default_users
            )
            conn.commit()
            print("[DB] Initialized database and seeded default medical personnel accounts.")

        # Seed sample rural patient cohort if patients table is empty
        cursor.execute("SELECT COUNT(*) FROM patients")
        patient_count = cursor.fetchone()[0]
        if patient_count == 0:
            seed_sample_patients_and_screenings(conn)

        # Apply schema extensions (doctor verification and dual model columns)
        apply_schema_migrations(conn)

    finally:
        conn.close()

def apply_schema_migrations(conn):
    """Adds new columns for doctor verification and hybrid models without altering existing data."""
    cursor = conn.cursor()
    new_columns = [
        ("verification_status", "VARCHAR(20) DEFAULT 'Pending'"),
        ("verified_by", "INTEGER"),
        ("verified_by_name", "VARCHAR(100)"),
        ("doctor_notes", "TEXT"),
        ("verified_at", "TIMESTAMP"),
        ("doctor_assigned_grade", "INTEGER"),
        ("retfound_grade", "INTEGER"),
        ("retfound_confidence", "REAL"),
        ("retfound_probs", "TEXT"),
        ("retfound_status", "VARCHAR(50)"),
        ("dr_mini_grade", "INTEGER"),
        ("dr_mini_confidence", "REAL"),
        ("dr_mini_probs", "TEXT"),
        ("dr_mini_status", "VARCHAR(50)"),
        ("hybrid_grade", "INTEGER"),
        ("hybrid_confidence", "REAL"),
        ("hybrid_probs", "TEXT"),
        ("hybrid_agreement", "VARCHAR(50)"),
        ("dr_mini_gradcam_path", "VARCHAR(255)"),
        ("retfound_gradcam_path", "VARCHAR(255)"),
        ("quality_gate_passed", "INTEGER DEFAULT 1"),
        ("quality_gate_details", "TEXT")
    ]

    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE screenings ADD COLUMN {col_name} {col_def};")
        except sqlite3.OperationalError:
            # Column already exists
            pass
    conn.commit()


def seed_sample_patients_and_screenings(conn):
    """Seeds a representative rural cohort to demonstrate dashboard dynamics and referral analytics."""
    cursor = conn.cursor()
    
    sample_patients = [
        ('DR-2026-0001', 'Basavaraj Gowda', 58, 'Male', '+91 98451 23410', 'Shirur', 'Kalaburagi', 'Karnataka', 'Type 2', 9.5, '2 years ago', 184.0, 'Complaining of mild blurring in right eye when reading.'),
        ('DR-2026-0002', 'Laxmibai Kulkarni', 63, 'Female', '+91 94482 77192', 'Kamalapur', 'Kalaburagi', 'Karnataka', 'Type 2', 14.0, 'Never', 220.0, 'Uncontrolled sugar levels, high HbA1c, notices dark floating spots.'),
        ('DR-2026-0003', 'Mallikarjun Patil', 47, 'Male', '+91 99014 55321', 'Mahagaon', 'Kalaburagi', 'Karnataka', 'Type 2', 3.0, '1 year ago', 138.0, 'Annual routine screening at PHC. No vision issues.'),
        ('DR-2026-0004', 'Kavitha S. Rathod', 52, 'Female', '+91 97312 88410', 'Tandur Cross', 'Kalaburagi', 'Karnataka', 'Type 2', 7.0, 'Never', 165.0, 'Frequent headaches, elevated fasting blood sugar.'),
        ('DR-2026-0005', 'Ramanna Biradar', 71, 'Male', '+91 98801 99234', 'Sedam', 'Kalaburagi', 'Karnataka', 'Type 2', 18.0, '3 years ago', 245.0, 'Significant visual disturbance, difficulty with daily agricultural chores.')
    ]

    cursor.executemany(
        """INSERT INTO patients (patient_id, name, age, gender, mobile, village, district, state, diabetes_history, duration_diabetes_years, previous_eye_exam, blood_glucose_level, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        sample_patients
    )
    conn.commit()

    # Retrieve patient IDs to link sample screening records
    cursor.execute("SELECT id, patient_id FROM patients ORDER BY id ASC")
    p_rows = cursor.fetchall()
    p_dict = {row['patient_id']: row['id'] for row in p_rows}

    # Sample screening records
    sample_screenings = [
        (
            'SCR-2026-0101', p_dict['DR-2026-0001'], 2, 'Right',
            'static/uploads/sample_mild.jpg', 'static/uploads/sample_mild_proc.jpg', 'static/uploads/sample_mild_gradcam.jpg', 'static/uploads/sample_mild_cv.jpg',
            'Pass', 94.2, json.dumps({'focus': 'Pass', 'illumination': 'Pass', 'fov': 'Pass', 'blur_score': 320.5}),
            1, 'Mild NPDR', 0.88, json.dumps([0.08, 0.88, 0.03, 0.01, 0.00]),
            0, 'Low', Config.DR_CLASSES[1]['recommendation_en'], 'demo_calibrated', 'Mild microaneurysms detected in temporal macula.', 'static/uploads/SCR-2026-0101_report.pdf'
        ),
        (
            'SCR-2026-0102', p_dict['DR-2026-0002'], 2, 'Left',
            'static/uploads/sample_severe.jpg', 'static/uploads/sample_severe_proc.jpg', 'static/uploads/sample_severe_gradcam.jpg', 'static/uploads/sample_severe_cv.jpg',
            'Pass', 91.0, json.dumps({'focus': 'Pass', 'illumination': 'Pass', 'fov': 'Pass', 'blur_score': 295.1}),
            3, 'Severe NPDR', 0.92, json.dumps([0.01, 0.02, 0.05, 0.92, 0.00]),
            1, 'Urgent (Referral)', Config.DR_CLASSES[3]['recommendation_en'], 'demo_calibrated', 'Multiple blot hemorrhages in all 4 quadrants.', 'static/uploads/SCR-2026-0102_report.pdf'
        ),
        (
            'SCR-2026-0103', p_dict['DR-2026-0003'], 2, 'Right',
            'static/uploads/sample_normal.jpg', 'static/uploads/sample_normal_proc.jpg', 'static/uploads/sample_normal_gradcam.jpg', 'static/uploads/sample_normal_cv.jpg',
            'Pass', 96.5, json.dumps({'focus': 'Pass', 'illumination': 'Pass', 'fov': 'Pass', 'blur_score': 410.0}),
            0, 'No DR', 0.96, json.dumps([0.96, 0.03, 0.01, 0.00, 0.00]),
            0, 'Routine', Config.DR_CLASSES[0]['recommendation_en'], 'demo_calibrated', 'Clear retina, sharp disc margins, no signs of retinopathy.', 'static/uploads/SCR-2026-0103_report.pdf'
        ),
        (
            'SCR-2026-0104', p_dict['DR-2026-0004'], 2, 'Right',
            'static/uploads/sample_mod.jpg', 'static/uploads/sample_mod_proc.jpg', 'static/uploads/sample_mod_gradcam.jpg', 'static/uploads/sample_mod_cv.jpg',
            'Pass', 89.4, json.dumps({'focus': 'Pass', 'illumination': 'Pass', 'fov': 'Pass', 'blur_score': 270.8}),
            2, 'Moderate NPDR', 0.85, json.dumps([0.03, 0.09, 0.85, 0.02, 0.01]),
            1, 'Moderate (Referral)', Config.DR_CLASSES[2]['recommendation_en'], 'demo_calibrated', 'Hard exudate cluster visible near superior arcade.', 'static/uploads/SCR-2026-0104_report.pdf'
        ),
        (
            'SCR-2026-0105', p_dict['DR-2026-0005'], 2, 'Both',
            'static/uploads/sample_pdr.jpg', 'static/uploads/sample_pdr_proc.jpg', 'static/uploads/sample_pdr_gradcam.jpg', 'static/uploads/sample_pdr_cv.jpg',
            'Pass', 88.0, json.dumps({'focus': 'Pass', 'illumination': 'Pass', 'fov': 'Pass', 'blur_score': 240.2}),
            4, 'Proliferative DR', 0.94, json.dumps([0.00, 0.01, 0.02, 0.03, 0.94]),
            1, 'Critical / Emergency', Config.DR_CLASSES[4]['recommendation_en'], 'demo_calibrated', 'Neovascularization of the disc (NVD) with vitreous traction risk.', 'static/uploads/SCR-2026-0105_report.pdf'
        )
    ]

    cursor.executemany(
        """INSERT INTO screenings (
            screening_id, patient_id, user_id, eye_side,
            original_image_path, processed_image_path, gradcam_image_path, cv_analysis_path,
            quality_status, quality_score, quality_details,
            predicted_grade, predicted_class_name, confidence_score, class_probabilities,
            is_referable, referral_urgency, clinical_recommendation,
            model_mode, screening_notes, pdf_report_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        sample_screenings
    )
    conn.commit()
    print("[DB] Seeded initial rural patient screening records.")
