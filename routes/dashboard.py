"""
Dashboard Route for DR Vision AI
Aggregates dynamic clinical statistics directly from SQLite database.
"""

from flask import Blueprint, render_template, jsonify, session
from database.db import get_db_connection
from routes.auth import login_required
from config import Config

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
@login_required
def index():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Core Summary Metrics
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings")
    total_screenings = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE predicted_grade = 0")
    normal_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE predicted_grade > 0")
    positive_cases = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE is_referable = 1 OR predicted_grade >= 2")
    referable_cases = cursor.fetchone()[0]

    # 2. DR Severity Level Breakdown (5-class distribution)
    cursor.execute("""
        SELECT predicted_grade, COUNT(*) as count 
        FROM screenings 
        WHERE predicted_grade IS NOT NULL 
        GROUP BY predicted_grade 
        ORDER BY predicted_grade ASC
    """)
    distribution_rows = cursor.fetchall()
    
    # Initialize all 5 grades with 0 count
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for row in distribution_rows:
        if row['predicted_grade'] in grade_counts:
            grade_counts[row['predicted_grade']] = row['count']

    # 3. Recent Screening Activity (last 8 records with patient details)
    cursor.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as patient_code, p.village
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        ORDER BY s.created_at DESC
        LIMIT 8
    """)
    recent_screenings = cursor.fetchall()

    # 4. Recent Patient Registrations (last 5)
    cursor.execute("""
        SELECT * FROM patients
        ORDER BY created_at DESC
        LIMIT 5
    """)
    recent_patients = cursor.fetchall()

    # 5. Screening Trends by Month / Day
    cursor.execute("""
        SELECT DATE(created_at) as screening_date, COUNT(*) as daily_count
        FROM screenings
        GROUP BY DATE(created_at)
        ORDER BY screening_date DESC
        LIMIT 7
    """)
    trend_rows = cursor.fetchall()
    trend_data = [{'date': r['screening_date'], 'count': r['daily_count']} for r in trend_rows][::-1]

    # 6. Doctor Review Queue (Pending Screenings awaiting Clinical Verification - F-06)
    cursor.execute("""
        SELECT s.*, p.name as patient_name, p.patient_id as patient_code, p.village, p.age, p.gender
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.verification_status = 'Pending' OR s.verification_status IS NULL
        ORDER BY s.created_at DESC
        LIMIT 10
    """)
    pending_reviews = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE verification_status = 'Pending' OR verification_status IS NULL")
    pending_count = cursor.fetchone()[0]

    conn.close()

    metrics = {
        'total_patients': total_patients,
        'total_screenings': total_screenings,
        'normal_cases': normal_cases,
        'positive_cases': positive_cases,
        'referable_cases': referable_cases,
        'referral_rate': round((referable_cases / total_screenings * 100), 1) if total_screenings > 0 else 0.0,
        'pending_count': pending_count
    }

    return render_template(
        'dashboard.html',
        metrics=metrics,
        grade_counts=grade_counts,
        recent_screenings=recent_screenings,
        recent_patients=recent_patients,
        trend_data=trend_data,
        pending_reviews=pending_reviews,
        dr_classes=Config.DR_CLASSES
    )

@dashboard_bp.route('/api/dashboard-stats')
@login_required
def api_stats():
    """Provides dynamic JSON for client-side chart refreshing."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT predicted_grade, COUNT(*) as count 
        FROM screenings 
        WHERE predicted_grade IS NOT NULL 
        GROUP BY predicted_grade 
        ORDER BY predicted_grade ASC
    """)
    rows = cursor.fetchall()
    
    grade_counts = [0, 0, 0, 0, 0]
    for r in rows:
        g = r['predicted_grade']
        if 0 <= g <= 4:
            grade_counts[g] = r['count']

    conn.close()
    return jsonify({
        'labels': [Config.DR_CLASSES[i]['name'] for i in range(5)],
        'counts': grade_counts
    })
