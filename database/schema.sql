-- DR Vision AI Database Schema
-- Compatible with SQLite3, easily migratable to PostgreSQL or MySQL

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'health_worker', -- 'admin', 'doctor', 'health_worker'
    center_name VARCHAR(150) DEFAULT 'Rural Primary Health Centre',
    district VARCHAR(100) DEFAULT 'Kalaburagi',
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    gender VARCHAR(20) NOT NULL,
    mobile VARCHAR(20) NOT NULL,
    village VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    diabetes_history VARCHAR(20) NOT NULL, -- 'Type 1', 'Type 2', 'Gestational', 'Pre-diabetic', 'None'
    duration_diabetes_years REAL DEFAULT 0.0,
    previous_eye_exam VARCHAR(50) DEFAULT 'Never',
    blood_glucose_level REAL, -- Fasting or Random Blood Sugar in mg/dL
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id VARCHAR(50) UNIQUE NOT NULL,
    patient_id INTEGER NOT NULL,
    user_id INTEGER,
    eye_side VARCHAR(15) NOT NULL DEFAULT 'Right', -- 'Left', 'Right'
    original_image_path VARCHAR(255) NOT NULL,
    processed_image_path VARCHAR(255),
    gradcam_image_path VARCHAR(255),
    cv_analysis_path VARCHAR(255),
    quality_status VARCHAR(20) NOT NULL, -- 'Pass', 'Warning', 'Fail'
    quality_score REAL NOT NULL,
    quality_details TEXT, -- JSON breakdown of metrics
    predicted_grade INTEGER, -- 0 to 4 (or NULL if rejected by quality guard)
    predicted_class_name VARCHAR(60),
    confidence_score REAL,
    class_probabilities TEXT, -- JSON array of 5 floats [p0, p1, p2, p3, p4]
    is_referable INTEGER DEFAULT 0, -- 1 if Grade >= 2
    referral_urgency VARCHAR(50),
    clinical_recommendation TEXT,
    model_mode VARCHAR(40) DEFAULT 'demo_calibrated', -- 'production_trained' or 'demo_calibrated'
    screening_notes TEXT,
    pdf_report_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action VARCHAR(100) NOT NULL,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for rapid querying in rural clinical environments
CREATE INDEX IF NOT EXISTS idx_patients_pid ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(name);
CREATE INDEX IF NOT EXISTS idx_patients_village ON patients(village);
CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);
CREATE INDEX IF NOT EXISTS idx_screenings_date ON screenings(created_at);
CREATE INDEX IF NOT EXISTS idx_screenings_grade ON screenings(predicted_grade);
