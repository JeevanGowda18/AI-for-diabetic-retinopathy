"""
DR Vision AI - Application Entry Point & Flask Server
Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, session, redirect, url_for, flash, request

from config import Config
from database.db import init_db
from routes.auth import auth_bp
from routes.dashboard import dashboard_bp
from routes.patients import patients_bp
from routes.screening import screening_bp
from routes.reports import reports_bp
from routes.about import about_bp
from routes.settings import settings_bp
from routes.tts import tts_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure required runtime folders exist
    Config.INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Initialize Database Schema & Seeding
    with app.app_context():
        init_db()

    # Register Route Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(screening_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(about_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(tts_bp)


    # Load Multilingual Translations
    translations = {}
    trans_dir = Config.BASE_DIR / 'translations'
    for lang in ['en', 'kn', 'hi']:
        fpath = trans_dir / f"{lang}.json"
        if fpath.exists():
            with open(fpath, 'r', encoding='utf-8') as f:
                translations[lang] = json.load(f)
        else:
            translations[lang] = {}

    @app.context_processor
    def inject_global_context():
        """Injects translation dictionary, language code, and PHC info into all templates."""
        current_lang = session.get('lang', 'en')
        t_dict = translations.get(current_lang, translations.get('en', {}))
        
        # Fallback to English if key missing in translation
        en_dict = translations.get('en', {})
        class TranslationWrapper:
            def __init__(self, primary, fallback):
                self.primary = primary
                self.fallback = fallback
            def __getitem__(self, key):
                return self.primary.get(key, self.fallback.get(key, key))
            def get(self, key, default=None):
                return self.primary.get(key, self.fallback.get(key, default or key))

        return {
            't': TranslationWrapper(t_dict, en_dict),
            'current_lang': current_lang,
            'dr_classes': Config.DR_CLASSES,
            'phc_info': Config.PHC_DEFAULT,
            'app_version': '2.4.0 (Rural PHC Edition)'
        }

    # Template Filters
    @app.template_filter('format_datetime')
    def format_datetime(value, format_str='%d-%b-%Y %I:%M %p'):
        if not value:
            return 'N/A'
        try:
            # Handles SQLite string timestamps
            if isinstance(value, str):
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d'):
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.strftime(format_str)
                    except ValueError:
                        pass
                return value
            return value.strftime(format_str)
        except Exception:
            return str(value)

    # HTTP Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('base.html', error_code=404, error_message="The requested clinical page was not found."), 404

    @app.errorhandler(413)
    def file_too_large(e):
        flash("Uploaded retinal photograph exceeds the 16 MB maximum size limit. Please upload an optimized image.", "danger")
        return redirect(request.referrer or url_for('screening.new_screening'))

    @app.errorhandler(500)
    def server_error(e):
        return render_template('base.html', error_code=500, error_message="An internal clinical processing error occurred."), 500

    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n========================================================")
    print(f" DR VISION AI - RURAL PHC RETINAL SCREENING SYSTEM")
    print(f" Access URL: http://127.0.0.1:{port}")
    print(f" Multi-language: English | Kannada (KN) | Hindi (HI)")
    print(f" Default Login: admin / admin123  or  phc_worker / worker123")
    print(f"========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
