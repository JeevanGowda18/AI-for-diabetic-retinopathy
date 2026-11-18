"""
About the Project & AI Explainability Educational Routes
Documents the rural PHC healthcare mission, architecture, clinical referral protocols,
and visual explainability principles.
"""

from flask import Blueprint, render_template, session, redirect, request
from routes.auth import login_required
from models.model_loader import ModelService
from config import Config

about_bp = Blueprint('about', __name__)

@about_bp.route('/about')
@login_required
def about_page():
    return render_template('about.html')

@about_bp.route('/explainability')
@login_required
def explainability_page():
    model_service = ModelService.get_instance()
    model_info = model_service.get_info()
    return render_template('explainability.html', model_info=model_info, dr_classes=Config.DR_CLASSES)

@about_bp.route('/set-language/<lang>')
def set_language(lang):
    if lang in ['en', 'kn', 'hi']:
        session['lang'] = lang
    return redirect(request.referrer or '/')
