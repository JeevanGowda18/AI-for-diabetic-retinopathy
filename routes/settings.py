"""
Settings & System Diagnostics Route
Allows Primary Health Centre administrators to inspect AI model status,
review clinical thresholds, update PHC identity, and configure camera settings.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from routes.auth import login_required
from models.model_loader import ModelService
from config import Config

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def index():
    model_service = ModelService.get_instance()
    model_info = model_service.get_info()

    if request.method == 'POST':
        center_name = request.form.get('center_name')
        district = request.form.get('district')
        state = request.form.get('state')
        officer = request.form.get('medical_officer')
        role = request.form.get('operator_role')
        lang = request.form.get('language')

        if center_name:
            Config.PHC_DEFAULT['center_name'] = center_name.strip()
        if district:
            Config.PHC_DEFAULT['district'] = district.strip()
        if state:
            Config.PHC_DEFAULT['state'] = state.strip()
        if officer:
            Config.PHC_DEFAULT['medical_officer'] = officer.strip()
        if role:
            Config.PHC_DEFAULT['operator_role'] = role.strip()
        if lang in ['en', 'kn', 'hi']:
            session['lang'] = lang

        flash("PHC operational parameters updated successfully.", "success")
        return redirect(url_for('settings.index'))

    return render_template(
        'settings.html',
        model_info=model_info,
        phc_config=Config.PHC_DEFAULT,
        thresholds=Config.QUALITY_THRESHOLDS
    )
