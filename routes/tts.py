"""
Multilingual Text-to-Speech (TTS) Module for DR Vision AI (F-05)
Generates spoken clinical explanations in English, Kannada, and Hindi
for rural patients and Primary Health Centre health workers.
"""

import os
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file, url_for, session
from gtts import gTTS

from database.db import get_db_connection
from config import Config

tts_bp = Blueprint('tts', __name__)

TTS_AUDIO_DIR = Config.UPLOAD_FOLDER / 'audio'


def get_audio_text_for_screening(screening, patient, lang='en') -> str:
    """Builds localized spoken clinical explanation for the patient."""
    grade = screening['predicted_grade'] if screening['predicted_grade'] is not None else 0
    grade_info = Config.DR_CLASSES.get(grade, Config.DR_CLASSES[0])
    p_name = patient['name']
    eye = "Right eye" if screening['eye_side'] == 'Right' else "Left eye"

    if lang == 'kn':
        # Kannada spoken explanation
        eye_kn = "ಬಲ ಕಣ್ಣು" if screening['eye_side'] == 'Right' else "ಎಡ ಕಣ್ಣು"
        rec = grade_info.get('recommendation_kn', grade_info['recommendation_en'])
        text = (
            f"ನಮಸ್ಕಾರ {p_name}. ನಿಮ್ಮ {eye_kn} ತಪಾಸಣೆ ಪೂರ್ಣಗೊಂಡಿದೆ. "
            f"ರೋಗ ನಿರ್ಣಯ: {grade_info['name']}. "
            f"ವೈದ್ಯಕೀಯ ಸಲಹೆ: {rec}"
        )
    elif lang == 'hi':
        # Hindi spoken explanation
        eye_hi = "दाहिनी आँख" if screening['eye_side'] == 'Right' else "बाईं आँख"
        rec = grade_info.get('recommendation_hi', grade_info['recommendation_en'])
        text = (
            f"नमस्ते {p_name}। आपकी {eye_hi} की जांच पूरी हो गई है। "
            f"निदान: {grade_info['name']}। "
            f"चिकित्सा सलाह: {rec}"
        )
    else:
        # English spoken explanation
        rec = grade_info.get('recommendation_en')
        text = (
            f"Hello {p_name}. Screening for your {eye} is complete. "
            f"Diagnosis: {grade_info['name']}, {grade_info['severity']}. "
            f"Medical recommendation: {rec}"
        )

    return text


@tts_bp.route('/api/tts/<screening_id>')
def generate_or_get_tts(screening_id):
    """
    Returns audio URL for spoken explanation in requested language.
    Query param: ?lang=en|kn|hi (defaults to session language or English).
    """
    lang = request.args.get('lang', session.get('lang', 'en'))
    if lang not in ('en', 'kn', 'hi'):
        lang = 'en'

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, p.name as patient_name
        FROM screenings s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.screening_id = ?
    """, (screening_id,))
    screening = cursor.fetchone()
    conn.close()

    if not screening:
        return jsonify({'success': False, 'error': 'Screening record not found'}), 404

    TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_filename = f"tts_{screening_id}_{lang}.mp3"
    audio_path = TTS_AUDIO_DIR / audio_filename

    spoken_text = get_audio_text_for_screening(
        dict(screening),
        {'name': screening['patient_name']},
        lang=lang
    )

    # Generate audio if not already cached
    if not audio_path.exists():
        try:
            tts = gTTS(text=spoken_text, lang=lang, slow=False)
            tts.save(str(audio_path))
        except Exception as exc:
            print(f"[TTS Error] gTTS generation failed ({exc}). Supplying text for Web Speech API fallback.")
            return jsonify({
                'success': True,
                'fallback': True,
                'lang': lang,
                'spoken_text': spoken_text,
                'audio_url': None,
                'message': 'Server offline TTS fallback; client Web Speech API recommended.'
            })

    audio_url = url_for('static', filename=f'uploads/audio/{audio_filename}')
    return jsonify({
        'success': True,
        'fallback': False,
        'lang': lang,
        'spoken_text': spoken_text,
        'audio_url': audio_url
    })
