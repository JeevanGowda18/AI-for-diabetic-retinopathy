"""
PDF Report Generator for DR Vision AI
Uses ReportLab to generate clinical screening reports for Primary Health Centres.
Includes patient demographics, fundus photography, Grad-CAM overlays,
quantitative AI confidence, referral protocols, and statutory medical disclaimers.
"""

import os
from pathlib import Path
from datetime import datetime
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from config import Config

def generate_screening_pdf(screening_data, patient_data, output_path):
    """
    Generates a medical-grade PDF screening summary.
    Args:
        screening_data: dict containing screening attributes, image paths, predictions, quality
        patient_data: dict containing patient demographics and history
        output_path: file path to save PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor('#0F2B48')     # Dark Navy
    teal_color = colors.HexColor('#0D9488')        # Teal
    accent_green = colors.HexColor('#16A34A')      # Medical green
    alert_red = colors.HexColor('#DC2626')         # Warning red
    gray_bg = colors.HexColor('#F8FAFC')           # Off-white
    border_color = colors.HexColor('#CBD5E1')      # Border slate

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569')
    )

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )

    bold_body_style = ParagraphStyle(
        'DocBodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#64748B')
    )

    story = []

    # -------------------------------------------------------------
    # 1. Header with Hospital / PHC Branding
    # -------------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>DR VISION AI</b><br/><font size='8'>Rural Tele-Ophthalmology & AI Screening Initiative</font>", title_style),
            Paragraph(
                f"<b>{Config.PHC_DEFAULT['center_name']}</b><br/>"
                f"District: {Config.PHC_DEFAULT['district']}, {Config.PHC_DEFAULT['state']}<br/>"
                f"Date: {datetime.now().strftime('%d-%b-%Y %H:%M')}",
                subtitle_style
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[3.2 * inch, 4.0 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=teal_color, spaceAfter=8))

    # -------------------------------------------------------------
    # 2. Patient Demographics & Clinical History
    # -------------------------------------------------------------
    story.append(Paragraph("PATIENT DEMOGRAPHICS & CLINICAL HISTORY", section_header_style))
    
    pat_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style),
            Paragraph(str(patient_data.get('name', 'N/A')), bold_body_style),
            Paragraph("<b>Patient ID:</b>", body_style),
            Paragraph(str(patient_data.get('patient_id', 'N/A')), bold_body_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_style),
            Paragraph(f"{patient_data.get('age', 'N/A')} yrs / {patient_data.get('gender', 'N/A')}", body_style),
            Paragraph("<b>Mobile:</b>", body_style),
            Paragraph(str(patient_data.get('mobile', 'N/A')), body_style)
        ],
        [
            Paragraph("<b>Village / Town:</b>", body_style),
            Paragraph(f"{patient_data.get('village', 'N/A')}, {patient_data.get('district', 'N/A')}", body_style),
            Paragraph("<b>Screening ID:</b>", body_style),
            Paragraph(str(screening_data.get('screening_id', 'N/A')), bold_body_style)
        ],
        [
            Paragraph("<b>Diabetes History:</b>", body_style),
            Paragraph(f"{patient_data.get('diabetes_history', 'N/A')} ({patient_data.get('duration_diabetes_years', 0)} yrs)", body_style),
            Paragraph("<b>Blood Glucose:</b>", body_style),
            Paragraph(f"{patient_data.get('blood_glucose_level', 'Not recorded')} mg/dL", body_style)
        ]
    ]

    pat_table = Table(pat_table_data, colWidths=[1.4 * inch, 2.2 * inch, 1.4 * inch, 2.2 * inch])
    pat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 3. Retinal Fundus & Grad-CAM Visualizations
    # -------------------------------------------------------------
    story.append(Paragraph(f"RETINAL FUNDUS IMAGING & GRAD-CAM EXPLAINABILITY (Eye: {screening_data.get('eye_side', 'Right')})", section_header_style))

    img_cells = []
    labels = []

    def load_flowable_img(img_path):
        if img_path and os.path.exists(img_path):
            try:
                return Image(img_path, width=2.25 * inch, height=2.25 * inch)
            except Exception:
                pass
        return Paragraph("<font color='gray'>[Image Unavailable]</font>", subtitle_style)

    img_orig = load_flowable_img(screening_data.get('original_image_path'))
    img_proc = load_flowable_img(screening_data.get('processed_image_path'))
    img_cam  = load_flowable_img(screening_data.get('gradcam_image_path'))

    images_row = [img_orig, img_proc, img_cam]
    labels_row = [
        Paragraph("<para align='center'><b>Original Fundus</b></para>", subtitle_style),
        Paragraph("<para align='center'><b>Enhanced (LAB CLAHE)</b></para>", subtitle_style),
        Paragraph("<para align='center'><b>Grad-CAM Attention Map</b></para>", subtitle_style)
    ]

    img_table = Table([images_row, labels_row], colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 4. Diagnostic Assessment & Quantitative Findings
    # -------------------------------------------------------------
    story.append(Paragraph("AI SCREENING ASSESSMENT & PREDICTION", section_header_style))

    grade = screening_data.get('predicted_grade', 0)
    grade_info = Config.DR_CLASSES.get(grade, Config.DR_CLASSES[0])
    confidence = float(screening_data.get('confidence_score', 0.0)) * 100
    quality_score = float(screening_data.get('quality_score', 0.0))
    quality_status = screening_data.get('quality_status', 'Pass')
    referral_urgency = screening_data.get('referral_urgency', grade_info['referral_urgency'])
    model_mode = screening_data.get('model_mode', 'demo_calibrated')

    findings_data = [
        [
            Paragraph("<b>Predicted Severity:</b>", body_style),
            Paragraph(f"<font size='10' color='{primary_color.hexval()}'><b>{grade_info['name']} - {grade_info['severity']}</b></font>", body_style),
            Paragraph("<b>Model Confidence:</b>", body_style),
            Paragraph(f"<b>{confidence:.1f}%</b>", body_style)
        ],
        [
            Paragraph("<b>Image Quality Check:</b>", body_style),
            Paragraph(f"<b>{quality_status}</b> (Score: {quality_score:.1f} / 100)", body_style),
            Paragraph("<b>Referral Urgency:</b>", body_style),
            Paragraph(f"<b>{referral_urgency}</b>", body_style)
        ],
        [
            Paragraph("<b>Model Status:</b>", body_style),
            Paragraph(f"{'Demonstration / Workflow Testing' if model_mode == 'demo_calibrated' else 'Clinical Model Active'}", body_style),
            Paragraph("<b>Eye Screened:</b>", body_style),
            Paragraph(f"{screening_data.get('eye_side', 'Right')} Eye", body_style)
        ]
    ]

    findings_table = Table(findings_data, colWidths=[1.6 * inch, 2.0 * inch, 1.6 * inch, 2.0 * inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------
    # 5. Probability Distribution Bar
    # -------------------------------------------------------------
    probs = screening_data.get('class_probabilities', [0.0, 0.0, 0.0, 0.0, 0.0])
    if isinstance(probs, str):
        try:
            probs = json.loads(probs)
        except Exception:
            probs = [0.0, 0.0, 0.0, 0.0, 0.0]

    prob_headers = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (PDR)"]
    prob_values = [f"{float(p)*100:.1f}%" for p in probs]

    prob_table_data = [
        [Paragraph(f"<para align='center'><b>{h}</b></para>", subtitle_style) for h in prob_headers],
        [Paragraph(f"<para align='center'><b>{v}</b></para>", body_style) for v in prob_values]
    ]
    prob_table = Table(prob_table_data, colWidths=[1.44 * inch] * 5)
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 1), (-1, 1), gray_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 6. Clinical Referral Recommendation & Action Plan
    # -------------------------------------------------------------
    story.append(Paragraph("CLINICAL RECOMMENDATION & REFERRAL PROTOCOL", section_header_style))
    
    rec_text = screening_data.get('clinical_recommendation') or grade_info['recommendation_en']
    rec_p = Paragraph(f"<b>Recommended Clinical Action:</b> {rec_text}", body_style)
    
    rec_table = Table([[rec_p]], colWidths=[7.2 * inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')), # Light blue highlight
        ('BOX', (0, 0), (-1, -1), 1.0, teal_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 7. Signature & Statutory Medical Disclaimer
    # -------------------------------------------------------------
    sig_data = [
        [
            Paragraph(f"<b>Examined By:</b> {Config.PHC_DEFAULT['operator_role']}", subtitle_style),
            Paragraph("<b>Medical Officer / Ophthalmologist Signature:</b>", subtitle_style)
        ],
        [
            Paragraph("Shirur Primary Health Centre", subtitle_style),
            Paragraph("___________________________________________", subtitle_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[3.6 * inch, 3.6 * inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 6))

    disclaimer_text = (
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> This report is generated by DR Vision AI as a triage and screening decision-support aid "
        "designed for Primary Health Centres (PHCs). This output DOES NOT constitute a definitive clinical diagnosis or treatment prescription. "
        "Visual attention heatmaps (Grad-CAM) depict mathematical saliency in deep feature maps and must not be interpreted as confirmed histological pathology. "
        "Patients with suspected or referable retinopathy must be evaluated by a certified ophthalmologist with slit-lamp bio-microscopy and indirect ophthalmoscopy."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build PDF
    doc.build(story)
    print(f"[PDF] Successfully generated screening report: {output_path}")
    return str(output_path)
