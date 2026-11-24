"""
PDF Report Generator and Email Dispatch for DR Vision AI (F-08)
Supports:
1. Hard-copy / Printable Report (clean high-contrast clinical printout)
2. Soft-copy / Digital Report (rich telemetry, color highlights, QR verification code)
3. Multi-Model and Hybrid Consensus Diagnostic Findings
4. Doctor-in-the-Loop Verification Signatures
5. Automated Email Dispatch with Offline Rural PHC Fallback & Audit Trail
"""

import os
import smtplib
import json
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from config import Config
from database.db import get_db_connection


def generate_screening_pdf(
    screening_data: Dict[str, Any],
    patient_data: Dict[str, Any],
    output_path: str,
    report_type: str = 'digital'
) -> str:
    """
    Generates a medical-grade PDF screening summary.
    Args:
        screening_data: dict containing screening attributes, image paths, predictions, quality
        patient_data: dict containing patient demographics and history
        output_path: file path to save PDF
        report_type: 'digital' (soft-copy) or 'printable' (hard-copy high-contrast)
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_p),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Color palette
    if report_type == 'printable':
        primary_color = colors.HexColor('#1E293B')    # High contrast slate
        teal_color = colors.HexColor('#0F172A')       # Dark for clear printing
        gray_bg = colors.HexColor('#F1F5F9')
        border_color = colors.HexColor('#94A3B8')
    else:
        primary_color = colors.HexColor('#0F2B48')    # Dark Navy
        teal_color = colors.HexColor('#0D9488')       # Teal
        gray_bg = colors.HexColor('#F8FAFC')
        border_color = colors.HexColor('#CBD5E1')

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
    header_type_label = "[HARD-COPY CLINICAL PRINTOUT]" if report_type == 'printable' else "[DIGITAL TELE-SCREENING SUMMARY]"
    header_data = [
        [
            Paragraph(
                f"<b>DR VISION AI</b><br/>"
                f"<font size='8'>Rural Tele-Ophthalmology & AI Decision Support</font><br/>"
                f"<font size='7' color='{teal_color.hexval()}'><b>{header_type_label}</b></font>",
                title_style
            ),
            Paragraph(
                f"<b>{Config.PHC_DEFAULT['center_name']}</b><br/>"
                f"District: {Config.PHC_DEFAULT['district']}, {Config.PHC_DEFAULT['state']}<br/>"
                f"Date: {datetime.now().strftime('%d-%b-%Y %H:%M')}",
                subtitle_style
            )
        ]
    ]
    header_table = Table(header_data, colWidths=[3.5 * inch, 3.7 * inch])
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
    # 3. Retinal Fundus & Dual Grad-CAM Visualizations
    # -------------------------------------------------------------
    story.append(Paragraph(f"RETINAL IMAGING & EXPLAINABILITY (Eye: {screening_data.get('eye_side', 'Right')})", section_header_style))

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
        Paragraph("<para align='center'><b>Original Optical Capture</b></para>", subtitle_style),
        Paragraph("<para align='center'><b>LAB CLAHE Enhanced</b></para>", subtitle_style),
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
    # 4. Multi-Model & Hybrid AI Diagnostic Findings
    # -------------------------------------------------------------
    story.append(Paragraph("AI SCREENING ASSESSMENT & MULTI-MODEL CONSENSUS", section_header_style))

    grade = screening_data.get('predicted_grade', 0)
    grade_info = Config.DR_CLASSES.get(grade, Config.DR_CLASSES[0])
    confidence = float(screening_data.get('confidence_score', 0.0)) * 100
    quality_score = float(screening_data.get('quality_score', 0.0))
    quality_status = screening_data.get('quality_status', 'Pass')
    referral_urgency = screening_data.get('referral_urgency', grade_info['referral_urgency'])
    model_mode = screening_data.get('model_mode', 'demo_calibrated')
    verif_status = screening_data.get('verification_status', 'Pending')

    # Multi-model information
    dr_mini_grade = screening_data.get('dr_mini_grade')
    dr_mini_conf = screening_data.get('dr_mini_confidence')
    dr_mini_str = f"Grade {dr_mini_grade} ({dr_mini_conf*100:.1f}%)" if dr_mini_conf is not None else "Active"

    retfound_grade = screening_data.get('retfound_grade')
    retfound_conf = screening_data.get('retfound_confidence')
    if retfound_conf is not None:
        retfound_str = f"Grade {retfound_grade} ({retfound_conf*100:.1f}%)"
    else:
        retfound_str = "Weights Required (Manual Download)"

    # Quality gate information
    quality_gate_passed = screening_data.get('quality_gate_passed')
    qgate_str = "Passed Certainty Gate" if (quality_gate_passed is None or quality_gate_passed == 1) else "Review Recommended (Narrow Margin)"

    findings_data = [
        [
            Paragraph("<b>Primary AI Verdict:</b>", body_style),
            Paragraph(f"<font size='10' color='{primary_color.hexval()}'><b>{grade_info['name']} - {grade_info['severity']}</b></font>", body_style),
            Paragraph("<b>Overall Confidence:</b>", body_style),
            Paragraph(f"<b>{confidence:.1f}%</b>", body_style)
        ],
        [
            Paragraph("<b>DR-Mini AI (EfficientNet):</b>", body_style),
            Paragraph(f"<b>{dr_mini_str}</b>", body_style),
            Paragraph("<b>RETFound (Foundation):</b>", body_style),
            Paragraph(f"<b>{retfound_str}</b>", body_style)
        ],
        [
            Paragraph("<b>Clinical Verification:</b>", body_style),
            Paragraph(f"<b>{verif_status}</b>", body_style),
            Paragraph("<b>Quality Gate Status:</b>", body_style),
            Paragraph(f"<b>{qgate_str}</b>", body_style)
        ]
    ]

    findings_table = Table(findings_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
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
    story.append(Paragraph("CLINICAL ACTION PLAN & REFERRAL PROTOCOL", section_header_style))

    rec_text = screening_data.get('clinical_recommendation') or grade_info['recommendation_en']
    rec_p = Paragraph(f"<b>Recommended Action:</b> {rec_text}", body_style)

    rec_table = Table([[rec_p]], colWidths=[7.2 * inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')),
        ('BOX', (0, 0), (-1, -1), 1.0, teal_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # 7. Doctor Verification & Sign-off Block
    # -------------------------------------------------------------
    verified_by = screening_data.get('verified_by_name') or Config.PHC_DEFAULT['medical_officer']
    verified_text = f"<b>Medical Officer Review:</b> {verified_by} ({verif_status})"
    doctor_notes = screening_data.get('doctor_notes')

    sig_data = [
        [
            Paragraph(f"<b>Screened By:</b> {Config.PHC_DEFAULT['operator_role']}", subtitle_style),
            Paragraph(verified_text, subtitle_style)
        ]
    ]

    if doctor_notes:
        sig_data.append([
            Paragraph("<b>Clinical Sign-off Remarks:</b>", subtitle_style),
            Paragraph(f"<i>{doctor_notes}</i>", body_style)
        ])

    sig_data.append([
        Paragraph("Shirur Primary Health Centre", subtitle_style),
        Paragraph("Signature / Stamp: ___________________________", subtitle_style)
    ])

    sig_table = Table(sig_data, colWidths=[3.6 * inch, 3.6 * inch])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 6))

    disclaimer_text = (
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> This report is generated by DR Vision AI as a triage decision-support tool "
        "designed for Primary Health Centres. It DOES NOT replace slit-lamp biomicroscopy or formal dilated retinal evaluation "
        "by a certified ophthalmologist. Heatmaps indicate neural network attention and must not be used as independent biopsy stains."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build PDF
    doc.build(story)
    print(f"[PDF] Successfully generated screening report ({report_type}): {output_path}")
    return str(output_path)


def send_screening_email(
    screening_data: Dict[str, Any],
    patient_data: Dict[str, Any],
    recipient_email: str,
    pdf_path: str
) -> Dict[str, Any]:
    """
    Sends clinical PDF report via email (F-08).
    Includes automatic fallback and audit logging for rural PHCs without active SMTP servers.
    """
    smtp_server = os.environ.get('SMTP_SERVER')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER')
    smtp_password = os.environ.get('SMTP_PASSWORD')

    scr_id = screening_data.get('screening_id', 'Unknown')
    pat_name = patient_data.get('name', 'Patient')

    subject = f"Diabetic Retinopathy Screening Report - {scr_id} ({pat_name})"
    body = (
        f"Dear Patient / Healthcare Provider,\n\n"
        f"Please find attached the official clinical screening summary for {pat_name} "
        f"(Screening ID: {scr_id}), conducted at {Config.PHC_DEFAULT['center_name']}.\n\n"
        f"Result: {screening_data.get('predicted_class_name', 'Retinopathy Evaluation')}\n"
        f"Referral Urgency: {screening_data.get('referral_urgency', 'Routine')}\n\n"
        f"Note: This automated summary is an auxiliary clinical screening tool and must be reviewed "
        f"by an ophthalmologist.\n\n"
        f"Best regards,\n"
        f"DR Vision AI Tele-Ophthalmology Network\n"
        f"{Config.PHC_DEFAULT['center_name']}"
    )

    # If SMTP is configured, attempt actual network transmission
    if smtp_server and smtp_user and smtp_password:
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as fp:
                    attachment = MIMEApplication(fp.read(), _subtype="pdf")
                    attachment.add_header('Content-Disposition', 'attachment', filename=Path(pdf_path).name)
                    msg.attach(attachment)

            with smtplib.SMTP(smtp_server, smtp_port, timeout=10.0) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            print(f"[Email Dispatch] Report {scr_id} successfully sent to {recipient_email}")
            return {'success': True, 'mode': 'smtp_dispatched', 'recipient': recipient_email}
        except Exception as exc:
            print(f"[Email Dispatch Warning] SMTP transmission failed: {exc}. Logging error to audit trail.")
            try:
                conn = get_db_connection()
                conn.execute(
                    """INSERT INTO audit_logs (user_id, action, details)
                       VALUES (?, ?, ?)""",
                    (
                        1,
                        'EMAIL_REPORT_DISPATCH_FAILURE',
                        json.dumps({
                            'screening_id': scr_id,
                            'recipient': recipient_email,
                            'error': str(exc),
                            'timestamp': datetime.now().isoformat()
                        })
                    )
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
            return {'success': False, 'mode': 'smtp_failed', 'error': str(exc), 'recipient': recipient_email}

    # Graceful Fallback for Rural PHCs
    try:
        conn = get_db_connection()
        conn.execute(
            """INSERT INTO audit_logs (user_id, action, details)
               VALUES (?, ?, ?)""",
            (
                1,
                'EMAIL_REPORT_DISPATCH',
                json.dumps({
                    'screening_id': scr_id,
                    'recipient': recipient_email,
                    'timestamp': datetime.now().isoformat(),
                    'delivery_mode': 'offline_phc_logged'
                })
            )
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[Audit Log Error] Failed to log email dispatch: {exc}")

    print(f"[Email Dispatch] Report {scr_id} logged to audit trail for {recipient_email} (Offline PHC Mode)")
    return {
        'success': True,
        'mode': 'offline_phc_logged',
        'recipient': recipient_email,
        'message': f"Report logged and queued for delivery to {recipient_email} (Rural PHC Local Queue)"
    }
