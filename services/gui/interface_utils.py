"""
Interface Utilities for Arthritis Care Assistant
Handles PDF generation, styling, and report formatting.
"""

import os
import base64
import io
import tempfile
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import difflib

# Try importing reportlab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will use fallback.")

# ==========================
# STYLING
# ==========================
def apply_professional_style(theme="Day"):
    
    glass_bg = "rgba(255,255,255,0.1)" if theme == "Day" else "rgba(255,255,255,0.05)"
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    <style>
    /* Root Variables */
    :root {{
        --bg-primary: {'linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)' if theme=="Night" else '#f0f4f8'};
        --glass-bg: {glass_bg};
        --glass-border: rgba(255,255,255,0.2);
        --text-primary: {'#e0e6ed' if theme=="Night" else '#1e293b'};
        --accent-blue: #3b82f6;
        --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%);
        --shadow: 0 8px 32px rgba(31,38,135,0.37);
        --shadow-hover: 0 20px 40px rgba(31,38,135,0.4);
    }}
    
    .stApp {{
        font-family: 'Inter', sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        padding: 2rem;
    }}
    
    /* Glassmorphism Cards */
    .glass-card {{
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    /* Report Cards */
    .report-card {{ 
        background: var(--glass-bg); 
        border-left: 5px solid var(--accent-blue); 
        padding: 1.5rem; 
        border-radius: 16px; 
        box-shadow: var(--shadow);
    }}
    </style>
    """
    
# ==========================
# REPORT DATA GENERATION
# ==========================
def generate_report_data(module_name, prediction, confidence, extra=None):
    """Generate structured report data."""
    risk_levels = {
        "Low Risk": {"color": "#28a745", "icon": "✅"},
        "Medium Risk": {"color": "#ffc107", "icon": "⚠️"},
        "High Risk": {"color": "#dc3545", "icon": "🚨"}
    }
    
    # Determine risk from prediction value
    if isinstance(prediction, (int, float)):
        if prediction < 0.33:
            risk = "Low Risk"
        elif prediction < 0.66:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
        score = float(prediction)
    elif isinstance(prediction, str):
        risk = prediction if prediction in risk_levels else "Unknown"
        score = confidence if isinstance(confidence, (int, float)) else 0.0
    else:
        risk = "Unknown"
        score = 0.0
    
    return {
        "module": module_name,
        "prediction": prediction,
        "confidence": confidence,
        "risk": risk,
        "score": round(score, 3),
        "color": risk_levels.get(risk, {}).get("color", "#6c757d"),
        "icon": risk_levels.get(risk, {}).get("icon", "ℹ️"),
        "extra": extra,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ==========================
# HTML RENDERING
# ==========================
def render_report_html(report, patient_id="Anonymous"):
    """Render HTML report card with proper type handling."""
    extra_html = f"<p><b>Note:</b> {report['extra']}</p>" if report.get('extra') else ""
    
    # Handle confidence that might be string or number
    conf = report.get('confidence', 0)
    if isinstance(conf, str):
        conf_display = conf
    elif isinstance(conf, (int, float)):
        try:
            conf_display = f"{conf:.1%}"
        except:
            conf_display = str(conf)
    else:
        conf_display = str(conf)
    
    # Handle score that might be string or number
    score_val = report.get('score', 0)
    if isinstance(score_val, str):
        score_display = score_val
    elif isinstance(score_val, (int, float)):
        try:
            score_display = f"{score_val:.3f}"
        except:
            score_display = str(score_val)
    else:
        score_display = str(score_val)
    
    return f"""
    <div class="report-card">
        <h3>{report['icon']} {report['module']} Analysis</h3>
        <p><b>Patient ID:</b> {patient_id}</p>
        <p><b>Risk Level:</b> <span style="color:{report['color']};font-weight:bold;">{report['risk']}</span></p>
        <p><b>Risk Score:</b> {score_display}</p>
        <p><b>Confidence:</b> {conf_display}</p>
        <p><b>Timestamp:</b> {report['timestamp']}</p>
        {extra_html}
    </div>
    """

def render_doctor_card(doc, index):
    """Render doctor info as HTML."""
    return f"""
    <div class="doctor-card">
        <h4>{index}. {doc.get('name', 'Unknown')} ⭐ {doc.get('rating', 'N/A')}</h4>
        <p><b>Specialty:</b> {doc.get('specialty', 'Rheumatology')}</p>
        <p><b>Hospital:</b> {doc.get('hospital', 'N/A')}</p>
        <p><b>Location:</b> {doc.get('city', doc.get('distance', 'N/A'))}</p>
        <p><b>Phone:</b> {doc.get('phone', 'N/A')}</p>
        {f"<p><b>Experience:</b> {doc.get('experience')}</p>" if doc.get('experience') else ""}
    </div>
    """

def render_precaution_card(precaution, index):
    """Render precaution item."""
    return f"""
    <div class="precaution-item">
        <b>{index}.</b> {precaution}
    </div>
    """

# ==========================
# PDF GENERATION
# ==========================
def create_pdf_report(patient_id, report_data, image=None, 
                      arthritis_type=None, precautions=None,
                      local_doctors=None, indian_doctors=None,
                      care_plan=None):
    """
    Create comprehensive PDF report.
    """
    if not REPORTLAB_AVAILABLE:
        return _create_fallback_pdf(patient_id, report_data)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=50, leftMargin=50,
                          topMargin=50, bottomMargin=30)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=20,
        alignment=1  # Center
    )
    
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=10,
        spaceBefore=15
    )
    
    # Title
    story.append(Paragraph("🏥 Arthritis Care Assistant - Medical Report", title_style))
    story.append(Paragraph(f"<b>Patient ID:</b> {patient_id} | <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Horizontal line
    story.append(Table([['']], colWidths=[500], style=TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 2, colors.HexColor("#3498db")),
    ])))
    story.append(Spacer(1, 15))
    
    # Arthritis Type Detection
    if arthritis_type:
        story.append(Paragraph("🩺 Arthritis Type Classification", section_style))
        type_data = [
            ["Detected Type:", str(arthritis_type)],
            ["Confidence:", f"{care_plan.get('type_confidence', 0):.1%}" if care_plan else "N/A"],
            ["Risk Level:", str(report_data.get('risk', 'Unknown'))],
        ]
        type_table = Table(type_data, colWidths=[150, 300])
        type_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#ecf0f1")),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(type_table)
        story.append(Spacer(1, 15))
    
    # Module Results
    story.append(Paragraph("📊 Diagnostic Module Results", section_style))
    
    # If report_data is a list of module results
    if isinstance(report_data, list):
        for r in report_data:
            story.append(Paragraph(f"<b>{r.get('module', 'Unknown')}</b>", styles['Heading4']))
            story.append(Paragraph(f"Risk: <font color='{r.get('color', '#000')}'>{r.get('risk', 'Unknown')}</font>", styles['Normal']))
            # Handle confidence that might be string
            conf = r.get('confidence', 0)
            conf_str = conf if isinstance(conf, str) else f"{conf:.2%}"
            story.append(Paragraph(f"Score: {r.get('score', 0)} | Confidence: {conf_str}", styles['Normal']))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(f"<b>{report_data.get('module', 'Analysis')}</b>", styles['Heading4']))
        story.append(Paragraph(f"Risk: <font color='{report_data.get('color', '#000')}'>{report_data.get('risk', 'Unknown')}</font>", styles['Normal']))
        conf = report_data.get('confidence', 0)
        conf_str = conf if isinstance(conf, str) else f"{conf:.2%}"
        story.append(Paragraph(f"Score: {report_data.get('score', 0)} | Confidence: {conf_str}", styles['Normal']))
    
    story.append(Spacer(1, 15))
    
    # Image if provided
    if image is not None:
        story.append(Paragraph("🖼️ Imaging Analysis", section_style))
        try:
            if isinstance(image, str):
                img = Image.open(image)
            else:
                img = image
            
            # Save to temp and add
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name, 'PNG')
                story.append(RLImage(tmp.name, width=400, height=300))
                os.unlink(tmp.name)
        except Exception as e:
            story.append(Paragraph(f"Image could not be included: {str(e)}", styles['Normal']))
        story.append(Spacer(1, 15))
    
    # Precautions
    if precautions:
        story.append(PageBreak())
        story.append(Paragraph("⚠️ Type-Specific Precautions", section_style))
        for i, p in enumerate(precautions, 1):
            story.append(Paragraph(f"{i}. {p}", styles['Normal']))
            story.append(Spacer(1, 5))
    
    # Doctors
    if local_doctors or indian_doctors:
        story.append(PageBreak())
        story.append(Paragraph("👨‍⚕️ Recommended Doctors", section_style))
        
        if local_doctors:
            story.append(Paragraph("<b>Local Doctors:</b>", styles['Heading4']))
            for i, doc in enumerate(local_doctors[:5], 1):
                doc_text = f"{i}. {doc.get('name', 'Unknown')} - {doc.get('specialty', '')} | {doc.get('hospital', '')} | {doc.get('phone', '')}"
                story.append(Paragraph(doc_text, styles['Normal']))
                story.append(Spacer(1, 3))
        
        if indian_doctors:
            story.append(Paragraph("<b>Top Indian Rheumatologists:</b>", styles['Heading4']))
            for i, doc in enumerate(indian_doctors[:5], 1):
                doc_text = f"{i}. {doc.get('name', 'Unknown')} - {doc.get('specialty', '')} | {doc.get('hospital', '')}, {doc.get('city', '')} | {doc.get('phone', '')}"
                story.append(Paragraph(doc_text, styles['Normal']))
                story.append(Spacer(1, 3))
    
    # Care Plan
    if care_plan:
        story.append(Spacer(1, 15))
        story.append(Paragraph("📋 Care Plan & Next Steps", section_style))
        story.append(Paragraph(f"<b>Next Steps:</b> {care_plan.get('next_steps', 'Consult a rheumatologist')}", styles['Normal']))
        
        if care_plan.get('guidelines', {}).get('suggestions'):
            story.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
            for s in care_plan['guidelines']['suggestions'][:5]:
                story.append(Paragraph(f"• {s}", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "<i>This report is generated by AI for informational purposes only. "
        "Always consult a qualified rheumatologist for diagnosis and treatment.</i>",
        styles['Normal']
    ))
    
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    
    return pdf


def _create_fallback_pdf(patient_id, report_data):
    """Create a simple text-based PDF fallback."""
    buffer = io.BytesIO()
    content = f"""
    ARTHRITIS CARE ASSISTANT - MEDICAL REPORT
    =========================================
    Patient ID: {patient_id}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ANALYSIS RESULTS:
    {report_data}
    
    DISCLAIMER: This report is for informational purposes only.
    """
    buffer.write(content.encode('utf-8'))
    return buffer.getvalue()


# ==========================
# IMAGE PROCESSING UTILS
# ==========================
def resize_image_for_display(image, max_size=(800, 600)):
    """Resize image for display while maintaining aspect ratio."""
    if isinstance(image, str):
        image = Image.open(image)
    
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    if isinstance(image, str):
        image = Image.open(image)
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def smart_column_mapper(df, expected_columns):
    """Map CSV columns to expected columns using fuzzy matching."""
    mapped = {}
    available_cols = list(df.columns)
    
    for expected in expected_columns:
        # Exact match first
        if expected in available_cols:
            mapped[expected] = expected
            continue
        
        # Fuzzy match
        matches = difflib.get_close_matches(expected.lower(), 
                                          [c.lower() for c in available_cols], 
                                          n=1, cutoff=0.6)
        if matches:
            original = [c for c in available_cols if c.lower() == matches[0]][0]
            mapped[expected] = original
    
    # Create mapped dataframe
    result = pd.DataFrame()
    for expected, actual in mapped.items():
        result[expected] = df[actual]
    
    # Fill missing columns with defaults
    for expected in expected_columns:
        if expected not in result.columns:
            result[expected] = 0.0
    
    return result

if __name__ == "__main__":
    # Test
    report = generate_report_data("Knee X-Ray", 0.75, 0.92)
    print(render_report_html(report, "PAT001"))
    
    # Test PDF
    pdf = create_pdf_report("PAT001", report)
    print(f"PDF generated: {len(pdf)} bytes")
