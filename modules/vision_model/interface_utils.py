import streamlit as st
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

# ==========================================
# 1. PROFESSIONAL STYLING (THEME ENGINE)
# ==========================================
def apply_professional_style(theme):
    """Injects CSS to force widget text colors and fix visibility."""
    
    if theme == "Night":
        # DARK MODE COLORS
        main_bg = "#0E1117"
        sidebar_bg = "#262730"
        text_color = "#ffffff"
        header_color = "#5dade2"
        card_bg = "#1e2130"
        input_bg = "#2b313e"
        input_text = "#ffffff"
        border_color = "#444"
        upload_bg = "#2b313e"
        placeholder_color = "#a0a0a0"
        btn_bg = "#2b313e"
        btn_text = "#ffffff"
    else:
        # DAY MODE COLORS (High Contrast Medical Style)
        main_bg = "#ffffff"
        sidebar_bg = "#f8f9fa" 
        text_color = "#000000"
        header_color = "#0056b3"
        card_bg = "#f1f3f4"
        input_bg = "#ffffff"   # Force White Box
        input_text = "#000000" # Force Black Text
        border_color = "#000000" # Black Border
        upload_bg = "#f0f8ff"
        placeholder_color = "#666666"
        btn_bg = "#ffffff"
        btn_text = "#000000"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        /* 1. GLOBAL TEXT RESET */
        html, body, [class*="css"], p, h1, h2, h3, label, .stMarkdown {{
            font-family: 'Inter', sans-serif;
            color: {text_color} !important;
        }}
        
        /* 2. BACKGROUNDS */
        .stApp {{ background-color: {main_bg} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
        
        /* 3. INPUT BOXES (Deep Fix for Visibility) */
        /* The container */
        div[data-baseweb="input"] {{
            background-color: {input_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 5px !important;
        }}
        
        /* The typing text */
        input[type="text"] {{
            color: {input_text} !important;
            -webkit-text-fill-color: {input_text} !important;
            caret-color: {input_text} !important;
        }}
        
        /* The placeholder text (e.g. 'Anonymous') */
        input::placeholder {{
            color: {placeholder_color} !important;
            opacity: 1 !important;
            -webkit-text-fill-color: {placeholder_color} !important;
        }}
        
        /* 4. FILE UPLOADER */
        /* The drop zone */
        div[data-testid="stFileUploader"] section {{
            background-color: {upload_bg} !important;
            border: 1px dashed {header_color} !important;
        }}
        
        /* The 'Browse files' button */
        div[data-testid="stFileUploader"] button {{
            background-color: {btn_bg} !important;
            color: {btn_text} !important;
            border: 1px solid {border_color} !important;
        }}
        
        /* The file name text */
        div[data-testid="stFileUploader"] div, 
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {{
            color: {text_color} !important;
        }}

        /* 5. REPORT CARD */
        .report-container {{
            background-color: {card_bg};
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            margin-top: 20px;
        }}
        .report-header {{
            border-bottom: 3px solid {header_color};
            padding-bottom: 15px; margin-bottom: 20px;
            color: {header_color} !important;
            font-size: 1.5rem; font-weight: 700;
        }}
        .section-title {{
            color: {text_color} !important;
            font-weight: 700; margin-top: 20px;
            text-transform: uppercase; font-size: 0.8rem; opacity: 0.8;
        }}
        .finding-text {{
            color: {text_color} !important;
            line-height: 1.6; font-size: 1.05rem;
        }}
        
        /* STATUS BOXES */
        .warning-box {{ background-color: #fff8e1; border-left: 5px solid #ffc107; padding: 15px; color: #856404 !important; }}
        .success-box {{ background-color: #e8f5e9; border-left: 5px solid #28a745; padding: 15px; color: #155724 !important; }}
        .danger-box {{ background-color: #ffebee; border-left: 5px solid #dc3545; padding: 15px; color: #721c24 !important; }}
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MEDICAL TEXT GENERATOR
# ==========================================
def generate_report_data(modality, grade, confidence, extra_info=None):
    if modality == "Knee":
        if extra_info == "Borderline":
            return {'title': "RADIOLOGY REPORT: KNEE X-RAY", 'findings': "Subtle osteophytic lipping. No severe narrowing definitively visualized.", 'impression': "Borderline Grade 3 (Moderate)", 'recommendation': "Clinical correlation essential. Repeat X-ray in 6 months.", 'style': "warning-box"}
        grades = {
            0: ("Grade 0 (Normal)", "No radiographic features of OA.", "success-box", "Routine monitoring."),
            1: ("Grade 1 (Doubtful)", "Doubtful narrowing of joint space.", "success-box", "Monitor symptoms."),
            2: ("Grade 2 (Minimal)", "Definite osteophytes, possible narrowing.", "warning-box", "Conservative management (PT)."),
            3: ("Grade 3 (Moderate)", "Moderate osteophytes, definite narrowing.", "warning-box", "Orthopedic consult."),
            4: ("Grade 4 (Severe)", "Large osteophytes, severe narrowing.", "danger-box", "Surgical consultation advised.")
        }
        d, f, s, r = grades.get(grade, ("Unknown", "N/A", "warning-box", "N/A"))
        return {'title': "RADIOLOGY REPORT: KNEE X-RAY", 'findings': f, 'impression': f"{d} (Confidence: {confidence:.1%})", 'recommendation': r, 'style': s}
    elif modality == "Spine":
        risk = grade
        if risk > 60: return {'title': "MRI ANALYSIS: LUMBAR SPINE", 'findings': "Significant signal reduction (Desiccation).", 'impression': f"High Risk - Score {risk:.1f}", 'recommendation': "MRI Review required.", 'style': "danger-box"}
        elif risk > 20: return {'title': "MRI ANALYSIS: LUMBAR SPINE", 'findings': "Mild signal heterogeneity.", 'impression': f"Moderate Risk - Score {risk:.1f}", 'recommendation': "Physical Therapy.", 'style': "warning-box"}
        else: return {'title': "MRI ANALYSIS: LUMBAR SPINE", 'findings': "Signal hyperintense (Healthy).", 'impression': f"Low Risk - Score {risk:.1f}", 'recommendation': "No intervention.", 'style': "success-box"}
    elif modality == "Fluid":
        if grade == 0: return {'title': "SYNOVIAL FLUID", 'findings': "Complex textures detected (Crystals).", 'impression': "POSITIVE for Inflammation", 'recommendation': "Rheumatology consult.", 'style': "danger-box"}
        else: return {'title': "SYNOVIAL FLUID", 'findings': "Clear fluid background.", 'impression': "NEGATIVE", 'recommendation': "Routine follow-up.", 'style': "success-box"}

def render_report_html(report, patient_id):
    return f"""
    <div class="report-container">
        <div class="report-header">🏥 {report['title']}</div>
        <div class="section-title">Patient ID</div>
        <div class="finding-text">{patient_id}</div>
        <div class="section-title">Detailed Findings</div>
        <div class="finding-text">{report['findings']}</div>
        <div class="section-title">Impression</div>
        <div class="{report['style']}">{report['impression']}</div>
        <div class="section-title">Plan</div>
        <div class="finding-text">👉 {report['recommendation']}</div>
    </div>
    """

# ==========================================
# 3. PDF GENERATOR (FIXED)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(0, 50, 100)
        self.cell(0, 10, 'MedView AI Diagnostic Center', 0, 1, 'L')
        self.line(10, 20, 200, 20)
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | AI Generated Report', 0, 0, 'C')

def create_pdf_report(patient_id, report_data, image_pil):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(40, 10, f'Patient: {patient_id}', 0, 1)
    pdf.cell(40, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', 0, 1)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image_pil.save(tmp.name)
        pdf.image(tmp.name, x=55, w=100)
    
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "IMPRESSION:", 0, 1)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 6, report_data['impression'])
    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "FINDINGS:", 0, 1)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(0, 6, report_data['findings'])
    
    # CRITICAL FIX: Convert bytearray to bytes for Streamlit
    return pdf.output(dest='S')