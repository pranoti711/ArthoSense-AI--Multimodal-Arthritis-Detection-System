import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
import SimpleITK as sitk
import tempfile
import pickle
import sys
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image as keras_image  # type: ignore

try:
    from interface_utils import apply_professional_style, generate_report_data, render_report_html, create_pdf_report
except ImportError:
    from services.gui.interface_utils import apply_professional_style, generate_report_data, render_report_html, create_pdf_report

# Guard against running with 'python' instead of 'streamlit run'
if not st.runtime.exists():
    
    print("ERROR: This script must be run with 'streamlit run modules/vision_model/app.py'")
    sys.exit(1)

st.set_page_config(page_title="MedView AI Pro", page_icon="🩺", layout="wide")

KNEE_MODEL_PATH = "modules/vision_model/oai_knee_model.pth"
CLUSTER_MODEL_PATH = "modules/vision_model/fluid_cluster_model.pkl"

class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit, self.tile_grid_size = clip_limit, tile_grid_size
    def __call__(self, img):
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return Image.fromarray(cv2.cvtColor(clahe.apply(img_gray), cv2.COLOR_GRAY2RGB))

@st.cache_resource
def load_knee_model():
    try:
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
        if os.path.exists(KNEE_MODEL_PATH):
            model.load_state_dict(torch.load(KNEE_MODEL_PATH, map_location="cpu"))
            model.eval()
            return model
    except: return None

def process_knee_image(image_file):
    img = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), ApplyCLAHE(2.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform(img).unsqueeze(0), img

def analyze_spine_health(mid_slice):
    height, width = mid_slice.shape
    center_crop = mid_slice[:, int(width*0.35):int(width*0.65)]
    _, mask = cv2.threshold(center_crop, 40, 255, cv2.THRESH_BINARY)
    heatmap = cv2.bitwise_and(cv2.applyColorMap(center_crop, cv2.COLORMAP_JET), cv2.applyColorMap(center_crop, cv2.COLORMAP_JET), mask=mask)
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    
    valid_pixels = center_crop[mask > 0]
    if valid_pixels.size == 0: return 0.0, heatmap_pil
    
    csf_baseline = np.percentile(valid_pixels, 98)
    
    ratios = []
    chunk_h = center_crop.shape[0] // 5
    for i in range(5):
        seg = center_crop[i*chunk_h:(i+1)*chunk_h, :]
        if np.any(mask[i*chunk_h:(i+1)*chunk_h, :] > 0): 
            disc_brightness = np.percentile(seg[mask[i*chunk_h:(i+1)*chunk_h, :] > 0], 90)
            ratios.append(disc_brightness / csf_baseline)
            
    if not ratios: return 0.0, heatmap_pil
    
    worst = np.min(ratios)
    return float(95.0 if worst <= 0.5 else 5.0 if worst >= 0.9 else (0.9 - worst)/0.4*100), heatmap_pil

def read_mha_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp: tmp.write(uploaded_file.getbuffer()); tmp_path = tmp.name
    try:
        img_3d = sitk.ReadImage(tmp_path)
        arr = sitk.GetArrayFromImage(img_3d)
        return cv2.normalize(arr[arr.shape[0]//2, :, :], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), arr.shape[0]
    except: return None, 0
    finally: os.remove(tmp_path) if os.path.exists(tmp_path) else None

@st.cache_resource
def load_resnet_extractor(): return ResNet50(weights='imagenet', include_top=False, pooling='avg')

def analyze_fluid_ml(uploaded_file, extractor):
    if not os.path.exists(CLUSTER_MODEL_PATH): return None, None
    with open(CLUSTER_MODEL_PATH, 'rb') as f: kmeans = pickle.load(f)
    img = Image.open(uploaded_file).convert('RGB')
    x = preprocess_input(np.expand_dims(keras_image.img_to_array(img.resize((224, 224))), axis=0))
    return kmeans.predict(extractor.predict(x, verbose=0))[0], img


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=70)
    st.title("MedView AI Pro")
    theme = st.radio("Appearance Mode:", ["Day", "Night"])
    apply_professional_style(theme)
    st.divider()
    mode = st.radio("Diagnostic Module:", ["Knee X-Ray", "Spine MRI", "Fluid Lab"])

if mode == "Knee X-Ray":
    st.header("Knee Osteoarthritis Detection")
    c1, c2 = st.columns([1, 2])
    f = c1.file_uploader("Upload X-Ray", type=["jpg", "png"])
    pid = c2.text_input("Patient ID", "Anonymous")
    if f:
        t, img = process_knee_image(f)
        model = load_knee_model()
        if model:
            with st.spinner("Analyzing..."):
                out = (torch.nn.functional.softmax(model(t), dim=1) + torch.nn.functional.softmax(model(torch.flip(t, [3])), dim=1)) / 2
                conf, pred = torch.max(out, 1)
                grade = pred.item()
            extra = "Borderline" if (grade == 3 and conf.item() < 0.60) else None
            cA, cB = st.columns([1, 1.5])
            cA.image(img, caption="Input", use_container_width=True)
            report = generate_report_data("Knee", grade, conf.item(), extra)
            cB.markdown(render_report_html(report, pid), unsafe_allow_html=True)
            pdf_bytes = create_pdf_report(pid, report, img)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name=f"Report_{pid}.pdf", mime="application/pdf")

elif mode == "Spine MRI":
    st.header("Spine Disc Analysis")
    f = st.file_uploader("Upload MRI (.mha)", type=["mha"])
    
    if f:
        sl, _ = read_mha_file(f)
        if sl is not None:
            risk, heat_pil = analyze_spine_health(sl)
            
            st.markdown("### 🖼️ Imaging Analysis")
            col_orig, col_heat = st.columns(2)
            
            with col_orig:
                st.image(sl, caption="Original T2 MRI Slice (Mid-Sagittal)", use_container_width=True)
                
            with col_heat:
                st.image(heat_pil, caption="Signal Intensity Heatmap (Red=High Signal)", use_container_width=True)
            
            # --- UPDATED EXPLANATION: Simple & Professional ---
            st.divider()
            with st.expander("ℹ️ How was this Risk Score calculated?", expanded=True):
                st.markdown("""
                ### 💧 Hydration Analysis (T2-Weighted MRI)
                
                This AI uses **digital signal processing** to measure the water content inside your spinal discs.
                
                * **The Concept:** Healthy spinal discs are like "water cushions"—they are full of fluid. In T2 MRI scans, fluid glows **Bright White**.
                * **The Problem:** As we age or develop arthritis, discs dry out (desiccate) and turn **Dark/Black** in the scan.
                
                **How we measure it:**
                1.  **Baseline:** We find the brightest fluid in your spine (Cerebrospinal Fluid) to set the "100% Hydration" standard.
                2.  **Comparison:** We check if your discs are glowing bright (Healthy) or fading to dark (At Risk).
                
                **Result Interpretation:**
                * ✅ **Low Score (Healthy):** Your discs are bright and well-hydrated.
                * ⚠️ **High Score (At Risk):** Your discs are dark, indicating dehydration or degeneration.
                """)

            report = generate_report_data("Spine", risk, 1.0)
            st.markdown(render_report_html(report, "Anonymous"), unsafe_allow_html=True)
            
            pdf_bytes = create_pdf_report("Anonymous", report, heat_pil)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name="Spine_Report.pdf", mime="application/pdf")

elif mode == "Fluid Lab":
    st.header("Synovial Fluid Analysis")
    f = st.file_uploader("Upload Photo", type=["jpg", "png"])
    if f:
        ext = load_resnet_extractor()
        cl, img = analyze_fluid_ml(f, ext)
        if cl is not None:
            cA, cB = st.columns([1, 1.5])
            cA.image(img, caption="Microscopy", use_container_width=True)
            report = generate_report_data("Fluid", cl, 1.0)
            cB.markdown(render_report_html(report, "Anonymous"), unsafe_allow_html=True)
            pdf_bytes = create_pdf_report("Anonymous", report, img)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name="Fluid_Report.pdf", mime="application/pdf")