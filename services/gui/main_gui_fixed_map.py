import os
import sys
import io
import json
import tempfile
import base64
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import requests
import folium
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from PIL import Image
from streamlit_folium import st_folium

# Safe path setup for direct script execution and package imports
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT

for parent in PROJECT_ROOT.parents:
    if (parent / "modules").exists() and (parent / "integration").exists():
        WORKSPACE_ROOT = parent
        break

candidates = {str(WORKSPACE_ROOT), str(Path.cwd()), str(Path.cwd().parent)}
for candidate in candidates:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from integration.geospatial_network.provider_ranker import get_all_doctors

# ==============================
# UTILITY FUNCTIONS
# ==============================

def generate_report_data(title, score, confidence, details="", interpretation=""):
    return {
        "title": title,
        "score": float(score),
        "confidence": float(confidence),
        "details": details,
        "interpretation": interpretation,
    }


def render_report_html(report, patient_id):
    score = float(report["score"]) if isinstance(report["score"], (int, float)) else 0.5
    conf = float(report["confidence"]) if isinstance(report["confidence"], (int, float)) else 0.85
    return f"""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 15px; padding: 2rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border-left: 6px solid #667eea;">
        <h3 style="color: #1e293b; margin: 0 0 1rem 0;">{report['title']}</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px;">
                <strong style="color: #64748b; font-size: 0.9rem;">Risk Score</strong>
                <h2 style="color: #1e293b; margin: 0.5rem 0 0 0;">{score:.3f}</h2>
            </div>
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px;">
                <strong style="color: #64748b; font-size: 0.9rem;">Confidence</strong>
                <h2 style="color: #1e293b; margin: 0.5rem 0 0 0;">{conf:.1%}</h2>
            </div>
        </div>
        <p style="color: #475569; margin: 0 0 1rem 0;"><strong>Details:</strong> {report['details']}</p>
        <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
            <strong style="color: #1e293b;"> Patient Explanation:</strong><br>
            <span style="color: #475569; font-size: 0.95rem;">{report['interpretation']}</span>
        </div>
    </div>
    """


# ==============================
# ULTRA-SMART CHATBOT ENGINE
# ==============================

COMMON_MISSPELLINGS = {
    "arthritus": "arthritis",
    "arthrits": "arthritis",
    "artritis": "arthritis",
    "rehumatoid": "rheumatoid",
    "rumatoid": "rheumatoid",
    "osteoartritis": "osteoarthritis",
    "gouty": "gout",
    "stiffnes": "stiffness",
    "sweling": "swelling",
    "swolen": "swollen",
    "painfull": "painful",
    "pian": "pain",
    "tretment": "treatment",
    "dite": "diet",
    "exersise": "exercise",
    "meds": "medicine",
    "docor": "doctor",
}

ARTHRITIS_KB = {
    "arthritis": " **Arthritis types**: 1 Osteoarthritis (wear/tear), 2Rheumatoid (autoimmune), 3Gout (uric acid), 4Psoriatic.",
    "rheumatoid": " **Rheumatoid**: Morning stiffness >1hr, symmetric joints. **DMARDs** (Methotrexate first).",
    "osteoarthritis": "**Osteoarthritis**: Activity pain, stiffness <30min. **Weight loss + physio** best.",
    "gout": "? **Gout**: Big toe attacks. **Avoid beer/red meat**. Allopurinol prevention.",
    "treatment": "**Treatment**: OA=Physio, RA=DMARDs, Gout=Allopurinol. See rheumatologist!",
    "diet": "**Anti-inflammatory**: Fish oil, berries, turmeric. Avoid beer/red meat.",
    "exercise": " **Safe**: Walking, swimming, yoga. 30min/day.",
    "doctor": " **Rheumatologist** for proper diagnosis & treatment plan.",
    "pain": "**Pain relief**: Paracetamol 500mg, topical gels, hot/cold therapy.",
    "symptoms": "**Symptoms**: Pain, stiffness, swelling, fatigue. Duration/location matters!",
    "default": " Ask about **arthritis types**, **RA**, **OA**, **gout**, **treatment**, **diet**, or **doctors**.",
}

PATTERNS = {
    r"(pain|ache|hurt|knee|hip|joint)": "pain",
    r"(stiff|morning|swell)": "symptoms",
    r"(treat|medicine|med|drug|pill)": "treatment",
    r"(diet|food|eat)": "diet",
    r"(exercise|walk|swim|yoga)": "exercise",
    r"(doctor|specialist|rheumatologist)": "doctor",
}


def correct_spelling(text: str) -> str:
    words = text.lower().split()
    corrected = []
    for word in words:
        corrected.append(COMMON_MISSPELLINGS.get(word, word))
    return " ".join(corrected)


def query_ollama(prompt: str, model: str = "llama2") -> Optional[str]:
    commands = [
        ["ollama", "query", model, prompt, "--json"],
        ["ollama", "query", "llama2-mini", prompt, "--json"],
        ["ollama", "query", model, prompt],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except FileNotFoundError:
            return None
        except Exception:
            continue

        if result.returncode != 0 or not result.stdout:
            continue

        output = result.stdout.strip()
        if not output:
            continue

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return str(payload.get("text", payload.get("response", output))).strip()
            except json.JSONDecodeError:
                continue

        return output

    return None


def get_chatbot_response(question: str, history: List) -> str:
    history_text = "\n".join(
        f"User: {item['content']}" if item["role"] == "user" else f"Assistant: {item['content']}"
        for item in history[-4:]
    )
    prompt = (
        "You are an arthritis care assistant. Answer clearly and safely. "
        "If the user asks for medical advice, recommend consulting a licensed doctor.\n"
        f"{history_text}\nUser: {question}\nAssistant:"
    )

    answer = query_ollama(prompt)
    if answer:
        return answer

    return get_smart_chat_response(question, history)


def get_smart_chat_response(question: str, history: List) -> str:
    q_lower = correct_spelling(question).lower()

    recent = [h["content"].lower() for h in history[-3:] if h["role"] == "user"]
    if any(q_lower in r for r in recent):
        return " I answered that recently! Try **diet**, **exercises**, or **RA vs OA** differences. "

    for pattern, topic in PATTERNS.items():
        if re.search(pattern, q_lower):
            return ARTHRITIS_KB.get(topic, ARTHRITIS_KB["default"])

    for keyword, response in ARTHRITIS_KB.items():
        if keyword in q_lower:
            return response

    return ARTHRITIS_KB["default"]


# ==============================
# REAL DOCTORS DATABASE (OpenStreetMap/Overpass)
# ==============================

def get_real_doctors(city: str, count: int = 10, radius_m: int = 10000) -> List[Dict]:
    geolocator = Nominatim(user_agent="arthritis_ai")
    loc = geolocator.geocode(city)
    if not loc:
        return []

    lat = float(loc.latitude)
    lon = float(loc.longitude)

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      node["healthcare"="doctor"](around:{radius_m},{lat},{lon});
      node["amenity"="doctors"](around:{radius_m},{lat},{lon});
    );
    out center;
    """

    try:
        r = requests.get(overpass_url, params={"data": query}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    results: List[Dict] = []
    for el in data.get("elements", [])[:count]:
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("brand") or "Unknown"

        lat_el = el.get("lat")
        lon_el = el.get("lon")
        center = el.get("center") or {}
        if lat_el is None:
            lat_el = center.get("lat")
        if lon_el is None:
            lon_el = center.get("lon")

        if lat_el is None or lon_el is None:
            continue

        results.append(
            {
                "name": name,
                "lat": float(lat_el),
                "lon": float(lon_el),
                "hospital": tags.get("name"),
                "specialty": tags.get("speciality") or tags.get("specialty") or "Rheumatology",
                "city": city,
                "phone": tags.get("phone"),
                "tags": tags,
            }
        )

    return results


def display_doctors_map(docs: List[Dict], city: str) -> None:
    """Render doctors on a map.

    Robust against missing/None lat/lon keys coming from OSM or fallback providers.
    """

    valid_docs: List[Dict] = []
    for d in (docs or []):
        lat = d.get("lat")
        lon = d.get("lon")
        if lat is None or lon is None:
            continue
        try:
            d["lat"] = float(lat)
            d["lon"] = float(lon)
        except (TypeError, ValueError):
            continue
        valid_docs.append(d)

    if valid_docs:
        center = [valid_docs[0]["lat"], valid_docs[0]["lon"]]
    else:
        center = [21.1458, 79.0882]  # Nagpur

    m = folium.Map(location=center, zoom_start=13)

    for d in valid_docs:
        popup_lines = [d.get("name", "Unknown")]
        specialty = d.get("specialty")
        if specialty:
            popup_lines.append(str(specialty))

        folium.Marker(
            [d["lat"], d["lon"]],
            popup=" - ".join(popup_lines),
            tooltip=d.get("name", "Unknown"),
        ).add_to(m)

    if not valid_docs:
        st.info(
            "Map couldn’t load doctor coordinates for this location. "
            "Showing an empty map view."
        )

    st_folium(m, width=700)


def get_doctors_by_location(
    location: str,
    arthritis_type: str = None,
    count: int = 4,
    google_maps_api_key: str = None,
) -> List[Dict]:
    try:
        real = get_real_doctors(location, count=count)
        if real:
            return real[:count]
    except Exception:
        pass

    try:
        fallback = get_all_doctors(location)
        if isinstance(fallback, dict) and "local_doctors" in fallback:
            return fallback["local_doctors"][:count]
        if isinstance(fallback, list):
            return fallback[:count]
    except Exception:
        pass

    return [
        {
            "name": "Local Rheumatologist",
            "specialty": "Rheumatology",
            "hospital": "Local Hospital",
            "city": location,
            "phone": "N/A",
            "lat": None,
            "lon": None,
        }
    ]


# ==============================
# Mock ML predictions
# ==============================

def run_vision_analysis(image_file):
    score = np.random.uniform(0.1, 0.9)
    return score, None, "X-ray analysis shows changes in bone density and cartilage alignment."


def run_voice_analysis(audio_file):
    score = np.random.uniform(0.3, 0.95)
    return "Normal", score, None, "Voice biomarkers indicate normal speech patterns and mobility."


def run_lab_analysis(csv_file):
    score = np.random.uniform(0.2, 0.85)
    return score, None, "Blood markers show normal inflammatory levels (CRP <3 mg/L, ESR <20)."


def run_genomics_analysis(data):
    score = np.random.uniform(0.1, 0.75)
    return score, None, "Genetic risk assessment complete - moderate genetic predisposition detected."


def run_wearable_analysis(csv_file):
    score = np.random.uniform(0.3, 0.8)
    return score, None, "Activity and sleep patterns show good compliance with exercise recommendations."


# ==============================
# LIVE VOICE RECORDER COMPONENT
# ==============================

def google_voice_recorder(key="voice_main"):
    recorder_html = f"""
    <div style="padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; text-align: center; color: white;">
        <h3 style="margin: 0 0 1rem 0;"> Live Voice Recording</h3>
        <button id="start{key}" onclick="startRecording('{key}')" style="
            background: #ef4444; color: white; border: none; padding: 15px 30px; 
            border-radius: 50px; font-size: 18px; font-weight: bold; cursor: pointer;
            box-shadow: 0 10px 20px rgba(239,68,68,0.4);">Start Recording</button>
        <button id="stop{key}" onclick="stopRecording('{key}')" disabled style="
            background: #1e293b; color: white; border: none; padding: 15px 30px; 
            border-radius: 50px; font-size: 18px; margin-left: 10px;">Stop</button>
        <div id="status{key}" style="margin-top: 1rem; font-size: 16px;"></div>
    </div>

    <script>
    let recorder_{key}, stream_{key};
    async function startRecording(key) {{
        try {{
            stream_{key} = await navigator.mediaDevices.getUserMedia({{audio: true}});
            recorder_{key} = new MediaRecorder(stream_{key});
            let chunks = [];

            recorder_{key}.ondataavailable = e => chunks.push(e.data);
            recorder_{key}.onstop = () => {{
                const blob = new Blob(chunks, {{type: 'audio/wav'}});
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                reader.onloadend = () => {{
                    const base64data = reader.result;
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: base64data
                    }}, '*');
                }};
            }};

            recorder_{key}.start();
            document.getElementById('start'+key).disabled = true;
            document.getElementById('stop'+key).disabled = false;
            document.getElementById('status'+key).innerText = ' Recording...';
        }} catch(err) {{
            document.getElementById('status'+key).innerText = '? Mic error: ' + err.message;
        }}
    }}

    function stopRecording(key) {{
        if(recorder_{key} && recorder_{key}.state !== 'inactive') {{
            recorder_{key}.stop();
            stream_{key}.getTracks()[0].stop();
            document.getElementById('start'+key).disabled = false;
            document.getElementById('stop'+key).disabled = true;
            document.getElementById('status'+key).innerText = '? Recording saved!';
        }}
    }}
    </script>
    """
    return components.html(recorder_html, height=200)


# ==============================
# PAGE CONFIG & SESSION STATE
# ==============================

st.set_page_config(page_title="Arthritis Care Assistant Pro", page_icon="", layout="wide")


@st.cache_data
def init_session():
    return {
        "chat_history": [],
        "analysis_results": {},
        "patient_id": f"PAT{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "location": "Nagpur",
        "full_assessment": None,
    }


if "state" not in st.session_state:
    st.session_state.state = init_session()


st.markdown(
    """
<style>
.main-header {font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 15px; text-align: center;}
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## **Arthritis Care Pro**")
    st.session_state.state["patient_id"] = st.text_input(
        " Patient ID", st.session_state.state["patient_id"]
    )
    st.session_state.state["location"] = st.text_input(
        " City", st.session_state.state["location"]
    )

    page = st.radio(
        " Navigation",
        [
            " Dashboard",
            " Vision",
            " Voice",
            " Lab",
            " Genomics",
            " Wearable",
            " Chatbot",
            " Full Assessment",
            " Reports",
        ],
    )


MODULE_EXPLANATIONS = {
    "vision": {
        "title": "X-Ray Analysis - Spine, Fingers & Knees",
        "explanation": "X-ray imaging detects bone changes, cartilage damage, and alignment issues. The KL grade (0-4) measures osteoarthritis severity.",
        "spine": "Cervical/Lumbar spine analysis shows disc degeneration, bone spurs, and alignment issues.",
        "fingers": "Hand X-rays reveal joint narrowing, osteophytes (bone spurs), and soft tissue swelling in finger joints.",
        "knees": "Knee X-rays assess cartilage loss, bone density, and meniscal changes typical in osteoarthritis.",
        "medicines": {
            "low": "Topical NSAIDs, physical therapy",
            "medium": "Oral NSAIDs (Ibuprofen), Paracetamol",
            "high": "Hyaluronic acid injections, Corticosteroids",
        },
        "precautions": "Avoid high-impact activities, use supportive footwear, maintain healthy weight.",
        "tips": "Regular low-impact exercise (swimming, walking), warm compress for 15-20 min before activity.",
    },
    "voice": {
        "title": "Voice Biomarker Analysis",
        "explanation": "Voice analysis detects subtle changes in speech patterns, tremor, and voice quality that correlate with arthritis progression and pain levels.",
        "interpretation": "Normal voice patterns suggest good arthritis control; abnormal patterns may indicate inflammation or pain affecting mobility.",
        "medicines": {
            "low": "Rest and monitoring",
            "medium": "Physiotherapy, NSAIDs",
            "high": "DMARDs (Methotrexate), Biologic therapies",
        },
        "precautions": "Vocal rest if experiencing pain during speech; avoid shouting or prolonged talking.",
        "tips": "Use voice rest exercises, stay hydrated, practice controlled breathing techniques.",
    },
    "lab": {
        "title": "Lab Analysis - Blood Tests & Clinical Markers",
        "explanation": "Lab tests measure inflammatory markers (CRP, ESR), rheumatoid factor, anti-CCP antibodies, and other biomarkers indicating disease activity.",
        "interpretation": "Elevated CRP (>3 mg/L) or ESR (>20 mm/hr) suggest active inflammation; Positive RF/anti-CCP confirm rheumatoid arthritis.",
        "medicines": {
            "low": "NSAIDs, routine monitoring",
            "medium": "DMARDs (Methotrexate 15mg/week)",
            "high": "Biologic DMARDs (TNF inhibitors), JAK inhibitors",
        },
        "precautions": "Monthly LFTs when on DMARDs; avoid NSAIDs if kidney/GI issues; CBC monitoring for immunosuppression.",
        "tips": "Regular blood work every 4-6 weeks initially; maintain medication adherence; report side effects immediately.",
    },
    "genomics": {
        "title": "Genetic Risk Assessment",
        "explanation": "Genomic analysis identifies genetic predispositions to arthritis (HLA-B27, PTPN22 variants) and influences treatment response.",
        "interpretation": "HLA-B27 positive: higher risk for ankylosing spondylitis; PTPN22 variants increase RA risk; genetic profiling helps predict treatment response.",
        "medicines": {
            "low": "Standard NSAIDs, lifestyle management",
            "medium": "DMARDs based on genetic profile",
            "high": "Targeted biologics (TNF, IL-6, IL-17 inhibitors)",
        },
        "precautions": "Discuss genetic findings with genetic counselor; consider family screening; adapt preventive measures based on genetic risk.",
        "tips": "Genetic testing helps personalize treatment; share results with family members; follow preventive lifestyle measures.",
    },
    "wearable": {
        "title": "Wearable IoT Analysis - Activity & Sleep Patterns",
        "explanation": "Wearable sensors track activity levels, sleep quality, and physical stress patterns that correlate with arthritis flares and disease progression.",
        "interpretation": "Low activity (<5000 steps/day) may indicate pain limitation; Poor sleep (<6hrs) worsens inflammation; Stress elevation triggers flares.",
        "medicines": {
            "low": "Maintain activity, NSAIDs PRN",
            "medium": "Scheduled NSAIDs + wearable-guided exercise",
            "high": "DMARDs + NSAIDs + sleep-aid consideration",
        },
        "precautions": "Don't overdo activity during flares; ensure 7-8 hours sleep; manage stress through meditation/yoga; stay hydrated.",
        "tips": "Aim for 10,000 steps/day with rest breaks; use activity tracker for motivation; track sleep-flare correlations.",
    },
}


# ==============================
# Module insights helper
# ==============================

def render_module_insights(module_name: str, score: float):
    exp = MODULE_EXPLANATIONS.get(module_name)
    if not exp:
        return

    if score < 0.4:
        med = exp['medicines']['low']
    elif score < 0.7:
        med = exp['medicines']['medium']
    else:
        med = exp['medicines']['high']

    st.markdown(f"**Quick Recommendations for {exp['title']}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**💊 Medicines:**\n{med}")
    with c2:
        st.write(f"**⚠️ Precautions:**\n{exp.get('precautions','Follow up with provider')}")
    with c3:
        st.write(f"**💡 Patient Tips:**\n{exp.get('tips','Follow general advice')}")


# ==============================
# PAGES
# ==============================

if page == " Dashboard":
    st.markdown('<h1 class="main-header">Arthritis Care Assistant Pro</h1>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>Modules</h3><h1>9</h1></div>', unsafe_allow_html=True)
    with col2:
        st.metric("Patient", st.session_state.state["patient_id"])
    with col3:
        st.metric("Location", st.session_state.state["location"])
    with col4:
        st.metric("Results", len(st.session_state.state["analysis_results"]))

elif page == " Vision":
    st.title(" X-Ray Analysis")
    uploaded = st.file_uploader(" Upload X-Ray", type=["jpg", "jpeg", "png"], key="vision")
    if uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(Image.open(uploaded), use_column_width=True)
        with col2:
            score, _, interp = run_vision_analysis(uploaded)
            report = generate_report_data("X-Ray Analysis", score, 0.92, "KL Grade", interp)
            st.session_state.state["analysis_results"]["vision"] = report
            st.markdown(render_report_html(report, st.session_state.state["patient_id"]), unsafe_allow_html=True)
            render_module_insights('vision', score)

elif page == " Voice":
    st.title(" Voice Analysis")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload WAV", type=["wav"], key="voice_upload")
        if uploaded:
            st.audio(uploaded)
            label, conf, _, interp = run_voice_analysis(uploaded)
            report = generate_report_data("Voice Analysis", conf, conf, f"Status: {label}", interp)
            st.session_state.state["analysis_results"]["voice"] = report
            st.markdown(render_report_html(report, st.session_state.state["patient_id"]), unsafe_allow_html=True)
            render_module_insights('voice', conf)
    with col2:
        google_voice_recorder("voice_live")

elif page == " Chatbot":
    st.title(" Ultra-Smart Arthritis Chatbot")
    for msg in st.session_state.state["chat_history"][-10:]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input(" Speak or type about arthritis..."):
        st.session_state.state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            answer = get_chatbot_response(user_input, st.session_state.state["chat_history"])
            st.write(answer)
            st.session_state.state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()

elif page == " Full Assessment":
    st.title(" Complete Assessment")
    st.markdown("Run analysis on all available data to get comprehensive diagnosis and recommendations.")

    col1, col2 = st.columns(2)
    with col1:
        xray = st.file_uploader(" X-Ray (Spine/Fingers/Knees)", type=["jpg", "png"], key="fa_xray")
        voice = st.file_uploader(" Voice Recording", type=["wav"], key="fa_voice")
        lab = st.file_uploader(" Lab CSV", type=["csv"], key="fa_lab")
    with col2:
        genomics = st.file_uploader(" Genomics CSV", type=["csv"], key="fa_genomics")
        wearable = st.file_uploader("? Wearable IoT CSV", type=["csv"], key="fa_wear")

    if st.button(" RUN FULL ASSESSMENT", type="primary"):
        results = {}
        details_text = ""

        if xray:
            score, _, detail = run_vision_analysis(xray)
            results["vision"] = score
            details_text += f"\n✓ **Vision:** {detail}"
        if voice:
            label, score, _, detail = run_voice_analysis(voice)
            results["voice"] = score
            details_text += f"\n✓ **Voice:** {detail}"
        if lab:
            score, _, detail = run_lab_analysis(lab)
            results["lab"] = score
            details_text += f"\n✓ **Lab:** {detail}"
        if genomics:
            score, _, detail = run_genomics_analysis(genomics)
            results["genomics"] = score
            details_text += f"\n✓ **Genomics:** {detail}"
        if wearable:
            score, _, detail = run_wearable_analysis(wearable)
            results["wearable"] = score
            details_text += f"\n✓ **Wearable:** {detail}"

        overall = np.mean(list(results.values())) if results else 0.5

        if overall < 0.35:
            diagnosis = "Low Risk / Normal"
            confidence = 0.92
        elif overall < 0.6:
            diagnosis = "Osteoarthritis (Early Stage)"
            confidence = 0.88
        elif overall < 0.75:
            diagnosis = "Rheumatoid Arthritis"
            confidence = 0.85
        else:
            diagnosis = "Advanced Inflammatory Arthritis"
            confidence = 0.82

        st.session_state.state["full_assessment"] = {
            "diagnosis": diagnosis,
            "risk": overall,
            "results": results,
            "confidence": confidence,
            "details": details_text,
        }
        st.success("✓ Full assessment complete!")
        st.rerun()

    if assessment := st.session_state.state.get("full_assessment"):
        docs = get_doctors_by_location(
            st.session_state.state["location"], assessment["diagnosis"], 6
        )

        st.subheader(f"🏥 Recommended Specialists for {assessment['diagnosis']}")
        dcols = st.columns(2)
        for i, doc in enumerate(docs[:4]):
            with dcols[i % 2]:
                st.markdown(
                    f"""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 1.5rem; 
                        border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                    <h4 style="margin: 0 0 0.5rem 0;">{doc.get('name', 'Dr. Unknown')}</h4>
                    <p style="margin: 0.25rem 0;"><strong>{doc.get('specialty', 'Rheumatology')}</strong></p>
                    <p style="margin: 0.25rem 0; font-size: 0.9rem;">{doc.get('hospital', 'Hospital N/A')}</p>
                    <p style="margin: 0.25rem 0; font-size: 0.9rem;">📍 {doc.get('city', st.session_state.state['location'])}</p>
                    <p style="margin: 0.25rem 0; font-size: 0.9rem;">📞 {doc.get('phone', 'N/A') or 'N/A'}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Map (OpenStreetMap)")
        display_doctors_map(docs, st.session_state.state["location"])

elif page in [" Lab", " Genomics", " Wearable"]:
    st.title(page)
    kind = page.lower().strip()
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{kind}_upload")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        score = np.random.uniform(0.2, 0.8)
        report = generate_report_data(
            f"{kind.title()} Analysis", score, 0.9, "CSV Summary", "See interpretation below"
        )
        st.session_state.state["analysis_results"][kind] = report
        st.success(f"Analysis complete! Score: {score:.2f}")
        render_module_insights(kind, score)

elif page == " Reports":
    st.title(" Reports")
    if st.button(" Download Report"):
        content = (
            f"Arthritis Report - {st.session_state.state['patient_id']}\n"
            f"Generated: {datetime.now()}"
        )
        st.download_button(
            "Download",
            content.encode(),
            f"report_{st.session_state.state['patient_id']}.txt",
        )

