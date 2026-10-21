"""Arthritis Care Assistant Pro (GUI)

- Fixed crash: robust OpenStreetMap doctor map rendering (no KeyError: 'lat').
- Removed chatbot UI.
- Report generation now includes:
  * Doctor recommendations
  * Precautions
  * Medicines/recommendations
  * Patient tips
  * Module evidence (from Full Assessment)

Note: This file is intentionally self-contained and syntactically valid.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from PIL import Image
from streamlit_folium import st_folium

# ==============================
# Safe path setup for package imports
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT
for parent in PROJECT_ROOT.parents:
    if (parent / "modules").exists() and (parent / "integration").exists():
        WORKSPACE_ROOT = parent
        break

for candidate in {str(WORKSPACE_ROOT), str(Path.cwd()), str(Path.cwd().parent)}:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from integration.geospatial_network.provider_ranker import get_all_doctors

# Optional API key support (if you use Google Maps)
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")

# ==============================
# UTILITY
# ==============================

def generate_report_data(title: str, score: float, confidence: float, details: str = "", interpretation: str = ""):
    return {
        "title": title,
        "score": float(score),
        "confidence": float(confidence),
        "details": details,
        "interpretation": interpretation,
    }


def render_report_html(report: Dict, patient_id: str):
    score = float(report.get("score", 0.5))
    conf = float(report.get("confidence", 0.85))
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
        <p style="color: #475569; margin: 0 0 1rem 0;"><strong>Details:</strong> {report.get('details','')}</p>
        <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
            <strong style="color: #1e293b;">Patient Explanation:</strong><br>
            <span style="color: #475569; font-size: 0.95rem;">{report.get('interpretation','')}</span>
        </div>
    </div>
    """


# ==============================
# CHATBOT (deleted UI) - keep logic minimal
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
    "arthritis": " **Arthritis types**: 1 Osteoarthritis (wear/tear), 2 Rheumatoid (autoimmune), 3 Gout (uric acid), 4 Psoriatic.",
    "rheumatoid": " **Rheumatoid**: Morning stiffness >1hr, symmetric joints. DMARDs (Methotrexate first).",
    "osteoarthritis": "**Osteoarthritis**: Activity pain, stiffness <30min. Weight loss + physio best.",
    "gout": " **Gout**: Big toe attacks. Avoid beer/red meat. Allopurinol prevention.",
    "treatment": "**Treatment**: OA=Physio, RA=DMARDs, Gout=Allopurinol. See rheumatologist!",
    "diet": "**Anti-inflammatory**: Fish oil, berries, turmeric. Avoid beer/red meat.",
    "exercise": " **Safe**: Walking, swimming, yoga. 30min/day.",
    "doctor": " **Rheumatologist** for proper diagnosis & treatment plan.",
    "pain": "**Pain relief**: Paracetamol 500mg, topical gels, hot/cold therapy.",
    "symptoms": "**Symptoms**: Pain, stiffness, swelling, fatigue. Duration/location matters!",
    "default": " Ask about arthritis types (RA/OA/gout), treatment, diet, exercises, or doctors.",
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
    return " ".join(COMMON_MISSPELLINGS.get(w, w) for w in words)


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
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return str(payload.get("text", payload.get("response", line))).strip()
            except json.JSONDecodeError:
                pass
        return result.stdout.strip()
    return None


# ==============================
# DOCTORS + MAP
# ==============================


def get_real_doctors(city: str, count: int = 10, radius_m: int = 10000) -> List[Dict]:
    """Fetch hospitals/clinics/doctors from OpenStreetMap via Overpass."""
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

    Robust against missing lat/lon.
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

    center = [valid_docs[0]["lat"], valid_docs[0]["lon"]] if valid_docs else [18.5204, 73.8567]

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

    st_folium(m, width=700)


def get_doctors_by_location(
    location: str,
    arthritis_type: Optional[str] = None,
    count: int = 4,
    google_maps_api_key: Optional[str] = None,
) -> List[Dict]:
    """Return doctor list.

    Prefer curated provider_ranker doctors because they contain real Dr names, hospitals, and phone.
    Then fallback to OSM.
    """
    try:
        ranked = get_all_doctors(location, api_key=google_maps_api_key or API_KEY)
        if isinstance(ranked, dict):
            local_docs = ranked.get("local_doctors", []) or []
            indian_docs = ranked.get("indian_doctors", []) or []
            merged = (local_docs + indian_docs)[:count]
            # Ensure lat/lon keys exist (map will skip if None)
            for d in merged:
                d.setdefault("lat", None)
                d.setdefault("lon", None)
            # De-duplicate by name
            seen = set()
            uniq = []
            for d in merged:
                nm = d.get("name")
                if nm in seen:
                    continue
                seen.add(nm)
                uniq.append(d)
            if uniq:
                return uniq
    except Exception:
        pass

    # OSM fallback
    try:
        real = get_real_doctors(location, count=count)
        if real:
            return real
    except Exception:
        pass

    return [
        {
            "name": "Dr. Unknown",
            "specialty": "Rheumatology",
            "hospital": "Hospital N/A",
            "city": location,
            "phone": "N/A",
            "lat": None,
            "lon": None,
        }
    ]


# ==============================
# MOCK ML predictions
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
# MODULE INSIGHTS
# ==============================

MODULE_EXPLANATIONS = {
    "vision": {
        "title": "X-Ray Analysis - Spine, Fingers & Knees",
        "explanation": "X-ray imaging detects bone changes, cartilage damage, and alignment issues.",
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
        "explanation": "Voice analysis detects subtle changes in speech patterns correlated with pain and progression.",
        "interpretation": "Normal voice patterns suggest good arthritis control.",
        "medicines": {"low": "Rest and monitoring", "medium": "Physiotherapy, NSAIDs", "high": "DMARDs (Methotrexate), Biologic therapies"},
        "precautions": "Vocal rest if experiencing pain during speech.",
        "tips": "Stay hydrated and practice controlled breathing techniques.",
    },
    "lab": {
        "title": "Lab Analysis - Blood Tests & Clinical Markers",
        "explanation": "Lab tests measure inflammatory markers indicating disease activity.",
        "interpretation": "Elevated CRP/ESR may indicate active inflammation.",
        "medicines": {"low": "NSAIDs, routine monitoring", "medium": "DMARDs (Methotrexate 15mg/week)", "high": "Biologic DMARDs (TNF inhibitors), JAK inhibitors"},
        "precautions": "Monitoring for liver/kidney and blood counts when on DMARDs.",
        "tips": "Maintain medication adherence and report side effects immediately.",
    },
    "genomics": {
        "title": "Genetic Risk Assessment",
        "explanation": "Genomic analysis identifies genetic predispositions.",
        "interpretation": "Genetic profiling helps predict risk and treatment response.",
        "medicines": {"low": "Standard NSAIDs, lifestyle management", "medium": "DMARDs based on genetic profile", "high": "Targeted biologics"},
        "precautions": "Discuss findings with a genetic counselor; consider family screening.",
        "tips": "Use results to personalize prevention and lifestyle.",
    },
    "wearable": {
        "title": "Wearable IoT Analysis - Activity & Sleep Patterns",
        "explanation": "Wearable sensors track activity and sleep that correlate with flares.",
        "interpretation": "Poor sleep/activity can worsen inflammation.",
        "medicines": {"low": "Maintain activity, NSAIDs PRN", "medium": "Scheduled NSAIDs + exercise plan", "high": "DMARDs + NSAIDs"},
        "precautions": "Avoid overexertion during flares; ensure adequate sleep.",
        "tips": "Track sleep/steps and adjust activity gradually.",
    },
}


def render_module_insights(module_name: str, score: float):
    exp = MODULE_EXPLANATIONS.get(module_name)
    if not exp:
        return

    if score < 0.4:
        med = exp["medicines"]["low"]
    elif score < 0.7:
        med = exp["medicines"]["medium"]
    else:
        med = exp["medicines"]["high"]

    st.markdown(f"**Quick Recommendations for {exp['title']}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**💊 Medicines:**\n{med}")
    with c2:
        st.write(f"**⚠️ Precautions:**\n{exp.get('precautions','Follow up with provider')}")
    with c3:
        st.write(f"**💡 Patient Tips:**\n{exp.get('tips','Follow general advice')}")


# ==============================
# STREAMLIT APP
# ==============================

st.set_page_config(page_title="Arthritis Care Assistant Pro", layout="wide")


@st.cache_data
def init_session():
    return {
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
    st.session_state.state["patient_id"] = st.text_input(" Patient ID", st.session_state.state["patient_id"])
    st.session_state.state["location"] = st.text_input(" City", st.session_state.state["location"])

    page = st.radio(
        " Navigation",
        [
            " Dashboard",
            " Vision",
            " Voice",
            " Lab",
            " Genomics",
            " Wearable",
            # Chatbot intentionally removed
            " Full Assessment",
            " Reports",
        ],
    )


# ---------- Dashboard / Vision / Voice / Lab / Genomics / Wearable ----------

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
            render_module_insights("vision", float(score))

elif page == " Voice":
    st.title(" Voice Analysis")
    uploaded = st.file_uploader("Upload WAV", type=["wav"], key="voice_upload")
    if uploaded:
        st.audio(uploaded)
        _, conf, _, interp = run_voice_analysis(uploaded)
        report = generate_report_data("Voice Analysis", float(conf), float(conf), "Status", interp)
        st.session_state.state["analysis_results"]["voice"] = report
        st.markdown(render_report_html(report, st.session_state.state["patient_id"]), unsafe_allow_html=True)
        render_module_insights("voice", float(conf))

elif page in [" Lab", " Genomics", " Wearable"]:
    st.title(page)
    kind = page.lower().strip()
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{kind}_upload")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        score = float(np.random.uniform(0.2, 0.8))
        report = generate_report_data(f"{kind.title()} Analysis", score, 0.9, "CSV Summary", "See interpretation below")
        st.session_state.state["analysis_results"][kind] = report
        st.success(f"Analysis complete! Score: {score:.2f}")
        render_module_insights(kind, score)

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
        wearable = st.file_uploader(" Wearable IoT CSV", type=["csv"], key="fa_wear")

    if st.button(" RUN FULL ASSESSMENT", type="primary"):
        results: Dict[str, float] = {}
        details_text = ""

        if xray:
            score, _, detail = run_vision_analysis(xray)
            results["vision"] = float(score)
            details_text += f"\n✓ **Vision:** {detail}"
        if voice:
            _, score, _, detail = run_voice_analysis(voice)
            results["voice"] = float(score)
            details_text += f"\n✓ **Voice:** {detail}"
        if lab:
            score, _, detail = run_lab_analysis(lab)
            results["lab"] = float(score)
            details_text += f"\n✓ **Lab:** {detail}"
        if genomics:
            score, _, detail = run_genomics_analysis(genomics)
            results["genomics"] = float(score)
            details_text += f"\n✓ **Genomics:** {detail}"
        if wearable:
            score, _, detail = run_wearable_analysis(wearable)
            results["wearable"] = float(score)
            details_text += f"\n✓ **Wearable:** {detail}"

        overall = float(np.mean(list(results.values()))) if results else 0.5

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
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white; padding: 2rem; border-radius: 20px;">
                <h1 style="margin: 0;">{assessment['diagnosis']}</h1>
                <p style="margin: 0.5rem 0;">Overall Risk Score: <strong>{assessment['risk']:.1%}</strong></p>
                <p style="margin: 0;">Diagnostic Confidence: {assessment['confidence']:.0%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.info(assessment["details"])

        st.subheader("Personalized Treatment Plan")
        for module_name, score in assessment["results"].items():
            exp_data = MODULE_EXPLANATIONS.get(module_name)
            if not exp_data:
                continue
            with st.expander(f"📋 {exp_data['title']} (Score: {score:.1%})"):
                if module_name == "vision":
                    st.markdown(f"**Your Results:**\n{exp_data['spine']}\n\n{exp_data['fingers']}\n\n{exp_data['knees']}")
                else:
                    st.markdown(f"**Interpretation:**\n{exp_data.get('interpretation', exp_data['explanation'])}")

                if score < 0.4:
                    med_rec = exp_data["medicines"]["low"]
                elif score < 0.7:
                    med_rec = exp_data["medicines"]["medium"]
                else:
                    med_rec = exp_data["medicines"]["high"]

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**💊 Recommended Medicines:**\n{med_rec}")
                with c2:
                    st.write(f"**⚠️ Precautions:**\n{exp_data['precautions']}")
                with c3:
                    st.write(f"**💡 Patient Tips:**\n{exp_data['tips']}")

        st.subheader("🏥 Recommended Specialists")
        docs = get_doctors_by_location(st.session_state.state["location"], assessment["diagnosis"], 6)
        display_doctors_map(docs, st.session_state.state["location"])

elif page == " Reports":
    st.title(" Reports")

    assessment = st.session_state.state.get("full_assessment")
    if not assessment:
        st.info("Run Full Assessment first to generate a detailed report.")
    else:
        docs = get_doctors_by_location(st.session_state.state["location"], assessment["diagnosis"], 4)

        precautions: List[str] = []
        medicines: List[str] = []
        tips: List[str] = []

        for module_name, score in assessment["results"].items():
            exp = MODULE_EXPLANATIONS.get(module_name)
            if not exp:
                continue
            if score < 0.4:
                med_rec = exp["medicines"]["low"]
            elif score < 0.7:
                med_rec = exp["medicines"]["medium"]
            else:
                med_rec = exp["medicines"]["high"]

            medicines.append(f"[{module_name}] {med_rec}")
            precautions.append(f"[{module_name}] {exp['precautions']}")
            tips.append(f"[{module_name}] {exp['tips']}")

        doctor_lines = [
            f"- {d.get('name','Dr. Unknown')} | {d.get('specialty','Rheumatology')} | {d.get('hospital','Hospital N/A')} | {d.get('city',st.session_state.state['location'])} | {d.get('phone','N/A')}"
            for d in docs
        ]

        content = (
            f"Arthritis Report\n"
            f"Patient ID: {st.session_state.state['patient_id']}\n"
            f"Generated: {datetime.now()}\n\n"
            f"Diagnosis (based on available modules): {assessment['diagnosis']}\n"
            f"Overall Risk: {assessment['risk']:.1%}\n"
            f"Confidence: {assessment['confidence']:.0%}\n\n"
            f"=== Doctor Recommendations ===\n" + "\n".join(doctor_lines) + "\n\n"
            f"=== Precautions ===\n" + "\n".join(precautions) + "\n\n"
            f"=== Recommendations (Medicines) ===\n" + "\n".join(medicines) + "\n\n"
            f"=== Patient Tips ===\n" + "\n".join(tips) + "\n\n"
            f"=== Module Evidence ===\n" + assessment.get("details", "") + "\n"
        )

        if st.button(" Download Report"):
            st.download_button(
                "Download",
                content.encode("utf-8"),
                f"report_{st.session_state.state['patient_id']}.txt",
            )