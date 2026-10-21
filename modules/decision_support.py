import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .patient_explanation import get_patient_explanation, format_explanation
from integration.geospatial_network.provider_ranker import get_all_doctors, get_indian_doctors

MOCK_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mock_data.json"
TRACKING_BY_ZIP = {
    "90210": {
        "city": "Beverly Hills, CA",
        "status": "active",
        "current_clinic": "Cedars-Sinai",
        "eta": "15 minutes",
        "coordinates": {"lat": 34.0736, "lon": -118.4004}
    },
    "10001": {
        "city": "New York, NY",
        "status": "active",
        "current_clinic": "Hospital for Special Surgery",
        "eta": "25 minutes",
        "coordinates": {"lat": 40.7506, "lon": -73.9970}
    }
}


def load_mock_data() -> Dict:
    if MOCK_DATA_PATH.exists():
        with open(MOCK_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "risk_guidelines": {},
        "local_doctors": {},
        "world_experts": []
    }


def get_risk_recommendations(risk_level: str) -> Dict:
    data = load_mock_data()
    return data['risk_guidelines'].get(risk_level, {})


def get_type_specific_local_doctors(
    arthritis_type: str,
    location: str,
    count: int = 5,
    api_key: Optional[str] = None
) -> List[Dict]:
    all_local = get_all_doctors(location, api_key=api_key)["local_doctors"]
    
    type_keywords = {
        "Osteoarthritis": ["osteo", "joint", "degenerative"],
        "Rheumatoid Arthritis": ["ra", "rheumatoid", "dmard", "autoimmune"],
        "Psoriatic Arthritis": ["psoriatic", "psa", "skin"],
        "Gout": ["gout", "uric", "crystal"],
        "Ankylosing Spondylitis": ["as", "spondylo", "axial", "hla-b27"]
    }
    
    keywords = type_keywords.get(arthritis_type, ["rheumatology"])
    scored_doctors = []
    
    for doc in all_local:
        score = sum(1 for kw in keywords if kw in str(doc).lower())
        if score > 0:
            doc["expertise_score"] = score
            scored_doctors.append(doc)
    
    scored_doctors.sort(key=lambda x: (-x["expertise_score"], -x.get("rating", 0)))
    return scored_doctors[:count]


def get_type_specific_country_doctors(arthritis_type: str, count: int = 5) -> List[Dict]:
    """Top country doctors filtered by type expertise"""
    all_indian = get_indian_doctors(10)  # Get more, filter
    
    type_keywords = {
        "Osteoarthritis": ["osteo", "gout"],
        "Rheumatoid Arthritis": ["ra", "spondylo"],
        "Psoriatic Arthritis": ["psoriatic"],
        "Gout": ["gout", "osteo"],
        "Ankylosing Spondylitis": ["ra", "spondylo"]
    }
    
    keywords = type_keywords.get(arthritis_type, ["ra"])
    scored_doctors = []
    
    for doc in all_indian:
        score = sum(1 for kw in keywords if kw in str(doc).lower())
        if score > 0 or "rheumatology" in doc.get("specialty", "").lower():
            doc["expertise_score"] = score
            scored_doctors.append(doc)
    
    scored_doctors.sort(key=lambda x: (-x["expertise_score"], -x.get("rating", 0)))
    return scored_doctors[:count]


def get_world_experts() -> List[Dict]:
    data = load_mock_data()
    return data['world_experts']


def get_medicines_by_type(arthritis_type: str) -> Dict[str, List[str]]:
    meds = {
        "Osteoarthritis": {
            "first_line": ["Paracetamol 500mg 3x/day", "Topical diclofenac gel", "Tramadol PRN"],
            "advanced": ["Duloxetine", "Hyaluronic acid injections", "PRP therapy"],
            "supplements": ["Glucosamine 1500mg", "Fish oil 2g", "Turmeric 500mg"]
        },
        "Rheumatoid Arthritis": {
            "first_line": ["Methotrexate 15mg/week", "Hydroxychloroquine 200mg 2x/day", "Prednisolone 5mg"],
            "advanced": ["Leflunomide", "TNF inhibitors (Etanercept)", "JAK inhibitors"],
            "labs_monitor": ["LFTs monthly", "CBC weekly first 3 months"]
        },
        "Gout": {
            "first_line": ["Allopurinol 100mg daily (titrate)", "Colchicine 0.6mg 2x/day acute"],
            "acute_attack": ["Prednisone 30mg x5 days", "NSAIDs"],
            "diet": ["No beer/shellfish", "3L water daily"]
        },
        "Psoriatic Arthritis": {
            "first_line": ["Methotrexate 15-20mg/week", "Sulfasalazine 2g/day"],
            "advanced": ["IL-17 inhibitors (Secukinumab)", "IL-23 inhibitors"],
            "skin_joint": ["Same meds treat both skin + joints"]
        },
        "Ankylosing Spondylitis": {
            "first_line": ["NSAIDs (Indomethacin 75mg 3x)", "Celecoxib 200mg 2x"],
            "advanced": ["TNF inhibitors (Infliximab)", "IL-17 inhibitors (Secukinumab)"],
            "physio": ["Daily stretching mandatory"]
        }
    }
    return meds.get(arthritis_type, {"first_line": ["Consult rheumatologist"]})


def get_nearest_doctors(zip_code: str = "90210") -> List[Dict]:
    from .decision_support_fixed import get_nearest_doctors as fixed_doctors
    return fixed_doctors(zip_code)


def get_patient_tracking(zip_code: str = "90210") -> Dict:
    return TRACKING_BY_ZIP.get(zip_code, {
        "city": "Unknown",
        "status": "location unavailable",
        "current_clinic": "Not assigned",
        "eta": "TBD",
        "coordinates": {}
    })


def classify_arthritis_type(
    genomics_score: float = None,
    voice_prediction: str = None,
    imaging_score: float = None,
    lab_score: float = None,
    temporal_score: float = None,
    wearable_score: float = None,
    symptoms_text: str = ""
) -> Tuple[str, float, Dict]:
    """
    Classify arthritis type based on all available modalities.
    Returns: (type_name, confidence, reasoning)
    """
    scores = {
        "Osteoarthritis": 0.0,
        "Rheumatoid Arthritis": 0.0,
        "Psoriatic Arthritis": 0.0,
        "Gout": 0.0,
        "Ankylosing Spondylitis": 0.0
    }
    
    reasons = {k: [] for k in scores.keys()}
    
    # Genomics markers
    if genomics_score is not None:
        if genomics_score > 0.6:
            scores["Rheumatoid Arthritis"] += 0.25
            reasons["Rheumatoid Arthritis"].append("High genetic risk markers (HLA-DRB1, PTPN22)")
        if genomics_score > 0.5:
            scores["Ankylosing Spondylitis"] += 0.15
            reasons["Ankylosing Spondylitis"].append("Genetic susceptibility detected")
    
    # Voice analysis
    if voice_prediction:
        vp = voice_prediction.lower()
        if "inflammation" in vp:
            scores["Rheumatoid Arthritis"] += 0.20
            scores["Psoriatic Arthritis"] += 0.15
            reasons["Rheumatoid Arthritis"].append("Voice biomarkers indicate inflammation")
            reasons["Psoriatic Arthritis"].append("Voice biomarkers indicate inflammation")
        elif "fatigue" in vp:
            scores["Rheumatoid Arthritis"] += 0.10
            reasons["Rheumatoid Arthritis"].append("Voice fatigue patterns detected")
    
    # Imaging analysis
    if imaging_score is not None:
        if imaging_score > 0.5:
            scores["Osteoarthritis"] += 0.30
            reasons["Osteoarthritis"].append(f"Imaging shows significant joint damage (grade {imaging_score:.2f})")
        if imaging_score > 0.7:
            scores["Rheumatoid Arthritis"] += 0.15
            reasons["Rheumatoid Arthritis"].append("Severe joint erosion visible")
    
    # Lab markers
    if lab_score is not None:
        if lab_score > 0.6:
            scores["Rheumatoid Arthritis"] += 0.20
            scores["Gout"] += 0.15
            reasons["Rheumatoid Arthritis"].append("Elevated inflammatory markers (CRP/ESR)")
            reasons["Gout"].append("Possible metabolic involvement")
        if lab_score > 0.7:
            scores["Gout"] += 0.20
            reasons["Gout"].append("High uric acid or ASO levels")
    
    # Temporal progression
    if temporal_score is not None:
        if temporal_score > 0.6:
            scores["Rheumatoid Arthritis"] += 0.10
            reasons["Rheumatoid Arthritis"].append("Rapid symptom progression pattern")
    
    # Wearable activity
    if wearable_score is not None:
        if wearable_score > 0.6:
            scores["Osteoarthritis"] += 0.10
            scores["Rheumatoid Arthritis"] += 0.10
            reasons["Osteoarthritis"].append("Declining activity levels")
            reasons["Rheumatoid Arthritis"].append("Activity limitation detected")
    
    # Symptom text analysis
    if symptoms_text:
        st_lower = symptoms_text.lower()
        keywords = {
            "Osteoarthritis": ["knee pain", "hip pain", "stiffness", "morning", "aging", "old", "wear"],
            "Rheumatoid Arthritis": ["both hands", "symmetrical", "morning stiffness", "fatigue", "swelling", "warm"],
            "Psoriatic Arthritis": ["skin", "psoriasis", "nail", "asymmetric", "toe", "finger"],
            "Gout": ["big toe", "sudden", "night", "severe pain", "red", "hot", "alcohol", "meat"],
            "Ankylosing Spondylitis": ["back pain", "spine", "morning", "young", "night pain", "stooping"]
        }
        for atype, kwords in keywords.items():
            matches = sum(1 for kw in kwords if kw in st_lower)
            if matches > 0:
                scores[atype] += 0.05 * matches
                reasons[atype].append(f"Symptom keywords matched: {matches}")
    
    # Determine best match
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    
    # Normalize confidence
    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0
    
    return best_type, confidence, {
        "all_scores": scores,
        "reasons": reasons[best_type] if reasons[best_type] else ["Based on general pattern matching"],
        "differential_diagnosis": sorted(
            [(k, v) for k, v in scores.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:3]
    }


def get_precautions_by_type(arthritis_type: str) -> List[str]:
    """Get type-specific precautions."""
    precautions = {
        "Osteoarthritis": [
            "Maintain healthy weight to reduce joint stress",
            "Use assistive devices for daily activities",
            "Apply heat/cold therapy for pain relief",
            "Take glucosamine/chondroitin supplements",
            "Avoid high-impact activities (running, jumping)",
            "Use knee braces or supports during activity",
            "Consider hyaluronic acid injections for knee OA"
        ],
        "Rheumatoid Arthritis": [
            "Take DMARDs exactly as prescribed - do not skip doses",
            "Monitor for infection signs (biologics suppress immunity)",
            "Get regular blood tests (liver function, blood counts)",
            "Gentle range-of-motion exercises daily",
            "Use splints during flares to protect joints",
            "Avoid smoking - it worsens RA progression",
            "Get annual eye exams (Sjogren's risk)"
        ],
        "Psoriatic Arthritis": [
            "Manage skin psoriasis with prescribed topical treatments",
            "Protect nails from trauma (PsA often affects nails)",
            "Monitor for eye inflammation (uveitis)",
            "Regular cardiovascular screening (increased heart risk)",
            "Maintain healthy weight",
            "Consider biologic therapy if joints are involved",
            "Use sunscreen to prevent skin flares"
        ],
        "Gout": [
            "Avoid purine-rich foods (red meat, organ meats, seafood)",
            "Limit alcohol consumption (especially beer)",
            "Stay well hydrated (2-3 liters water daily)",
            "Avoid fructose-sweetened beverages",
            "Take allopurinol regularly if prescribed",
            "During acute attack: rest, ice, elevate limb",
            "Monitor uric acid levels every 3-6 months"
        ],
        "Ankylosing Spondylitis": [
            "Daily stretching and posture exercises",
            "Sleep on firm mattress, avoid pillows that flex neck",
            "Stop smoking (worsens spinal fusion)",
            "Regular swimming (best exercise for spine)",
            "Monitor for osteoporosis (get DEXA scans)",
            "Consider TNF inhibitors if progression continues",
            "Maintain upright posture throughout the day"
        ],
        "Unknown": [
            "Consult a rheumatologist for proper diagnosis",
            "Maintain healthy lifestyle (diet, exercise, sleep)",
            "Track symptoms in a daily journal",
            "Avoid self-medicating with steroids",
            "Get comprehensive blood work and imaging",
            "Consider genetic testing (HLA-B27, RF, anti-CCP)"
        ]
    }
    return precautions.get(arthritis_type, precautions["Unknown"])


def generate_care_plan(
    risk_level: str, 
    patient_zip: str = "90210",
    arthritis_type: str = "Unknown",
    arthritis_confidence: float = 0.0,
    all_scores: Dict = None,
    precautions: List[str] = None
) -> Dict:
    """
    Enhanced comprehensive care plan with type-specific everything
    """
    patient_expl = get_patient_explanation(arthritis_type)
    meds = get_medicines_by_type(arthritis_type)
    
    plan = {
        "risk_level": risk_level,
        "arthritis_type": arthritis_type,
        "type_confidence": round(arthritis_confidence, 2),
        "patient_explanation": patient_expl,
        "formatted_explanation": format_explanation(arthritis_type, arthritis_confidence),
        "guidelines": get_risk_recommendations(risk_level),
        "nearest_doctors": get_type_specific_local_doctors(arthritis_type, patient_zip),
        "country_experts": get_type_specific_country_doctors(arthritis_type),
        "medications": meds,
        "precautions": precautions or get_precautions_by_type(arthritis_type),
        "tracking": get_patient_tracking(patient_zip),
        "next_steps": f"Consult top {arthritis_type} specialist within {7 if risk_level == 'High Risk' else 30} days",
        "type_reasoning": all_scores.get("reasons", []) if all_scores else [],
        "urgency": "HIGH" if risk_level == 'High Risk' else "MEDIUM"
    }
    return plan


if __name__ == "__main__":
    # Test
    atype, conf, reason = classify_arthritis_type(
        genomics_score=0.7,
        voice_prediction="inflammation",
        imaging_score=0.6,
        lab_score=0.5,
        symptoms_text="both hands swollen morning stiffness"
    )
    print(f"Type: {atype}, Confidence: {conf:.2f}")
    print(f"Reasons: {reason['reasons']}")
    print(f"\nPrecautions:")
    for p in get_precautions_by_type(atype):
        print(f"  - {p}")
    
    plan = generate_care_plan("High Risk", arthritis_type=atype, arthritis_confidence=conf, all_scores=reason)
    print(f"\nCare Plan: {plan['next_steps']}")

