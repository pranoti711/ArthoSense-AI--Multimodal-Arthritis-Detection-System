import json
import os
import sys
from pathlib import Path

import numpy as np

# Fix project root on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from modules.genomics_model.predict_genomics import predict_genomics
from modules.voice_model.predict import predict as predict_voice
from modules.vision_model.preprocess_knee_fixed import predict_knee
from modules.temporal_model.predict_temporal_fixed import predict_temporal
from modules.tabular_model.predict_tabular import predict_tabular

MODEL_PATH = os.path.join(root_dir, "models", "fusion_model.h5")
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
NEUTRAL_SCORE = 0.5


def load_mock_explanations():
    data_path = os.path.join(root_dir, "data", "mock_data.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("EXPLANATIONS", {})
    return {
        "genomics": "Genomics risk is unavailable; defaulting to neutral estimate.",
        "voice": "Voice score unavailable; using neutral baseline.",
        "imaging": "Imaging score unavailable; using neutral baseline.",
        "temporal": "Temporal progression unavailable; using neutral baseline.",
        "lab": "Lab signal unavailable; using neutral baseline."
    }


def load_model():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fusion model not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def normalize_score(value):
    try:
        value = float(value)
    except Exception:
        return NEUTRAL_SCORE
    return max(0.0, min(1.0, value))


def voice_score_from_prediction(prediction):
    if prediction is None:
        return NEUTRAL_SCORE

    # Handle tuple (label, confidence) from predict_voice()
    if isinstance(prediction, (tuple, list)) and len(prediction) == 2:
        label = str(prediction[0]).lower()
        confidence = float(prediction[1])
    elif isinstance(prediction, dict):
        label = str(prediction.get("label", "")).lower()
        confidence = float(prediction.get("confidence", 0.5))
    elif isinstance(prediction, (float, int)):
        label = ""
        confidence = float(prediction)
    else:
        return NEUTRAL_SCORE

    if "inflammation" in label:
        return normalize_score(0.6 + 0.35 * confidence)
    if "fatigue" in label:
        return normalize_score(0.45 + 0.3 * confidence)
    if "normal" in label:
        return normalize_score(0.2 + 0.2 * (1.0 - confidence))

    return normalize_score(confidence)


def risk_label_from_score(score):
    if score < 0.33:
        return RISK_LABELS[0]
    if score < 0.66:
        return RISK_LABELS[1]
    return RISK_LABELS[2]


def blend_scores(scores):
    weights = {
        "genomics": 0.24,
        "voice": 0.24,
        "imaging": 0.24,
        "temporal": 0.14,
        "lab": 0.14
    }
    total_weight = 0.0
    weighted_sum = 0.0

    for modality, score in scores.items():
        if score is None:
            continue
        weighted_sum += score * weights.get(modality, 0.0)
        total_weight += weights.get(modality, 0.0)

    if total_weight == 0.0:
        return NEUTRAL_SCORE
    return normalize_score(weighted_sum / total_weight)


def compute_confidence(score, available_count, fallback=False):
    base = 0.45 if fallback else 0.55
    boost = min(0.35, 0.05 * available_count)
    variance = abs(score - 0.5) * 0.3
    return float(max(0.45, min(0.99, base + boost + variance)))


def predict_final_enhanced(genomics_input, voice_input, imaging_input, temporal_input, lab_input):
    explanations = load_mock_explanations()
    paths_used = {
        "genomics": genomics_input,
        "voice": voice_input,
        "imaging": imaging_input,
        "temporal": temporal_input,
        "lab": lab_input
    }
    scores = {k: None for k in paths_used.keys()}
    metadata = {k: {} for k in paths_used.keys()}

    if genomics_input:
        try:
            scores["genomics"] = normalize_score(predict_genomics(genomics_input))
            metadata["genomics"]["source"] = "genomics_model"
        except Exception as e:
            metadata["genomics"]["error"] = str(e)

    if voice_input:
        try:
            voice_pred = predict_voice(voice_input)
            scores["voice"] = voice_score_from_prediction(voice_pred)
            metadata["voice"]["raw_output"] = voice_pred
        except Exception as e:
            metadata["voice"]["error"] = str(e)

    if imaging_input:
        try:
            scores["imaging"] = normalize_score(predict_knee(imaging_input))
            metadata["imaging"]["source"] = "vision_model"
        except Exception as e:
            metadata["imaging"]["error"] = str(e)

    if temporal_input:
        try:
            scores["temporal"] = normalize_score(predict_temporal(temporal_input))
            metadata["temporal"]["source"] = "temporal_model"
        except Exception as e:
            metadata["temporal"]["error"] = str(e)

    if lab_input:
        try:
            scores["lab"] = normalize_score(predict_tabular(lab_input))
            metadata["lab"]["source"] = "tabular_model"
        except Exception as e:
            metadata["lab"]["error"] = str(e)

    modalities_used = [k for k, v in scores.items() if v is not None]
    available_core = [scores[m] for m in ["genomics", "voice", "imaging"] if scores.get(m) is not None]

    if len(available_core) == 3:
        try:
            model = load_model()
            model_input = np.array([[scores["genomics"], scores["voice"], scores["imaging"]]])
            prediction = model.predict(model_input, verbose=0)
            class_idx = int(np.argmax(prediction[0]))
            risk = RISK_LABELS[class_idx]
            confidence = float(np.max(prediction[0]))
            score = float(np.dot(prediction[0], [0.1, 0.5, 0.9]))
            fusion_used = "fusion_model"
        except Exception as e:
            score = blend_scores(scores)
            risk = risk_label_from_score(score)
            confidence = compute_confidence(score, len(modalities_used), fallback=True)
            fusion_used = "fallback_average"
            metadata["fusion_error"] = str(e)
    else:
        score = blend_scores(scores)
        risk = risk_label_from_score(score)
        confidence = compute_confidence(score, len(modalities_used), fallback=True)
        fusion_used = "fallback_average"

    result = {
        "risk": risk,
        "score": score,
        "confidence": confidence,
        "fusion_method": fusion_used,
        "scores": scores,
        "paths": paths_used,
        "modalities_used": modalities_used,
        "explanation": {modality: explanations.get(modality, "No explanation available.") for modality in scores.keys()},
        "summary": f"Predicted {risk} using {len(modalities_used)} available modalities.",
        "metadata": metadata
    }

    return result


def predict_final(genomics_input, voice_input, imaging_input, temporal_input, lab_input):
    result = predict_final_enhanced(genomics_input, voice_input, imaging_input, temporal_input, lab_input)
    return result["risk"], result["confidence"]


def find_default_knee_image():
    root = Path(root_dir)
    search_dirs = [root / "data" / "processed" / "imaging", root / "data" / "raw" / "imaging"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for path in search_dir.rglob(ext):
                return str(path)
    return None


if __name__ == "__main__":
    knee_image = find_default_knee_image() or "test_knee_xray.png"
    print("✅ Arthritis Multimodal Fusion Ready")
    result = predict_final_enhanced(
        None,
        None,
        knee_image,
        None,
        None
    )
    print(result)

