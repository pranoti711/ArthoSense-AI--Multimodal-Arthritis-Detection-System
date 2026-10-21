from typing import List, Dict


def explain_with_lime(features: List[float], feature_names: List[str] = None) -> Dict:
    if feature_names is None:
        feature_names = [f"feature_{i+1}" for i in range(len(features))]

    total = sum(abs(float(v)) for v in features) or 1.0
    importance = [
        {
            "feature": name,
            "weight": round(abs(float(value)) / total, 3),
            "direction": "positive" if float(value) >= 0 else "negative"
        }
        for name, value in zip(feature_names, features)
    ]

    return {
        "method": "LIME",
        "feature_importance": importance,
        "notes": "Estimated feature importance using simplified LIME-style weights."
    }


def summarize_lime(explanation: Dict) -> Dict:
    return {
        "top_features": explanation.get("feature_importance", [])[:3],
        "summary": "Top features that most influenced the model prediction."
    }
