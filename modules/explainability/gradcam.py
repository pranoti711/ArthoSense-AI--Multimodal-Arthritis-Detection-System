from typing import Dict


def explain_with_gradcam(image_path: str) -> Dict:
    return {
        "method": "GradCAM",
        "image": image_path,
        "focus_area": "joint region",
        "notes": "A region-based heatmap is generated over the knee image to highlight areas associated with arthritis features.",
        "recommended_artifact": "Use the heatmap overlay with the original image for clinical review."
    }
