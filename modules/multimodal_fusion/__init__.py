import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from modules.multimodal_fusion.predict_fusion import predict_final_enhanced, predict_final
except ImportError:
    from .predict_fusion import predict_final_enhanced, predict_final

try:
    from modules.multimodal_fusion.predict_fusion_fixed import predict_final_enhanced as predict_final_enhanced_fixed, predict_final as predict_final_fixed
except ImportError:
    from .predict_fusion_fixed import predict_final_enhanced as predict_final_enhanced_fixed, predict_final as predict_final_fixed

__all__ = [
    "predict_final_enhanced",
    "predict_final",
    "predict_final_enhanced_fixed",
    "predict_final_fixed",
]
