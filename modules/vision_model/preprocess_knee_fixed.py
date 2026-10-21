import os
import cv2
import zipfile
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(__file__).resolve().parent  # modules/vision_model/

def predict_knee(image_path):
    """
    Predict knee OA grade from image.
    Returns score (0-1) based on grade / 4.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found, returning default score 0.5")
        return 0.5
    
    # Try local models
    model_paths = [
        MODEL_DIR / "arthritis_model_knee.pth",
        MODEL_DIR / "oai_knee_model.pth",
        BASE_DIR / "mlops/model_registry/knee_model.pt"
    ]
    
    model_path = None
    for mp in model_paths:
        if mp.exists():
            model_path = mp
            print(f"✅ Using model: {model_path}")
            break
    
    if not model_path:
        print("❌ No knee model found. Available: arthritis_model_knee.pth, oai_knee_model.pth")
        return 0.5

    # Load model
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)  # 5 classes for grades 0-4
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e} - using default 0.5")
        return 0.5

    # Preprocess
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image: {e}, returning default score 0.5")
        return 0.5
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Return score as grade / 4.0
    return prediction / 4.0

if __name__ == "__main__": 
    # Test with CWD image
    score = predict_knee("../test_knee_xray.png")
    print(f"Test score: {score}")

