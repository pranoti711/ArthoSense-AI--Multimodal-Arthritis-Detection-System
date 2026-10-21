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
# Detects if running on Google Colab or Local VS Code
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = Path('/content')
    ZIP_PATH = '/content/drive/My Drive/imaging.zip' 
    RAW_DIR = BASE_DIR / 'temp_raw_extract'
    PROCESSED_DIR = BASE_DIR / 'final_training_data'
else:
    # LOCAL VS CODE PATHS
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    ZIP_PATH = BASE_DIR / 'data/raw/imaging/imaging.zip'
    RAW_DIR = BASE_DIR / 'data/interim/temp_raw'
    PROCESSED_DIR = BASE_DIR / 'data/processed/imaging/knee_oa_processed'

def preprocess_knee_pipeline():
    # 1. Unzip
    if not os.path.exists(RAW_DIR):
        print(f"📂 Unzipping {ZIP_PATH}...")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
        except FileNotFoundError:
            print(f"❌ Error: Zip file not found at {ZIP_PATH}")
            return

    # 2. Find Knee Data
    print("🕵️‍♂️ Locating Knee X-ray folders...")
    knee_train_path = None
    for root, dirs, files in os.walk(RAW_DIR):
        # We look for the folder containing '0', '1', '2', '3', '4'
        if set(['0', '1', '2', '3', '4']).issubset(set(dirs)):
            knee_train_path = root
            print(f"✅ Found raw data at: {knee_train_path}")
            break
            
    if not knee_train_path:
        print("❌ Error: Could not find Knee OA folders (0-4) in zip.")
        return

    # 3. Process (CLAHE + Resize)
    print(f"⚙️ Processing images to {PROCESSED_DIR}...")
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    count = 0

    classes = ['0', '1', '2', '3', '4']
    for cls in classes:
        src_folder = os.path.join(knee_train_path, cls)
        dest_folder = os.path.join(PROCESSED_DIR, cls)
        os.makedirs(dest_folder, exist_ok=True)
        
        files = os.listdir(src_folder)
        for file in tqdm(files, desc=f"Class {cls}"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(src_folder, file), 0)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    img = clahe.apply(img) # Contrast Enhancement
                    cv2.imwrite(os.path.join(dest_folder, file), img)
                    count += 1

    print(f"\n✅ Preprocessing Complete! {count} images ready.")

if __name__ == "__main__":
    preprocess_knee_pipeline()


def predict_knee(image_path):
    """
    Predict knee OA grade from image.
    Returns score (0-1) based on grade / 4.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found, returning default score 0.5")
        return 0.5
    
    # Load model
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)  # 5 classes for grades 0-4
    model_path = Path(__file__).parent.parent.parent / "mlops" / "model_registry" / "knee_model.pt"
    print(f"🔍 Looking for knee model at: {model_path}")
    if not model_path.exists():
        model.eval()
        return 0.5
    try:
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(f"✅ Knee model loaded successfully!")
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
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Return score as grade / 4.0
    return prediction / 4.0