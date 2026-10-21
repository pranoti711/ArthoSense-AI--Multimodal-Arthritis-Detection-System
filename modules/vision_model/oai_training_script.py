# =============================================================================
# PROJECT: Knee Osteoarthritis Detection (OAI Dataset)
# ENVIRONMENT: Google Colab (T4 GPU)
# MODEL ARCHITECTURE: DenseNet121 (Pretrained on ImageNet)
# PREPROCESSING: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# =============================================================================

import os
import zipfile
import time
import copy
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")

# ==========================================
# 1. DATA EXTRACTION (Google Drive)
# ==========================================
# This section handles data retrieval from Google Drive storage
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    ZIP_PATH = '/content/drive/MyDrive/imaging.zip'
    EXTRACT_PATH = '/content/dataset'

    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)

    print("🚀 Extracting OAI Dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        # Extract only the relevant 'oai_kaggle' folder
        files = [f for f in zip_ref.namelist() if 'oai_kaggle' in f]
        zip_ref.extractall(EXTRACT_PATH, members=files)
    print("✅ Extraction Complete.")
    
    TRAIN_DIR = os.path.join(EXTRACT_PATH, 'imaging', 'oai_kaggle', 'train')
    VAL_DIR = os.path.join(EXTRACT_PATH, 'imaging', 'oai_kaggle', 'val')

except ImportError:
    print("⚠️ Google Colab environment not detected. skipping Drive mount.")
    # Fallback paths for documentation purposes
    TRAIN_DIR = './dataset/train'
    VAL_DIR = './dataset/val'

# ==========================================
# 2. ADVANCED PREPROCESSING (CLAHE)
# ==========================================
class ApplyCLAHE(object):
    """
    Custom Transform: Applies Contrast Limited Adaptive Histogram Equalization.
    Why: Enhances local contrast to make bone spurs (osteophytes) and 
    joint space narrowing visible in low-contrast X-rays.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        
        # Ensure image is grayscale
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe = clahe.apply(img_gray)
        
        # Convert back to RGB for DenseNet compatibility
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

# ==========================================
# 3. DATA PIPELINE & AUGMENTATION
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),       # Standard Input Size for DenseNet
        ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8,8)), # Preprocessing
        transforms.RandomHorizontalFlip(),   # Augmentation: Handle L/R orientation
        transforms.RandomRotation(10),       # Augmentation: Handle patient positioning
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8,8)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# ==========================================
# 4. TRAINING CONFIGURATION
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + Optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 5. EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # Initialize Datasets
    image_datasets = {x: datasets.ImageFolder(eval(f"{x.upper()}_DIR"), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # Initialize Model (Transfer Learning)
    # Using DenseNet121 for high feature reuse and accuracy on medical images
    model = models.densenet121(weights='IMAGENET1K_V1')
    
    # Modify Classifier for 5 KL Grades (0-4)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 5)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Run Training
    trained_model = train_model(model, criterion, optimizer, num_epochs=15)

    # Save Model
    torch.save(trained_model.state_dict(), 'oai_knee_model.pth')
    print("Model saved successfully.")