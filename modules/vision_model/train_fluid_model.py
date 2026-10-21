# =============================================================================
# SCRIPT: Synovial Fluid Analysis Training (MFII Dataset)
# ENVIRONMENT: Google Colab (Executed)
# MODEL: ResNet50 (Feature Extractor) + K-Means Clustering
# DESCRIPTION: Extracts texture features from microscopy photos and groups them
#              into 'Healthy' vs 'Inflamed' clusters.
# =============================================================================

import os
import glob
import numpy as np
import pickle
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to the 'Photo' folder in the MFII dataset
TARGET_FOLDER = 'dataset/imaging/mfii_kaggle/DCIM/Photo' 
MODEL_SAVE_NAME = 'fluid_cluster_model.pkl'

def train_final_model():
    print(f"🎯 Locking onto folder: {TARGET_FOLDER}")

    if not os.path.exists(TARGET_FOLDER):
        print(f"⚠️ Note: Path '{TARGET_FOLDER}' not found locally.")
        print("   (This script is for documentation of the Colab training process.)")
        # In a real run, we would return here, but for docs we continue logic
        # return

    # 2. FIND IMAGES
    image_paths = []
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']

    for ext in extensions:
        # Recursive search pattern
        found = glob.glob(os.path.join(TARGET_FOLDER, ext))
        image_paths.extend(found)

    image_paths = list(set(image_paths)) # Remove duplicates
    print(f"📊 Found {len(image_paths)} valid photos for training.")

    # 3. INITIALIZE BRAIN (ResNet50)
    print("\n🧠 Loading ResNet50 (The Vision AI)...")
    # include_top=False removes the classification layer, leaving only feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # 4. EXTRACT FEATURES
    print("🚀 Reading images and extracting patterns...")
    features_list = []

    # (In documentation, we simulate the loop logic)
    for i, img_path in enumerate(image_paths):
        try:
            img = keras_image.load_img(img_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Extract the "mathematical vectors" of the image
            features = base_model.predict(x, verbose=0)
            features_list.append(features.flatten())
        except Exception as e:
            print(f"   ⚠️ Skipping unreadable file: {os.path.basename(img_path)}")

    # 5. TRAIN CLUSTERING
    if features_list:
        X = np.array(features_list)
        print(f"\n🤖 Training AI to split {X.shape[0]} images into 2 groups...")

        # K-Means groups images based on visual similarity (Texture)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X)

        # 6. SAVE THE BRAIN
        with open(MODEL_SAVE_NAME, 'wb') as f:
            pickle.dump(kmeans, f)

        print("🎉 SUCCESS! Model Trained.")
        print(f"💾 Model saved as '{MODEL_SAVE_NAME}'")
    else:
        print("❌ No features extracted (Dataset missing locally).")

if __name__ == "__main__":
    train_final_model()