import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
processed_dir = PROJECT_ROOT / "data" / "processed" / "labs_clinical" / "arthritis_profile_processed"
df = pd.read_parquet(str(processed_dir / "multimodal.parquet"))

print(f"Loaded processed data: {df.shape} rows/cols. Columns: {df.columns.tolist()}")

print("Data Summary:")
print(df.describe())
print("Target balance:", df['target_encoded'].value_counts().to_dict())

features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
target = 'target_encoded'

X = df[features]
y = df[target]

model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    for feature, importance in zip(features, importances):
        print(f"{feature}: {importance:.4f}")

model_path = processed_dir / "trained_model.pkl"
joblib.dump(model, str(model_path))
print(f"Model saved at {model_path}")

if __name__ == "__main__":
    print("Starting model training...")
    
