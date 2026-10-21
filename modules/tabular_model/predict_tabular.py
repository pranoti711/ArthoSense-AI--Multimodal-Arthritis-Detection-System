import os
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_sample_data(output_path=None):
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "labs_clinical" / "new_samples.csv"
    output_path = Path(output_path)
    features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
    num_samples = 10  
    
    dummy_df = pd.DataFrame({
        'ESRh': np.random.uniform(0, 50, num_samples),
        'ESRo': np.random.uniform(0, 50, num_samples),
        'CRP': np.random.uniform(0, 50, num_samples),
        'Uric_Acid': np.random.uniform(0, 10, num_samples),
        'ASO': np.random.uniform(0, 500, num_samples),
        'age_binned': np.random.choice([0, 1, 2, 3], num_samples),
        'sex_encoded': np.random.choice([0, 1], num_samples)
    })
    
    os.makedirs(str(Path(output_path).parent), exist_ok=True)  
    dummy_df.to_csv(str(output_path), index=False)
    print(f"Sample data created and saved to {output_path}")
    return dummy_df

def predict_tabular(csv_path):
    new_data_path = Path(csv_path)
    processed_dir = PROJECT_ROOT / "data" / "processed" / "labs_clinical" / "arthritis_profile_processed"
    model = joblib.load(str(processed_dir / "trained_model.pkl")) 
    scaler = joblib.load(str(processed_dir / "scaler.pkl"))  
    if not new_data_path.exists():
        print(f"New data file not found at {new_data_path}. Creating a sample file...")
        new_df = create_sample_data(str(new_data_path))
    else:
        new_df = pd.read_csv(str(new_data_path))
    
    features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
    
    
    if not all(feature in new_df.columns for feature in features):
        print("Error: New data is missing some required features.")
        return None
    
    X_new = new_df[features]  
    
    try:
        X_new_scaled = scaler.transform(X_new)  
        X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=features)  
        
        predictions = model.predict(X_new_scaled_df) 
        probabilities = model.predict_proba(X_new_scaled_df)[:, 1] 
        
        new_df['prediction'] = predictions
        new_df['probability'] = probabilities
        
        output_path = processed_dir / "predictions.csv"
        new_df.to_csv(str(output_path), index=False)
        print(f"Predictions saved to {output_path}")
        return probabilities[0]  # Return the probability for the first row
    except Exception as e:
        print(f"Error during predictions: {e}")
        return None

if __name__ == "__main__":
    print("Starting predictions...")
    predict_tabular(PROJECT_ROOT / "data" / "processed" / "labs_clinical" / "new_samples.csv")
