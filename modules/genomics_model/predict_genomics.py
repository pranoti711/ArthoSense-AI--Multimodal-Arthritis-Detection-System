import torch
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from io import StringIO

# Fix path for standalone
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from modules.genomics_model.model import GenomicsModel

# List all SNPs your Genomics model uses
snps = ["rs2476601", "rs1801133", "HLA-B27", "PTPN22"]  # Add more if your model has more SNPs

def predict_genomics(input_data):
    """
    Predict RA risk from genomics CSV path or SNP list.
    Expects CSV with columns matching snps, or list of len(snps) [0/1].
    Returns risk score (0-1)
    """
    if isinstance(input_data, str):
        # CSV path
        df = pd.read_csv(input_data)
        
        if len(df) == 0:
            raise ValueError("CSV is empty")
        
        # Assume first row or single patient
        patient_data = df.iloc[0][snps].values.astype(float)
        
        # Validate SNPs columns
        if not all(col in df.columns for col in snps):
            raise ValueError(f"CSV missing required SNP columns. Expected: {snps}. Found: {list(df.columns)}")
    else:
        # List of SNPs
        patient_data = np.array(input_data, dtype=float)
        if len(patient_data) != len(snps):
            raise ValueError(f"Expected {len(snps)} SNPs, got {len(patient_data)}")
    
    X_tensor = torch.tensor(patient_data, dtype=torch.float32).unsqueeze(0)
    
    # Load model with fallback
    try:
        model = GenomicsModel(input_dim=len(snps))
        model_path = os.path.join(os.path.dirname(__file__), "../../mlops/model_registry/genomics_ra_model.pt")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Predict
        with torch.no_grad():
            prediction = model(X_tensor).item()
    except Exception as e:
        print(f"⚠️ Model load failed ({e}), using fallback: mean SNP risk")
        prediction = float(np.mean(patient_data))
    
    return prediction

print("✅ Genomics RA Risk Prediction module loaded")

# Interactive mode for standalone use
if __name__ == "__main__":
    print("Standalone test with genomics_sample.csv:")
    try:
        score = predict_genomics("../../../genomics_sample.csv")
        print(f"RA Risk Score for sample: {score:.4f}")
    except Exception as e:
        print(f"Sample test failed: {e}")
    
    print("\nInteractive mode:")
    # Ask user for number of patients
    try:
        n_patients = int(input("Enter number of patients to predict (or 0 to skip): "))
        if n_patients < 1:
            print("Skipping interactive.")
        else:
            all_inputs = []
            for i in range(n_patients):
                print(f"\nPatient {i+1}:")
                patient_input = []
                for snp in snps:
                    while True:
                        try:
                            val = int(input(f"  {snp} (0 or 1): "))
                            if val not in [0, 1]:
                                raise ValueError
                            patient_input.append(val)
                            break
                        except ValueError:
                            print("  Invalid! Enter 0 or 1")
                score = predict_genomics(patient_input)
                all_inputs.append(patient_input)
                print(f"  RA Risk Score: {score:.4f}")
            
            # Simple summary
            print("\nSummary:")
            for i, inp in enumerate(all_inputs):
                score = predict_genomics(inp)
                print(f"Patient {i+1}: score {score:.4f}")
    except KeyboardInterrupt:
        print("\nExiting.")

