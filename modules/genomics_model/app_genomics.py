from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import torch
import pandas as pd
from modules.genomics_model.model import GenomicsModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


snps = ["rs2476601", "rs1801133", "HLA-B27", "PTPN22"]  # match model input
model = GenomicsModel(input_dim=len(snps))
# Robustly load checkpoint on CPU (handles state_dict or full model)
ckpt_path = PROJECT_ROOT / "mlops" / "model_registry" / "genomics_ra_model.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu")
if isinstance(ckpt, dict):
   
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
else:
   
    model = ckpt
model.eval()


st.title("Genomics RA Risk Prediction")
st.write("Enter SNP values (0 = absent, 1 = present) for each patient.")


snp_values = []
for snp in snps:
    val = st.radio(f"{snp}", options=[0, 1], index=0)
    snp_values.append(val)

if st.button("Predict RA Risk"):
    X_tensor = torch.tensor([snp_values], dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor).item()

    st.subheader(f"Predicted RA risk probability: {pred:.2f}")

    if pred < 0.4:
        risk = "Low Risk"
    elif pred < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    st.subheader(f"Risk Category: {risk}")


