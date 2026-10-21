

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------
# PATH TO RAW CSV FILE WITH 5 PARAMETERS
# -------------------------------------------------------------
RAW_DATA_PATH = r'C:\Users\Pranoti munjankar\Desktop\arthritis-detection-care-assistant-local\data\raw\wearables_iot_demographic.csv'
# -------------------------------------------------------------
# SAFE CSV READER
# -------------------------------------------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


# -------------------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# -------------------------------------------------------------
def preprocess_demographic():
    print("📥 Loading demographic dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ CSV not found at: {RAW_DATA_PATH}")

    df = safe_read_csv(RAW_DATA_PATH)
    print("➡ Columns found:", df.columns.tolist())

    # Required columns
    required_cols = ["code", "weight", "height", "age", "gender"]

    # Make sure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠ Missing column: {col} → filling with default value 0")
            df[col] = 0

    # Clean numeric columns
    for col in ["weight", "height", "age", "code"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert gender to numeric (male=1, female=0)
    df["gender"] = df["gender"].astype(str).str.lower()
    df["gender"] = df["gender"].map({"male": 1, "female": 0})
    df["gender"] = df["gender"].fillna(0)

    # Add dummy target label (needed for training)
    df["flare_label"] = 0

    print("✅ Preprocessing complete.")
    return df


# -------------------------------------------------------------
# SAVE PROCESSED FILE
# -------------------------------------------------------------
if __name__ == "__main__":
    df_processed = preprocess_demographic()

    SAVE_PATH = r"C:\Users\Pranoti munjankar\Desktop\arthritis-detection-care-assistant-local\data\raw\wearables_iot\motionsense_kaggle\data_subjects_info.csv"
    df_processed.to_csv(SAVE_PATH, index=False)

    print("🎉 File saved successfully at:")
    print(SAVE_PATH)
