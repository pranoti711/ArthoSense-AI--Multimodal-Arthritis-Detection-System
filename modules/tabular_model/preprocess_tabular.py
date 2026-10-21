
import os
import uuid
import numpy as np
import pandas as pd
import joblib  
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_arthritis_profile(raw_file="data/raw/labs_clinical/arthritis_profile_kaggle/APDDataset.xlsx"):
    # Load Excel
    df = pd.read_excel(raw_file)
    print(f"Loaded Arthritis data: {df.shape} rows/cols. Columns: {df.columns.tolist()}")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    df['patient_uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    if 'Age' in df.columns:
        df['age_binned'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3])
    if 'Gender_M' in df.columns:
        le_sex = LabelEncoder()
        df['sex_encoded'] = le_sex.fit_transform(df['Gender_M'].astype(str))
    
    ra_vals = df.get('RA', pd.Series(0)).fillna(0)
    crp_vals = df.get('CRP', pd.Series(0)).fillna(0)
    uric_vals = df.get('Uric_Acid', pd.Series(0)).fillna(0)
    esr_vals = df.get('ESRh', pd.Series(0)).fillna(0)
    ra_high = ra_vals > 15
    crp_high = crp_vals > 10
    uric_high = uric_vals > 7
    esr_high = esr_vals > 30
    df['synthetic_target'] = (ra_high | crp_high | uric_high | esr_high).astype(int)
    target_col = df['synthetic_target']
    print("Synthetic target created (rules: RA>15 OR CRP>10 OR Uric>7 OR ESRh>30). Distribution:", target_col.value_counts().to_dict())
    
    feature_cols = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using features: {available_features}")
    
    df_subset = df[available_features + ['synthetic_target']].copy()
    df_subset = df_subset.rename(columns={'synthetic_target': 'target_raw'})
    
    
    print("Initial missing % per feature:")
    missing_pct = (df_subset.isnull().sum() / len(df_subset) * 100).round(1)
    print(missing_pct[missing_pct > 0])
    
    core_features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO']  
    nan_count = df_subset[core_features].isnull().sum(axis=1)
    df_subset = df_subset[nan_count <= 3]  
    print(f"Dropped {len(df) - len(df_subset)} rows with high missingness. Remaining: {len(df_subset)}")
    
   
    numeric_cols = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO']
    available_numeric = [col for col in numeric_cols if col in df_subset.columns]
    if len(available_numeric) > 0:
       
        avg_missing = missing_pct[available_numeric].mean()
        if avg_missing < 50:
            n_neighbors = min(3, len(df_subset) - 1)  
            imputer = KNNImputer(n_neighbors=n_neighbors)
            print(f"Using KNNImputer (n_neighbors={n_neighbors})")
            df_subset[available_numeric] = imputer.fit_transform(df_subset[available_numeric])
        else:
            imputer = SimpleImputer(strategy='mean')
            print("Using SimpleImputer (mean) for numeric columns due to high missingness")
            df_subset[available_numeric] = imputer.fit_transform(df_subset[available_numeric])
        
       
        remaining_nans_numeric = df_subset[available_numeric].isnull().sum().sum()
        print(f"NaNs after numeric imputation: {remaining_nans_numeric}")
    

    cat_cols = ['age_binned', 'sex_encoded']
    for col in cat_cols:
        if col in df_subset.columns:
            mode_val = df_subset[col].mode()[0] if not df_subset[col].mode().empty else 0
            df_subset[col] = df_subset[col].fillna(mode_val)
            print(f"Imputed {col} with mode: NaNs remaining: {df_subset[col].isnull().sum()}")
    
    
    if df_subset.isnull().any().any():
        final_imputer = SimpleImputer(strategy='constant', fill_value=0)  # Fill with 0 as fallback
        df_subset = pd.DataFrame(final_imputer.fit_transform(df_subset), columns=df_subset.columns)
        print("Final imputation applied for any remaining NaNs.")
    
    
    scaler_cols = available_numeric + [col for col in cat_cols if col in df_subset.columns]
    scaler = StandardScaler()
    scaler_path = "data/processed/labs_clinical/arthritis_profile_processed/scaler.pkl"
    print(f"Scaler created with sklearn {scaler.__module__.split('.')[0]}")
    if len(scaler_cols) > 0:
        df_subset[scaler_cols] = scaler.fit_transform(df_subset[scaler_cols])
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")
    
    
    le_target = LabelEncoder()
    df_subset['target_encoded'] = le_target.fit_transform(df_subset['target_raw'])

    
    synthetic_high = pd.DataFrame({
        'ESRh': np.random.uniform(35, 50, 30),  
        'ESRo': np.random.uniform(40, 55, 30),
        'CRP': np.random.uniform(25, 45, 30),
        'Uric_Acid': np.random.uniform(7.5, 9.5, 30),
        'ASO': np.random.uniform(250, 450, 30),
        'age_binned': np.random.choice([2, 3], 30, p=[0.6, 0.4]),
        'sex_encoded': np.random.choice([0, 1], 30, p=[0.4, 0.6]),
        'target_encoded': 1,  
        'target_raw': 1  
    })
  
    synthetic_high = synthetic_high.dropna()
    df_subset = pd.concat([df_subset, synthetic_high], ignore_index=True)
    print(f"Added {len(synthetic_high)} synthetic high-risk samples. New N: {len(df_subset)}")

    remaining_nans = df_subset.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"Warning: NaNs after imputation: {remaining_nans} (proceeding anyway)")
    else:
        print("No NaNs remaining after final checks.")
    
    print(f"Validation: CRP range {df_subset.get('CRP', pd.Series()).min():.1f}-{df_subset.get('CRP', pd.Series()).max():.1f} (expected 0-100)")
    print(f"Target balance: {df_subset['target_encoded'].value_counts().to_dict()} (positive rate: {df_subset['target_encoded'].mean():.2%})")
    
    
    processed_dir = "data/processed/labs_clinical/arthritis_profile_processed"
    os.makedirs(processed_dir, exist_ok=True)
    df_subset.to_parquet(f"{processed_dir}/multimodal.parquet")
    joblib.dump(scaler, f"{processed_dir}/scaler.pkl")
    joblib.dump(le_target, f"{processed_dir}/target_encoder.pkl")
    
    print(f"Processed {len(df_subset)} samples to {processed_dir}. Features: {available_features}")
    return df_subset, scaler, le_target

def preprocess_oai_labs(raw_csv="data/raw/labs_clinical/oai_labs/oai-clinical/clinical_info.csv"):
    df = pd.read_csv(raw_csv)
    print(f"Loaded OAI data: {df.shape} rows/cols. Columns: {df.columns.tolist()}")
    
   
    df['patient_uuid'] = df.get('ID', range(len(df))).apply(lambda x: str(uuid.uuid4()))
    
    
    feature_cols = ['CRP', 'ESR', 'pain_score']  
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using OAI features: {available_features}")

    
    df_subset = df[available_features + ['patient_uuid']].copy()
    print("OAI initial missing %:")
    print((df_subset.isnull().sum() / len(df_subset) * 100).round(1)[(df_subset.isnull().sum() / len(df_subset) * 100) > 0])
    
    
    if len(available_features) > 0:
        nan_count = df_subset[available_features].isnull().sum(axis=1)
        df_subset = df_subset[nan_count <= 1]  
        print(f"OAI: Dropped {len(df) - len(df_subset)} rows. Remaining: {len(df_subset)}")
        
       
        imputer = SimpleImputer(strategy='mean')
        df_subset[available_features] = imputer.fit_transform(df_subset[available_features])
    
    
    scaler = StandardScaler()
    if len(available_features) > 0:
        df_subset[available_features] = scaler.fit_transform(df_subset[available_features])
    
    processed_dir = "data/processed/labs_clinical/oai_labs_processed"
    os.makedirs(processed_dir, exist_ok=True)
    df_subset.to_parquet(f"{processed_dir}/trends.parquet")
    joblib.dump(scaler, f"{processed_dir}/scaler.pkl")
    
    print(f"OAI labs processed to {processed_dir} ({len(df_subset)} unique patients)")
    return df_subset, scaler

if __name__ == "__main__":
    try:
        preprocess_arthritis_profile()
        preprocess_oai_labs()
    except Exception as e:
        print(f"Error during processing: {e}")
