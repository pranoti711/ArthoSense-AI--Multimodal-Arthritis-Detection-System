import pandas as pd
import os
import glob

raw_path = r'C:\Users\Pranoti munjankar\Desktop\arthritis-detection-care-assistant-local\data\raw\genomics_omics'

if not os.path.exists(raw_path):
    dir_path = os.path.dirname(raw_path)
    if os.path.isdir(dir_path):
        csvs = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {dir_path}. Available entries: {os.listdir(dir_path)}"
            )
        
        raw_path = csvs[0]
        print(f"Using detected CSV: {raw_path}")
    else:
        raise FileNotFoundError(f"Expected file or directory not found: {raw_path}")

try:
    df = pd.read_csv(
        raw_path,
        sep=None,
        engine="python",
        comment="#",
        on_bad_lines="skip"
    )
except Exception as e:
    raise RuntimeError(f"Failed to read CSV at {raw_path}: {e}")

# SNPs of interest (rsIDs)
target_snps = ["rs2476601", "rs1801133", "HLA-B27", "PTPN22"]  # PTPN22-related, inflammation-related

# determine likely column that holds SNP identifiers
name_col = None
lower_cols = {c.lower(): c for c in df.columns}
for cand in ("name", "rsid", "snp", "id", "marker"):  # common options
    if cand in lower_cols:
        name_col = lower_cols[cand]
        break
if name_col is None:
    raise KeyError(f"Could not find SNP identifier column. CSV columns: {list(df.columns)}")

# Create simple binary features
values = df[name_col].astype(str).values
features = {snp: (1 if snp in values else 0) for snp in target_snps}

X = pd.DataFrame([features])


processed_dir = "data/processed/genomics_omics/ra_genotype_processed"
os.makedirs(processed_dir, exist_ok=True)

X.to_csv(os.path.join(processed_dir, "X.csv"), index=False)

print("✅ Preprocessing complete")
print(X)




