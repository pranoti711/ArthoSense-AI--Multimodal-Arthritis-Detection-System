import os
import shutil
import sys
import pandas as pd
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import uvicorn

# ==============================
# FIX PATH
# ==============================
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# from modules.decision_support import generate_care_plan

# ==============================
# INIT APP
# ==============================
app = FastAPI(
    title="Arthritis Care Assistant API",
    description="Flexible multimodal arthritis risk assessment with JSON-only responses.",
    version="2.2",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==============================
# TEMP UPLOAD DIR
# ==============================
UPLOAD_DIR = os.path.join(root_dir, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================
# SAVE FILE
# ==============================
def save_file(upload_file: UploadFile, filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

# ==============================
# VALIDATE CSV
# ==============================
def validate_csv_file(upload_file: UploadFile, file_type: str):
    if not upload_file:
        return None

    try:
        contents = upload_file.file.read()
        df = pd.read_csv(io.BytesIO(contents))

        print(f"✓ {file_type.upper()} CSV received: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        df.columns = df.columns.str.lower().str.strip()

        file_path = os.path.join(UPLOAD_DIR, f"{file_type}.csv")
        df.to_csv(file_path, index=False)

        return file_path

    except Exception as e:
        print(f"⚠ CSV issue ({file_type}): {e}")
        return save_file(upload_file, f"{file_type}.csv")

# ==============================
# VALIDATE IMAGE
# ==============================
def validate_image_file(upload_file: UploadFile):
    if not upload_file:
        return None

    try:
        filename = upload_file.filename or "image.png"
        print(f"✓ IMAGE received: {filename}")
        return save_file(upload_file, filename)
    except Exception as e:
        print(f"⚠ Image issue: {e}")
        return save_file(upload_file, "image.png")

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Arthritis Multimodal API is live",
        "docs": "http://127.0.0.1:8000/docs"
    }

# ==============================
# TEST UPLOAD ENDPOINT
# ==============================
@app.post("/test-upload")
async def test_upload(
    patient_zip: str = Form("90210"),
    genomics_file: Optional[UploadFile] = File(None),
    voice_file: Optional[UploadFile] = File(None),
    imaging_file: Optional[UploadFile] = File(None),
    temporal_file: Optional[UploadFile] = File(None),
    lab_file: Optional[UploadFile] = File(None)
):
    files_received = {}

    if genomics_file:
        validate_csv_file(genomics_file, "genomics")
        files_received["genomics"] = genomics_file.filename

    if voice_file:
        save_file(voice_file, "voice.wav")
        files_received["voice"] = voice_file.filename

    if imaging_file:
        validate_image_file(imaging_file)
        files_received["imaging"] = imaging_file.filename

    if temporal_file:
        validate_csv_file(temporal_file, "temporal")
        files_received["temporal"] = temporal_file.filename

    if lab_file:
        validate_csv_file(lab_file, "lab")
        files_received["lab"] = lab_file.filename

    return {
        "status": "success",
        "files_received": files_received,
        "files_count": len(files_received)
    }

# ==============================
# MAIN PREDICT ENDPOINT
# ==============================
@app.post("/predict")
async def predict(
    patient_zip: str = Form("90210"),
    genomics_file: Optional[UploadFile] = File(None),
    voice_file: Optional[UploadFile] = File(None),
    imaging_file: Optional[UploadFile] = File(None),
    temporal_file: Optional[UploadFile] = File(None),
    lab_file: Optional[UploadFile] = File(None)
):
    try:
        print("\n📥 NEW REQUEST")

        # Save files
        genomics_path = validate_csv_file(genomics_file, "genomics") if genomics_file else None
        voice_path = save_file(voice_file, "voice.wav") if voice_file else None
        imaging_path = validate_image_file(imaging_file) if imaging_file else None
        temporal_path = validate_csv_file(temporal_file, "temporal") if temporal_file else None
        lab_path = validate_csv_file(lab_file, "lab") if lab_file else None

        print("Files ready")

        # Import model
        from modules.multimodal_fusion.predict_fusion import predict_final_enhanced

        prediction_data = predict_final_enhanced(
            genomics_path,
            voice_path,
            imaging_path,
            temporal_path,
            lab_path
        )

        # Generate care plan
        care_plan = generate_care_plan(prediction_data["risk"], patient_zip)

        return {
            "status": "success",
            "patient_zip": patient_zip,
            "modalities_used": prediction_data["modalities_used"],

            "prediction": {
                "risk_level": prediction_data["risk"],
                "score": prediction_data.get("score"),
                "confidence": prediction_data["confidence"],
                "summary": prediction_data.get("summary")
            },

            "explainability": {
                "summary": prediction_data.get("summary"),
                "modality_contributions": prediction_data.get("explanation", {}),
                "modalities_used": prediction_data["modalities_used"]
            },

            "recommendations": {
                "suggestions": care_plan["guidelines"].get("suggestions"),
                "prevention": care_plan["guidelines"].get("prevention"),
                "nearest_doctors": care_plan["nearest_doctors"],
                "world_experts": care_plan["world_experts"],
                "next_steps": care_plan["next_steps"]
            },

            "tracking": care_plan.get("tracking")
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    uvicorn.run(
        "services.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )