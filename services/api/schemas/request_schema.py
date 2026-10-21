from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import UploadFile, File

class PredictRequest(BaseModel):
    patient_zip: str = Field("90210", description="Patient ZIP code for location-based recommendations")
    
    genomics_file: Optional[UploadFile] = Field(None, description="Genomics CSV (any columns)")
    voice_file: Optional[UploadFile] = Field(None, description="Audio file (.wav, .mp3)")
    imaging_file: Optional[UploadFile] = Field(None, description="Medical image (.png, .jpg, .dicom)")
    temporal_file: Optional[UploadFile] = Field(None, description="Time-series CSV (any columns)")
    lab_file: Optional[UploadFile] = Field(None, description="Lab results CSV (any columns)")

