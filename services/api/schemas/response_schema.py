from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Health check message")
    documentation: str = Field(..., description="Swagger UI URL")
    test_endpoint: str = Field(..., description="Test endpoint URL")

class TestResponse(BaseModel):
    status: str
    message: str
    test_data: Dict[str, Any]

class ModalityScore(BaseModel):
    genomics: float = Field(..., description="Genomics risk score (0-1)")
    voice: float = Field(..., description="Voice biomarker score (0-1)")
    imaging: float = Field(..., description="Imaging grade (0-1)")
    temporal: float = Field(..., description="Temporal progression score (0-1)")
    lab: float = Field(..., description="Lab biomarker score (0-1)")

class PredictionResponse(BaseModel):
    status: str = "success"
    patient_zip: str
    modalities_received: List[str]
    prediction: Dict[str, Any] = Field(..., description="Risk assessment")
    modality_scores: ModalityScore
    explainability: Dict[str, Any]
    recommendations: Dict[str, Any]
    tracking: Optional[Dict[str, Any]] = None

