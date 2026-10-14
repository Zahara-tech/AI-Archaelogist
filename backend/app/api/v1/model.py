"""
Model Information and Verification Endpoints
"""

from fastapi import APIRouter, HTTPException
from app.services.model_service import model_service

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info")
def get_model_info():
    """Get model information."""
    try:
        info = model_service.get_model_info()
        if info.get("status") == "error":
            raise HTTPException(status_code=500, detail=info.get("error"))
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify")
def verify_model():
    """Verify model with synthetic input."""
    try:
        result = model_service.verify_model()
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def model_health():
    """Check if model is loaded."""
    return {
        "loaded": model_service.model_loaded,
        "device": str(model_service.device)
    }
