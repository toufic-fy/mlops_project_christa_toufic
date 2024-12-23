from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from api.dependencies import get_inference_pipeline

# API Router
router = APIRouter()

# Health check endpoint
@router.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}

# Request schema for inference
class InferenceRequest(BaseModel):
    email_body: str

# Response schema for inference
class InferenceResponse(BaseModel):
    prediction: int
    confidence: float

# Inference endpoint
@router.post("/inference", tags=["Inference"], response_model=InferenceResponse)
def inference(
    request: InferenceRequest,
    pipeline=Depends(get_inference_pipeline)
):
    """
    Perform inference on the given email body.
    """
    try:
        result = pipeline.run(data=[request.email_body], include_confidence=True)

        prediction = result["predictions"][0]
        confidence = result["confidences"][0]

        return InferenceResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
