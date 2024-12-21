from pydantic import BaseModel

# Request schema for inference
class InferenceRequest(BaseModel):
    email_body: str

# Response schema for inference
class InferenceResponse(BaseModel):
    prediction: int
    confidence: float
