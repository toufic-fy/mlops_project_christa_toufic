from pydantic import BaseModel

# Request schema for inference
class InferenceRequest(BaseModel):
    email_body: str

# Response schema for inference
class InferenceResponse(BaseModel):
    prediction: int
    confidence: float

class TrainingRequest(BaseModel):
    config_path: str

class TrainingResponse(BaseModel):
    status: str
    message: str
