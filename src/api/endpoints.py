from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from api.dependencies import get_pipeline, load_data
from api.schemas import InferenceRequest, InferenceResponse, TrainingResponse


# API Router
router = APIRouter()

# Health check endpoint
@router.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}


# Inference endpoint
@router.post("/inference", tags=["Inference"], response_model=InferenceResponse)
def inference(
    request: InferenceRequest,
    pipeline=Depends(lambda: get_pipeline("inference"))
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


@router.get("/train", tags=["Train"], response_model=TrainingResponse)
def train(
    background_tasks: BackgroundTasks,
    config_path: str = Query(..., description="path to the config file...")
    ):
    """
    Perform training in the background and return success if the job started.
    """
    try:
        data = load_data(config_path=config_path)
        pipeline= get_pipeline("training", config_path=config_path)
        background_tasks.add_task(pipeline.run, data=data["body"], labels=data["label"])
        return TrainingResponse(status="success", message="Training pipeline successfully started")
    except Exception as e:
        print(f"error in train endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
