from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from api.dependencies import get_config, get_pipeline, load_data, get_experiment_name
from api.schemas import InferenceRequest, InferenceResponse, TrainingResponse
from email_classifier.config import PipelineType


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
    config=Depends(get_config),
    pipeline=Depends(lambda config=Depends(get_config): get_pipeline(PipelineType.inference, config)),
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
        config = get_config(config_path)
        data = load_data(config)
        experiment_name = get_experiment_name(config)
        pipeline = get_pipeline(PipelineType.training, config)
        background_tasks.add_task(pipeline.run, data=data["body"], labels=data["label"], experiment_name=experiment_name)
        return TrainingResponse(status="success", message="Training pipeline successfully started")
    except Exception as e:
        print(f"error in train endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
