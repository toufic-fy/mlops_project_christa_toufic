from functools import lru_cache
from sklearn.pipeline import Pipeline
import mlflow.sklearn
import mlflow.pyfunc

@lru_cache()
def load_model_pipeline(model_name: str, stage: str):
    """
    Load the combined model and vectorizer pipeline from MLflow.
    """
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)

def save_model_with_vectorizer(model, vectorizer, artifact_path="model_pipeline"):
    """
    Save a combined model and vectorizer as a single artifact.
    """
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])
    mlflow.sklearn.log_model(pipeline, artifact_path)