from functools import lru_cache
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

@lru_cache()
def load_model_pipeline(model_name: str, stage: str):
    """
    Load the combined model and vectorizer pipeline from MLflow.
    """
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)

def configure_mlflow(tracking_uri: str, experiment_name: str):
    """
    Configure MLflow tracking URI and experiment name.
    If experiment doesn't exist, it creates it.

    Args:
        tracking_uri (str): The URI for the MLflow tracking server.
        experiment_name (str): The name of the MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found. Creating it now...")
        mlflow.create_experiment(experiment_name)
    else:
        print(f"Using experiment: {experiment.name} (ID: {experiment.experiment_id})")
    mlflow.set_experiment(experiment_name)


def save_model_with_vectorizer(model, vectorizer, artifact_path="model_pipeline"):
    """
    Save a combined model and vectorizer as a single artifact.
    """
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])
    mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)
