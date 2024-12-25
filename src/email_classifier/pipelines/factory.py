from ..config import Config
from .training_pipeline import TrainingPipeline
from .inference_pipeline import InferencePipeline
from ..data_handling.vectorizer.factory import VectorizerFactory
from ..training.classifier_model.factory import ClassifierFactory
from ...utils.mlf_utils import load_model_pipeline
class PipelineFactory:
    @staticmethod
    def get_pipeline(pipeline_type: str, config: Config, use_mlflow: bool=False):
        if use_mlflow:
            # Load the combined model pipeline from MLflow
            pipeline = load_model_pipeline(
                model_name=config.mlflow.model.name,
                stage=config.mlflow.model.stage
            )
            if pipeline_type == "inference":
                return InferencePipeline.from_pipeline(pipeline)
            elif pipeline_type == "training":
                raise ValueError("Training pipeline cannot use a preloaded model.")
        else:
            # Create model and vectorizer separately
            model = ClassifierFactory.get_classifier(config.classification.type)
            vectorizer = VectorizerFactory.get_vectorizer(config.vectorization.type, **config.vectorization.params)

            if pipeline_type == "training":
                return TrainingPipeline(model, vectorizer)
            elif pipeline_type == "inference":
                return InferencePipeline(model, vectorizer)
            else:
                raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
