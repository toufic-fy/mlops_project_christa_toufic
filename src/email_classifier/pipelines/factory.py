from ..config import Config, PipelineType
from .training_pipeline import TrainingPipeline
from .inference_pipeline import InferencePipeline
from ..data_handling.vectorizer.factory import VectorizerFactory
from ..training.classifier_model.factory import ClassifierFactory
from utils.mlf_utils import load_model_pipeline, configure_mlflow
class PipelineFactory:
    @staticmethod
    def get_pipeline(pipeline_type: PipelineType, config: Config):

        configure_mlflow(
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name=config.mlflow.experiment_name
        )

        if pipeline_type == PipelineType.inference:
            # Load the combined model pipeline from MLflow
            pipeline = load_model_pipeline(
                model_name=config.mlflow.model.name,
                stage=config.mlflow.model.stage
            )

            return InferencePipeline.from_pipeline(pipeline)
        else:
            # Create model and vectorizer separately
            model = ClassifierFactory.get_classifier(config.classification.type)
            vectorizer = VectorizerFactory.get_vectorizer(config.vectorization.type, **config.vectorization.params)
            return TrainingPipeline(model, vectorizer)
