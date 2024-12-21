from .training_pipeline import TrainingPipeline
from .inference_pipeline import InferencePipeline
from ..training.classifier_model.base_classifier import BaseClassifier
from ..data_handling.vectorizer.base_text_vectorizer import BaseTextVectorizer

class PipelineFactory:
    @staticmethod
    def get_pipeline(pipeline_type, model: BaseClassifier, vectorizer: BaseTextVectorizer):
        if pipeline_type == "training":
            return TrainingPipeline(model, vectorizer)
        elif pipeline_type == "inference":
            return InferencePipeline(model, vectorizer)
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")
