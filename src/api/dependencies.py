from email_classifier.pipelines.factory import PipelineFactory
from email_classifier.config import load_config
from email_classifier.training.classifier_model.factory import ClassifierFactory
from email_classifier.data_handling.vectorizer.factory import VectorizerFactory

# Dependency to get the inference pipeline
def get_inference_pipeline():
    """
    Dependency to load and return the inference pipeline.
    """
    # Load configuration
    config = load_config("config/config.yaml")

    # Load model and vectorizer (mocked for now)
    # TODO: load model from MLFlow
    model = ClassifierFactory.get_classifier(config.classification.type)
    vectorizer = VectorizerFactory.get_vectorizer(config.vectorization.type)

    # Return an instance of the inference pipeline
    return PipelineFactory.get_pipeline("inference", model, vectorizer)
