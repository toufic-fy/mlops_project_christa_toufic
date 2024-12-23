from email_classifier.pipelines.factory import PipelineFactory
from email_classifier.config import load_config
from email_classifier.data_handling.data_loader.factory import DataLoaderFactory
from email_classifier.training.classifier_model.factory import ClassifierFactory
from email_classifier.data_handling.vectorizer.factory import VectorizerFactory

# Dependency to get the inference pipeline
def get_pipeline(type: str, config_path: str = "config/config.yaml"):
    """
    Dependency to load and return the inference pipeline.
    """
    # Load configuration
    config = load_config(config_path)

    # Load model and vectorizer (mocked for now)
    # TODO: load model from MLFlow
    model = ClassifierFactory.get_classifier(config.classification.type)
    vectorizer = VectorizerFactory.get_vectorizer(config.vectorization.type)

    # Return an instance of the inference pipeline
    return PipelineFactory.get_pipeline(type, model, vectorizer)

def load_data(config_path: str = "config/config.yaml"):
    """
    Dependency to load the data
    """
    config = load_config(config_path)
    data_loader = DataLoaderFactory.get_data_loader(file_type=config.data.file_type)
    return data_loader.load_and_preprocess_data(config.data.file_path)