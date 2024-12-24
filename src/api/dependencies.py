from email_classifier.pipelines.factory import PipelineFactory
from email_classifier.config import load_config, Config
from functools import lru_cache
from email_classifier.data_handling.data_loader.factory import DataLoaderFactory

@lru_cache()
def get_config(config_path: str = "config/config.yaml") -> Config:
    """
    Cached dependency to load the configuration.
    """
    return load_config(config_path)

# Dependency to get the inference pipeline
def get_pipeline(type: str, config: Config):
    """
    Dependency to load and return the inference pipeline.
    """
    # Return an instance of the inference pipeline
    return PipelineFactory.get_pipeline(type, config)

def load_data(config: Config):
    """
    Dependency to load the data
    """
    data_loader = DataLoaderFactory.get_data_loader(file_type=config.data.file_type)
    return data_loader.load_and_preprocess_data(config.data.file_path)

def get_experiment_name(config: Config):
    """
    Dependency to get mlflow experiment's name
    """
    return config.mlflow.experiment_name
