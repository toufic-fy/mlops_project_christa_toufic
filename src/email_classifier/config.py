from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import Enum
from typing import List, Dict, Union, Any, cast
class FileType(str, Enum):
    csv = "csv"

class VectorizerType(str, Enum):
    tfidf = "tfidf"
    bow = "bow"

class ClassifierType(str, Enum):
    sgd = "sgd"
    logistic = "logistic"
class ProjectConfig(BaseModel):
    name: str
    version: str

    @field_validator("name")
    def name_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Project name cannot be empty.")
        return value

    @field_validator("version")
    def version_must_follow_format(cls, value):
        import re
        if not re.match(r"^\d+\.\d+\.\d+$", value):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0).")
        return value

class DataConfig(BaseModel):
    file_path: str = Field(..., description="Path to the dataset file")
    file_type: FileType = Field(..., description="Type of the dataset file")

    @field_validator("file_path")
    def raw_data_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Path to raw_data cannot be empty.")
        return value
class VectorizationConfig(BaseModel):
    type: VectorizerType = Field(..., description="Type of vectorizer (e.g., tfidf, bow)")
    params: Dict[str, Union[int, str]] = Field(
        default_factory=dict,
        description="Parameters for the vectorizer (e.g., max_features, stop_words)"
    )

    @field_validator("params", mode="before")
    def validate_vectorizer_params(cls, params, info):
        if info.data["type"] == "tfidf" and "max_features" not in params:
            raise ValueError("For tfidf vectorizer, 'max_features' must be specified in params.")
        return params
class ClassificationConfig(BaseModel):
    type: ClassifierType = Field(..., description="Type of classifier (e.g., sgd, logistic)")
    params: Dict[str, List[Union[int, float, str, str]]] = Field(
        default_factory=dict,
        description="Hyperparameters for the classifier (e.g., alpha, max_iter, tol)"
    )

    @field_validator("params")
    def validate_classifier_params(cls, params, info):
        if info.data["type"] == "sgd" and "alpha" not in params:
            raise ValueError("For sgd classifier, 'alpha' must be specified in params.")
        return params

class Config(BaseModel):
    project: ProjectConfig
    data: DataConfig
    vectorization: VectorizationConfig
    classification: ClassificationConfig

def load_config(config_path: str = "config.yaml") -> Config | None:
    """
    Loads the config from a given path. By default it loads ./config.yaml
    """
    try:
        # Load the YAML file with OmegaConf
        raw_config = OmegaConf.load(config_path)
        # Convert the OmegaConf object to a dictionary and validate with Pydantic
        config_container = cast(Dict[str, Any], OmegaConf.to_container(raw_config, resolve=True))


        if not isinstance(config_container, dict):
            raise TypeError(f"Expected a dictionary, but got {type(config_container)}")

        return Config(**config_container)
    except ValidationError as ve:
        print(f"error loading config: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
