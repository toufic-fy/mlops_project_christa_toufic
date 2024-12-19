from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import Enum
from typing import List, Dict, Union
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

class PathsConfig(BaseModel):
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

    @field_validator("params", always=True)
    def validate_vectorizer_params(cls, params, values):
        if values["type"] == "tfidf" and "max_features" not in params:
            raise ValueError("For tfidf vectorizer, 'max_features' must be specified in params.")
        return params
class ClassificationConfig(BaseModel):
    type: ClassifierType = Field(..., description="Type of classifier (e.g., sgd, logistic)")
    params: Dict[str, List[Union[int, float, str]]] = Field(
        default_factory=dict,
        description="Hyperparameters for the classifier (e.g., alpha, max_iter, tol)"
    )

    @field_validator("params", always=True)
    def validate_classifier_params(cls, params, values):
        if values["type"] == "sgd" and "alpha" not in params:
            raise ValueError("For sgd classifier, 'alpha' must be specified in params.")
        return params

class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig

# Function to load the config.yaml file
def load_config(config_path: str = "config.yaml") -> Config:
    try:
        # Load the YAML file with OmegaConf
        raw_config = OmegaConf.load(config_path)
        # Convert the OmegaConf object to a dictionary and validate with Pydantic
        return Config(**OmegaConf.to_container(raw_config, resolve=True))
    except ValidationError as ve:
        print(f"error loading config: {ve}")
