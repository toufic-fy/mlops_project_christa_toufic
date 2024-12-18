from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import Enum
class FileType(str, Enum):
    csv = "csv"
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
