from omegaconf import OmegaConf
from pydantic import BaseModel

# Define basic Pydantic models for validation
class ProjectConfig(BaseModel):
    name: str
    version: str

class PathsConfig(BaseModel):
    raw_data: str
    file_type: str

class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig

# Function to load the config.yaml file
def load_config(config_path: str = "config.yaml") -> Config:
    # Load the YAML file with OmegaConf
    raw_config = OmegaConf.load(config_path)
    # Convert the OmegaConf object to a dictionary and validate with Pydantic
    return Config(**OmegaConf.to_container(raw_config, resolve=True))
