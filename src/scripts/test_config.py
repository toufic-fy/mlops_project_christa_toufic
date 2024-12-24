from email_classifier.config import load_config

# Load the configuration file
config = load_config("config/config.yaml")

# Print MLflow configuration to verify
print("MLflow Configuration:")
print(f"Tracking URI: {config.mlflow.tracking_uri}")
print(f"Experiment Name: {config.mlflow.experiment_name}")
