from email_classifier.config import Config
from email_classifier.app_core import load_pipeline
from email_classifier.config import load_config
from typing import Optional


# TODO: next major - remove this default config fallback
# and create a base script with CLI wrapper
def main(config: Optional[Config] = None):

    if config is None:
        config = load_config("config/config.yaml")

    print("✅ Starting training pipeline")
    pipeline = load_pipeline(
        pipeline_type="training",
        config=config
    )

    data = pipeline.load_data(file_type=config.data.file_type, file_path=config.data.file_path)
    print("✅ Data loading and preprocessing done, running inference...")

    pipeline.run(data=data["body"], labels=data["label"], experiment_name = config.mlflow.experiment_name)
    print("✅ Training and evaluation successfully done")

if __name__ == "__main__":
    main()
