from email_classifier.config import Config
from email_classifier.app_core import load_pipeline

def main(config: Config):
    print("✅ Starting inference pipeline")
    pipeline = load_pipeline(
        pipeline_type="inference",
        vectorization_config=config.vectorization,
        model_config=config.classification
    )

    data = pipeline.load_data(file_type=config.data.file_type, file_path=config.data.file_path)
    print("✅ Data loading and preprocessing done, running inference...")

    pipeline.run(data=data)

if __name__ == "__main__":
    main()
