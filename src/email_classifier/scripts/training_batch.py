from ..config import Config
from ..app_core import load_pipeline

def main(config: Config):
    print("✅ Starting training pipeline")
    pipeline = load_pipeline(
        pipeline_type="training",
        vectorization_config=config.vectorization,
        model_config=config.classification
    )

    data = pipeline.load_data(file_type=config.data.file_type, file_path=config.data.file_path)
    print("✅ Data loading and preprocessing done, running inference...")
    print(data["body"])
    pipeline.run(data=data["body"], labels=data["label"])
    print("✅ Training and evaluation successfully done")

if __name__ == "__main__":
    main()
