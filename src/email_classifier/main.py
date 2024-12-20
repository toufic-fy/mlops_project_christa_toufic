# src/
import argparse
from loguru import logger
from dotenv import load_dotenv
from email_classifier.config import load_config
from sklearn.model_selection import train_test_split
from .data_pipeline.preprocessor.email_preprocessor import EmailPreprocessor
from .data_pipeline.data_loader.factory import DataLoaderFactory
from .data_pipeline.vectorizer.factory import VectorizerFactory
from .training.classifier_model.factory import ClassifierFactory
from .training.trainer import Trainer

logger.add("logs/main_pipeline.log", rotation="500 MB")

load_dotenv()
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the email classifier application.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        #load config
        config = load_config(args.config_path)

        # Step 1: Load and preprocess data
        data_loader = DataLoaderFactory.get_data_loader(config.data.file_type)
        print("✅ Data loading done")
        data = data_loader.load_and_preprocess_data(file_path=config.data.file_path, preprocessors=[EmailPreprocessor()])
        print("✅ Data preprocessing done")

        vectorizer = VectorizerFactory.get_vectorizer(config.vectorization.type, **config.vectorization.params)
        # vectorized_emails: list[str] = vectorizer.vectorize(data["body"].to_list())

        # Step 2: Training preps
        # TODO: next major - make test_size, stratify and radom_seed configurable
        X_train, X_test, y_train, y_test = train_test_split(data["body"], data["label"], test_size=0.2, random_state=42)
        print("✅ Training preparations done, starting training...")

        # Step 3: Train and evaluate the model
        classifier = ClassifierFactory.get_classifier(config.classification.type)
        trainer = Trainer(classifier, vectorizer)

        trained_model = trainer.train(X_train.tolist(), y_train.tolist())

        _ = trainer.evaluate(trained_model, X_test.tolist(), y_test.tolist())
        logger.info("Pipeline execution completed successfully.")
        # pred = trained_model.predict(X_test)
        # cf_matrix = confusion_matrix(y_test, pred)
        # sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    except FileNotFoundError as f:
        print(f"❌ Error: Configuration file not found at {args.config_path}")
        logger.error(f"main.py: Error loading file {f}")
    except Exception as e:
        print(f"❌ Unexpected error running main: {e}")
        logger.exception(f"main.py: {e.with_traceback} ")

if __name__ == "__main__":
    main()
