# src/
import argparse
from dotenv import load_dotenv
from email_classifier.config import load_config
from .data_pipeline.preprocessor.email_preprocessor import EmailPreprocessor
from .data_pipeline.data_loader.factory import DataLoaderFactory

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
        data = data_loader.load_and_preprocess_data(file_path=config.data.file_path, preprocessors=[EmailPreprocessor])
        print(f"Preprocessed Data Shape: {data.shape}")

    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {args.config_path}")
    except Exception as e:
        print(f"❌ Unexpected error running main: {e}")

if __name__ == "__main__":
    main()