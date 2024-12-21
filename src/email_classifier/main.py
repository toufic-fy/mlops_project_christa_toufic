import argparse
from loguru import logger
from email_classifier.config import load_config
from scripts.training_batch import main as training_main
from scripts.inference_batch import main as inference_main

parser = argparse.ArgumentParser(description="Run the Email Classifier pipeline.")
logger.add("logs/main_pipeline.log", rotation="500 MB")

def parse_command_args():
    parser.add_argument(
        "--config-path",
        type=str,
        required=False,
        default="config/config.yaml",
        help="Path to the configuration file (YAML format). Default is config/config.yaml",
    )
    parser.add_argument(
        "--script",
        type=str,
        choices=["training", "inference"],
        required=True,
        help="Which script to run: 'training' or 'inference'.",
    )
    return parser.parse_args()

def main():
    try:
        args = parse_command_args()
        # Load configuration
        config = load_config(args.config_path)
        if not config:
            print("Failed to load configuration.")
            return

        # Determine which script to run
        if args.script == "training":
            print("Running training script...")
            training_main(config)
        elif args.script == "inference":
            print("Running inference script...")
            inference_main(config)
        else:
            print(f"Unknown script: {args.script}")

    except FileNotFoundError as f:
        print(f"❌ Error: Configuration file not found at {args.config_path}")
        logger.error(f"main.py: Error loading file {f}")
    except Exception as e:
        print(f"❌ Unexpected error running main: {e}")
        logger.exception(f"main.py: {e.with_traceback} ")

if __name__ == "__main__":
    main()
