import argparse
from email_classifier.config import load_config

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
        config = load_config(args.config_path)
        print("halo from main")
        print(f"loaded config project name: {config.project.name}")
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {args.config_path}")
    except Exception as e:
        print(f"❌ Unexpected error running main: {e}")

if __name__ == "__main__":
    main()