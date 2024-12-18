from email_classifier.config import load_config

def main():
    try:
        config = load_config("config/config.yaml")
        print("halo from main")
        print(f"loaded config project name: {config.project.name}")
    except Exception:
        print("error running main")

if __name__ == "__main__":
    main()