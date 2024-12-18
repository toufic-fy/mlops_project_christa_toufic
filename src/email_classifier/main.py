from email_classifier.config import load_config

def main():
    config = load_config("config/config.yaml")
    print("halo from main")
    print(f"loaded config project name: {config.project.name}")

if __name__ == "__main__":
    main()