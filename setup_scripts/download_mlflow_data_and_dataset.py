import os
import zipfile
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor


MLRUNS_ID = "1SwgZPy9V6Q-_zTGcEN7bGWUGQWS3OFjT"  
DB_ID = "1YjynBatGkMPLhOYPiSmZ4HrNr7T0S4N8"
DATASET_ID = "1V-TFy-ZS5EufUPkxEk2wG3sh3HYEEO-b" 

BASE_URL = "https://drive.google.com/uc?id={}&export=download"
MLRUNS_URL = BASE_URL.format(MLRUNS_ID)
DB_URL = BASE_URL.format(DB_ID)
DATASET_URL = BASE_URL.format(DATASET_ID)

# Local paths to save the downloaded files
DOWNLOAD_FOLDER = "./data"
MLRUNS_FILE = os.path.join(DOWNLOAD_FOLDER, "mlruns.zip")
DB_FILE = os.path.join(DOWNLOAD_FOLDER, "mlflow.db")
DATASET_FILE = os.path.join(DOWNLOAD_FOLDER, "preprocessed_emails_dataset.csv")

def download_file(url, save_path):
    print(f"Starting download from {url}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with requests.Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    # Printing the progress...
                    percent_complete = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    print(f"\rDownloading {os.path.basename(save_path)}: {downloaded_size} / {total_size} bytes "
                        f"({percent_complete:.2f}%)", end="")
        print(f"\nDownloaded: {save_path}")

def extract_mlruns():
    print("Extracting mlruns folder...")
    with zipfile.ZipFile(MLRUNS_FILE, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_FOLDER)
    print("Extraction complete.")
    if os.path.exists(MLRUNS_FILE):
        try:
            os.remove(MLRUNS_FILE)
            print(f"Removed: {MLRUNS_FILE}")
        except Exception as e:
            print(f"Failed to remove {MLRUNS_FILE}: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download MLflow data and/or dataset files.")
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Download only the dataset file."
    )
    parser.add_argument(
        "--mlflow-data-only",
        action="store_true",
        help="Download only the MLflow data files (mlruns and database)."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    download_dataset = not args.mlflow_data_only
    download_mlflow_data = not args.dataset_only

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        if download_mlflow_data:
            futures.append(executor.submit(download_file, MLRUNS_URL, MLRUNS_FILE))
            futures.append(executor.submit(download_file, DB_URL, DB_FILE))
        if download_dataset:
            futures.append(executor.submit(download_file, DATASET_URL, DATASET_FILE))

        for future in futures:
            future.result()
        
    if download_mlflow_data and os.path.exists(MLRUNS_FILE):
        extract_mlruns()

    print("Downloads and extraction complete.")
    print("To start MLflow with the downloaded data, run the following command:")
    print(f"mlflow server --backend-store-uri sqlite:///{DB_FILE} "
          f"--default-artifact-root {os.path.join(DOWNLOAD_FOLDER, 'mlruns')} "
          f"--host 0.0.0.0 --port 5000")

if __name__ == "__main__":
    main()
