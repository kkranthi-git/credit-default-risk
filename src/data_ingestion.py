import os
import zipfile
import requests
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
DOWNLOADED_FILE = "default of credit card clients.xls"
DATA_FILE = RAW_DATA_PATH / "credit_default.xls"

def run_data_ingestion():
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_PATH / "credit_default.zip"

    # Step 1: Download raw zip archive
    print(f"Downloading dataset from {URL}...")
    try:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download raw data: {e}")

    # Step 2: Unpack contents
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_PATH)
        print(f"Extracted contents to '{RAW_DATA_PATH}'.")
    except zipfile.BadZipFile:
        raise ValueError("Downloaded file is corrupted or not a valid zip file.")
    finally:
        # Step 3: Cleanup archive file
        if zip_path.exists():
            zip_path.unlink()

    downloaded_path = RAW_DATA_PATH / DOWNLOADED_FILE
    if downloaded_path.exists():
        downloaded_path.replace(DATA_FILE)

    # Step 4: Verification
    if DATA_FILE.exists():
        print(f"Success: Ingested file available at '{DATA_FILE}'")
    else:
        raise FileNotFoundError("Ingestion completed, but expected Excel file was not found.")

if __name__ == "__main__":
    run_data_ingestion()
