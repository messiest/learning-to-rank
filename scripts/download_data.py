import os
import argparse
import requests
import zipfile
import shutil
from tqdm import tqdm

# Constants for Dataset URLs
# These are the direct download links often associated with the datasets.
DATASET_URLS = {
    "MSLR-WEB10K": "https://onedrive.live.com/download?cid=1496350798606400&resid=1496350798606400%21235&authkey=AAjwN2h2qJ01rK0",
    "MSLR-WEB30K": "https://onedrive.live.com/download?cid=1496350798606400&resid=1496350798606400%21237&authkey=AAjwN2h2qJ01rK0"
}

def download_file(url, dest_path):
    """
    Downloads a file from a URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if response.status_code != 200:
        raise ConnectionError(f"Failed to connect. Status Code: {response.status_code}")

    print(f" [INFO] Downloading to {dest_path}...")
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory.
    """
    print(f" [INFO] Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(" [INFO] Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download MSLR Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["MSLR-WEB10K", "MSLR-WEB30K"], 
                        help="Which dataset to download.")
    parser.add_argument("--output_dir", type=str, default="data/raw", 
                        help="Directory to save the raw data.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Check if data already exists to avoid re-downloading
    expected_folder = os.path.join(args.output_dir, args.dataset)
    if os.path.exists(expected_folder):
        print(f" [INFO] Dataset folder {expected_folder} already exists. Skipping download.")
        return

    # 2. Download
    zip_filename = f"{args.dataset}.zip"
    zip_path = os.path.join(args.output_dir, zip_filename)
    url = DATASET_URLS[args.dataset]

    try:
        download_file(url, zip_path)
    except Exception as e:
        print(f"\n [ERROR] Download failed: {e}")
        print(" [HINT] Microsoft download links can be tricky. You can download manually here:")
        print(f"        https://www.microsoft.com/en-us/research/project/mslr/")
        print(f"        Place the downloaded zip file at: {zip_path}")
        return

    # 3. Extract
    try:
        extract_zip(zip_path, args.output_dir)
    except zipfile.BadZipFile:
        print(" [ERROR] The downloaded file is not a valid zip. Try downloading manually.")
        return

    # 4. Cleanup (Optional: remove zip to save space)
    # os.remove(zip_path)
    # print(f" [INFO] Removed {zip_path}")

    print(f"\n [SUCCESS] Dataset ready at {expected_folder}")

if __name__ == "__main__":
    main()
