"""Download the official BANKING77 train/test CSV files."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = Path(__file__).resolve().parent
URLS = {
    "banking_train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "banking_test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}

for filename, url in URLS.items():
    destination = BASE / filename
    if destination.exists():
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename}...")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")
