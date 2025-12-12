import os
import requests
import zipfile

# ---------------------------------------------------------
# Create destination folder
# ---------------------------------------------------------
data_dir = "Data"
os.makedirs(data_dir, exist_ok=True)

# ---------------------------------------------------------
# URL of dataset
# ---------------------------------------------------------
url = "https://www.kaggle.com/api/v1/datasets/download/naureenmohammad/mmu-iris-dataset"
zip_path = os.path.join(data_dir, "MMU_orginal.zip")

# ---------------------------------------------------------
# Download the ZIP file
# ---------------------------------------------------------
print("Downloading dataset...")
response = requests.get(url, stream=True)

with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download completed.")

