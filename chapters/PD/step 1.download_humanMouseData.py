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
url = "https://zenodo.org/records/4488164/files/NN_human_mouse_eyes.zip?download=1"
zip_path = os.path.join(data_dir, "NN_human_mouse_eyes.zip")

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

