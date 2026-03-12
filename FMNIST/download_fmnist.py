"""
Manual FashionMNIST Dataset Downloader

This script downloads FashionMNIST dataset from alternative mirrors
when the default PyTorch download fails.
"""

import os
import urllib.request
import gzip
import shutil

# Create data directory
data_dir = "./data/FashionMNIST/raw"
os.makedirs(data_dir, exist_ok=True)

# Alternative mirrors for FashionMNIST
base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

files = {
    "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte"
}

print("Downloading FashionMNIST dataset from alternative source...")
print(f"Saving to: {data_dir}")
print("="*60)

for filename, extracted_name in files.items():
    url = base_url + filename
    filepath = os.path.join(data_dir, filename)
    extracted_path = os.path.join(data_dir, extracted_name)
    
    # Skip if already exists
    if os.path.exists(extracted_path):
        print(f"✓ {extracted_name} already exists, skipping...")
        continue
    
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded {filename}")
        
        # Extract the gzip file
        print(f"Extracting {filename}...")
        with gzip.open(filepath, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Extracted to {extracted_name}")
        
        # Remove the compressed file
        os.remove(filepath)
        
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        continue

print("\n" + "="*60)
print("Download complete! You can now run main_complete_rl.py")
print("="*60)
