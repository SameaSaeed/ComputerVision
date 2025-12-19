import os
import requests
import zipfile
from urllib.parse import urlparse

def download_file(url, local_filename):
    """Download file from URL"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def create_sample_dataset():
    """Create sample agricultural dataset structure"""
    
    # Create directory structure
    categories = ['healthy_crops', 'diseased_crops', 'bare_soil', 'weeds']
    
    for category in categories:
        os.makedirs(f'data/{category}', exist_ok=True)
    
    print("Dataset structure created successfully!")
    print("Categories:", categories)

if __name__ == "__main__":
    create_sample_dataset()