import os
import shutil
import requests
import tarfile
import zipfile
from urllib.parse import urlparse
from PIL import Image
import numpy as np
import pydicom
from astropy.io import fits

def load_image(file_path):
    """Load an image from a file and return it as a numpy array."""
    if file_path.endswith('.dcm'):
        ds = pydicom.dcmread(file_path)
        image = ds.pixel_array
    elif file_path.endswith('.fits'):
        hdul = fits.open(file_path)
        image = hdul[0].data
    else:
        image = Image.open(file_path)
        image = np.array(image)
    return image

def get_image_properties(image):
    """Get the shape, dimensions, and data type of an image."""
    shape = image.shape
    dimensions = len(shape)
    dtype = str(image.dtype)
    return shape, dimensions, dtype

def download_and_extract(url, extract_dir):
    """Download and extract a compressed file from a URL."""
    os.makedirs(extract_dir, exist_ok=True)
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    filepath = os.path.join(extract_dir, filename)

    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the file
    print(f"Extracting {filename}...")
    if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    elif filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(path=extract_dir)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # Remove the downloaded file
    os.remove(filepath)

def infer_classes_from_folder(folder):
    """Infer class labels from subfolders in the given folder."""
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    return sorted(subfolders)

def prepare_dataset(data_path, dataset_name):
    """Prepare the dataset by downloading, extracting, and organizing it."""
    data_dir = os.path.join("data", dataset_name)
    train_dir = os.path.join(data_dir, "train_data")
    os.makedirs(train_dir, exist_ok=True)

    if data_path.startswith(('http://', 'https://', 'ftp://')):
        # Download and extract the dataset
        download_and_extract(data_path, train_dir)
    elif os.path.isdir(data_path):
        # Copy or link the dataset
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.symlink(data_path, train_dir)
    else:
        # Check if the dataset is available in PyTorch
        try:
            import torchvision.datasets as datasets
            dataset_class = getattr(datasets, dataset_name.upper(), None)
            if dataset_class:
                dataset = dataset_class(root=data_dir, train=True, download=True)
                return os.path.join(data_dir, dataset_name.upper())
            else:
                raise ValueError(f"Dataset {dataset_name} not found in PyTorch.")
        except ImportError:
            raise ValueError("PyTorch is required to download standard datasets.")

    # Merge train and test folders if they exist
    train_subdir = os.path.join(train_dir, "train")
    test_subdir = os.path.join(train_dir, "test")
    if os.path.exists(train_subdir) and os.path.exists(test_subdir):
        merge = input("Found train and test folders. Merge them? (y/n): ").lower() == 'y'
        if merge:
            for class_folder in os.listdir(test_subdir):
                src = os.path.join(test_subdir, class_folder)
                dst = os.path.join(train_subdir, class_folder)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            shutil.rmtree(test_subdir)

    return train_dir
