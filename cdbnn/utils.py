import os
import shutil
import requests
import tarfile
import zipfile
import gzip
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
    elif file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            image = np.frombuffer(f.read(), dtype=np.uint8)
        # Handle MNIST .gz files specifically
        if 'idx3-ubyte' in file_path:  # MNIST image file
            image = image[16:].reshape(-1, 28, 28)
        elif 'idx1-ubyte' in file_path:  # MNIST label file
            image = image[8:]
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
    elif filename.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[:-3], 'wb') as f_out:  # Remove .gz extension
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # Remove the downloaded file
    os.remove(filepath)

def find_images_folder(root_dir):
    """Recursively find the folder containing image files."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dcm', '.fits')):
                return dirpath  # Found a folder containing images
    return None

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

    # Recursively find the folder containing images
    images_folder = find_images_folder(train_dir)
    if not images_folder:
        raise ValueError(f"No image folders found in the dataset directory: {train_dir}")

    # Backtrack one step to find the class labels
    class_folder = os.path.dirname(images_folder)
    return class_folder
