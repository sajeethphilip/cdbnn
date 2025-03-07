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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import logging
import pandas as pd

logger = logging.getLogger(__name__)

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
    if not os.path.isdir(folder):
        raise ValueError(f"Invalid folder path: {folder}")
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    return sorted(subfolders)

def _process_torchvision(dataset_name, output_dir):
    """Process torchvision dataset."""
    dataset_name=dataset_name.upper()
    if not hasattr(datasets, dataset_name):
        raise ValueError(f"Torchvision dataset {dataset_name} not found")

    # Setup paths in dataset-specific directory
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Download and process datasets
    transform = transforms.ToTensor()

    train_dataset = getattr(datasets, dataset_name)(
        root=output_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = getattr(datasets, dataset_name)(
        root=output_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Save images with class directories
    def save_dataset_images(dataset, output_dir, split_name):
        logger.info(f"Processing {split_name} split...")

        class_to_idx = getattr(dataset, 'class_to_idx', None)
        if class_to_idx:
            idx_to_class = {v: k for k, v in class_to_idx.items()}

        with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
            for idx, (img, label) in enumerate(dataset):
                class_name = idx_to_class[label] if class_to_idx else str(label)
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)

                img_path = os.path.join(class_dir, f"{idx}.png")
                img.save(img_path)
                pbar.update(1)

    save_dataset_images(train_dataset, train_dir, "training")
    save_dataset_images(test_dataset, test_dir, "test")

    return train_dir  # Return only the train directory

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
            return _process_torchvision(dataset_name, data_dir)
        except ValueError as e:
            raise ValueError(f"Dataset {dataset_name} not found in PyTorch: {e}")

    # Recursively find the folder containing images
    images_folder = find_images_folder(train_dir)
    if not images_folder:
        raise ValueError(f"No image folders found in the dataset directory: {train_dir}")

    # Backtrack one step to find the class labels
    class_folder = os.path.dirname(images_folder)
    return class_folder


def reconstruct_images(csv_path, model, output_dir, device):
    """Reconstruct images from features in CSV file."""
    import pandas as pd
    import numpy as np
    from PIL import Image

    # Read the CSV file
    df = pd.read_csv(csv_path)
    features = df.drop(columns=['target']).values
    labels = df['target'].values

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (feature, label) in enumerate(zip(features, labels)):
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            reconstructed_image = model(feature_tensor).squeeze(0).cpu().numpy()
            reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Scale to 0-255

            # Create subfolder based on class label
            class_dir = os.path.join(output_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            # Save the reconstructed image
            image_path = os.path.join(class_dir, f"reconstructed_{i}.png")
            Image.fromarray(reconstructed_image.transpose(1, 2, 0)).save(image_path)

    print(f"Reconstructed images saved to {output_dir}")

def predict_model(model, predict_loader, device):
    """Generate predictions for the input dataset."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in predict_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(list(zip(predict_loader.dataset.img_files, predicted.cpu().numpy())))

    return predictions

def test_model(model, test_loader, criterion, device):
    """Evaluate the model on the test dataset."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    return test_loss, test_accuracy
