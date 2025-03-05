import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import load_image, get_image_properties

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = os.listdir(img_dir)
        self.image_properties = self._get_image_properties()

    def _get_image_properties(self):
        """Get the properties of the first image in the directory."""
        first_image_path = os.path.join(self.img_dir, self.img_files[0])
        first_image = load_image(first_image_path)
        return get_image_properties(first_image)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = load_image(img_path)

        # Convert grayscale images to 3 channels if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Convert to PIL Image for transformations
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Assuming filename format: label_imageid.ext
        label = int(self.img_files[idx].split('_')[0])
        return image, label
