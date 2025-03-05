import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import load_image, get_image_properties, prepare_dataset, infer_classes_from_folder

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = infer_classes_from_folder(img_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_files = self._get_image_files()
        if not self.img_files:
            raise ValueError(f"No image files found in the dataset directory: {img_dir}")
        self.image_properties = self._get_image_properties()

    def _get_image_files(self):
        """Get a list of all image files in the dataset."""
        img_files = []
        for cls in self.classes:
            cls_dir = os.path.join(self.img_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Class directory not found: {cls_dir}")
                continue
            for file in os.listdir(cls_dir):
                file_path = os.path.join(cls_dir, file)
                if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dcm', '.fits')):
                    img_files.append((file_path, self.class_to_idx[cls]))
                else:
                    print(f"Warning: Skipping unsupported file: {file_path}")
        return img_files

    def _get_image_properties(self):
        """Get the properties of the first image in the dataset."""
        first_image_path = self.img_files[0][0]
        first_image = load_image(first_image_path)
        return get_image_properties(first_image)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path, label = self.img_files[idx]
        image = load_image(img_path)

        # Convert grayscale images to 3 channels if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Convert to PIL Image for transformations
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
