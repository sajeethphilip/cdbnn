import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import load_image, get_image_properties, prepare_dataset, infer_classes_from_folder

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, input_size=(224, 224)):
        self.img_dir = img_dir
        self.input_size = input_size  # Default size that works with the model

        # Create a transform that includes resizing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor()
            ])
        else:
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
        try:
            # Load as PIL Image for proper resizing
            pil_image = Image.open(first_image_path).convert('RGB')

            # Check channels
            if pil_image.mode == 'L':  # Grayscale
                in_channels = 1
            elif pil_image.mode == 'RGB':  # RGB
                in_channels = 3
            else:
                in_channels = len(pil_image.getbands())

            return in_channels
        except Exception as e:
            print(f"Error reading first image: {e}")
            # Default to 3 channels if there's an error
            return 3

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path, label = self.img_files[idx]

        try:
            # Load as PIL Image for better resizing
            pil_image = Image.open(img_path)

            # Convert grayscale to RGB if needed
            if pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')

            # Apply transformations (including resize)
            image = self.transform(pil_image)

            return image, label

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a blank image in case of error
            blank = torch.zeros((3, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            return blank, label
