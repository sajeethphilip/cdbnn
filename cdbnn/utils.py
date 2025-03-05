import os
import numpy as np
from PIL import Image
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
