from PIL import Image
import pydicom
from astropy.io import fits

def load_image(file_path):
    if file_path.endswith('.dcm'):
        ds = pydicom.dcmread(file_path)
        image = ds.pixel_array
    elif file_path.endswith('.fits'):
        hdul = fits.open(file_path)
        image = hdul[0].data
    else:
        image = Image.open(file_path)
    return image
