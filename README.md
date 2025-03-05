# CNN for Image Classification with Subtle Detail Detection

This repository contains a PyTorch implementation of a 7-layer Convolutional Neural Network (CNN) designed for image classification with a focus on detecting subtle details in images. The model is generic and can be adapted to various types of image data, such as radiology, astronomy, or standard image formats.

## Features

- **Dynamic Image Properties**: Automatically reads image shape, dimensions, and data type from input files.
- **Generic Dataset Support**: Works with any dataset by specifying the dataset name (e.g., `cdbnn`).
- **Output CSV Format**: Saves extracted features in the specified format (`feature_0`, `feature_1`, ..., `feature_127`, `target`).
- **Flexible Input Formats**: Supports various input formats (e.g., DICOM, FITS, PNG, JPG, etc.).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnn-subtle-detail-detection.git
   cd cnn-subtle-detail-detection
