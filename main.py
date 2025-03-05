import argparse
from mnist_cnn.data_loader import CustomImageDataset
from mnist_cnn.model import SubtleDetailCNN
from mnist_cnn.train import train_model
from mnist_cnn.config_generator import generate_config_json, generate_output_json
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for image classification.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., cdbnn).")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    # Load dataset to determine image properties
    train_dataset = CustomImageDataset(img_dir=args.train_dir, transform=None)
    image_shape = train_dataset.image_properties[0]

    # Generate configuration files
    generate_config_json(args.dataset_name, image_shape, args.num_classes)
    generate_output_json(args.dataset_name, image_shape, args.num_classes)

    # Initialize model, optimizer, and loss function
    model = SubtleDetailCNN(in_channels=image_shape[0], num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_dataset, criterion, optimizer, num_epochs=args.epochs, dataset_name=args.dataset_name)

if __name__ == "__main__":
    main()
