import argparse
from cdbnn.data_loader import CustomImageDataset
from cdbnn.model import SubtleDetailCNN
from cdbnn.train import train_model
from cdbnn.config_generator import generate_config_json, generate_output_json
from cdbnn.utils import prepare_dataset
import torch
import torch.optim as optim
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for image classification.")
    parser.add_argument("--dataset", type=str, required=True, help="Name or path of the dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    # Prepare the dataset
    train_dir = prepare_dataset(args.dataset, args.dataset)

    # Load dataset to determine image properties and number of classes
    train_dataset = CustomImageDataset(img_dir=train_dir, transform=None)
    image_shape = train_dataset.image_properties
    num_classes = len(train_dataset.classes)

    # Generate configuration files
    generate_config_json(args.dataset, image_shape, num_classes)
    generate_output_json(args.dataset, image_shape, num_classes)

    # Initialize model, optimizer, and loss function
    model = SubtleDetailCNN(in_channels=image_shape[0], num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_dataset, criterion, optimizer, num_epochs=args.epochs, dataset_name=args.dataset)

if __name__ == "__main__":
    main()
