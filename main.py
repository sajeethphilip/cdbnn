import argparse
from cdbnn.data_loader import CustomImageDataset, get_transform
from cdbnn.model import SubtleDetailCNN
from cdbnn.train import train_model
from cdbnn.config_generator import generate_config_json
from cdbnn.utils import load_image
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for image classification.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    # Generate configuration file
    image_shape = (224, 224)
    generate_config_json(image_shape, args.num_classes, args.dataset_name)

    # Load dataset
    transform = get_transform()
    train_dataset = CustomImageDataset(img_dir=args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = SubtleDetailCNN(in_channels=1, num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, dataset_name=args.dataset_name)

if __name__ == "__main__":
    main()
