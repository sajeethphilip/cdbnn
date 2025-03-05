import argparse
from cdbnn.data_loader import CustomImageDataset
from cdbnn.model import SubtleDetailCNN
from cdbnn.train import train_model
from cdbnn.config_generator  import ConfigGenerator
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for image classification.")
    parser.add_argument("--dataset", type=str, required=True, help="Name or path of the dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    # Prepare the dataset
    dataset_name = args.dataset
    dataset_dir = os.path.join("data", dataset_name)
    train_dir = os.path.join(dataset_dir, "train_data")

    # Generate configurations
    config_generator = ConfigGenerator(dataset_name, dataset_dir)
    config = config_generator.generate_default_config(train_dir)

    # Load dataset to determine image properties and number of classes
    train_dataset = CustomImageDataset(img_dir=train_dir, transform=None)
    image_shape = train_dataset.image_properties
    num_classes = len(train_dataset.classes)

    # Initialize model, optimizer, and loss function
    model = SubtleDetailCNN(in_channels=image_shape[0], num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_dataset, criterion, optimizer, num_epochs=args.epochs, dataset_name=args.dataset)

if __name__ == "__main__":
    main()
