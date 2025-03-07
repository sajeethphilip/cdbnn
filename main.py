import argparse
from cdbnn.data_loader import CustomImageDataset
from cdbnn.model import SubtleDetailCNN
from cdbnn.train import train_model
from cdbnn.config_generator import ConfigGenerator
from cdbnn.utils import prepare_dataset, reconstruct_images
import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train, test, or predict using a CNN model for image classification.")
    parser.add_argument("--dataset", type=str, required=True, help="Name or path of the dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "predict"], default=None,
                        help="Mode to run: train, test, or predict. If not specified, config settings will be used.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the saved model for testing or prediction.")
    parser.add_argument("--invert_DBNN", action="store_true", help="Enable inverse DBNN for image reconstruction.")
    parser.add_argument("--fresh_start", action="store_true", help="Start training from scratch, ignoring saved models.")
    args = parser.parse_args()

    # Prepare the dataset
    dataset_name = args.dataset
    dataset_dir = os.path.join("data", dataset_name)

    # Check if the dataset exists locally
    if not os.path.exists(dataset_dir):
        print(f"Dataset '{dataset_name}' not found locally. Attempting to download...")
        try:
            # Use the prepare_dataset function to download and organize the dataset
            dataset_dir = prepare_dataset(dataset_name, dataset_name)
            print(f"Dataset '{dataset_name}' downloaded and organized at: {dataset_dir}")
        except Exception as e:
            raise ValueError(f"Failed to download or prepare dataset '{dataset_name}': {e}")

    # Set train and test directories
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Ensure the train directory exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Generate configurations
    config_generator = ConfigGenerator(dataset_name, dataset_dir)
    config = config_generator.generate_default_config(train_dir)

    # Load dataset to determine image properties and number of classes
    train_dataset = CustomImageDataset(img_dir=train_dir, transform=None)
    in_channels = train_dataset.image_properties  # Fetch the number of channels
    num_classes = len(train_dataset.classes)
    print(f"The number of channels input is {in_channels}")

    # Initialize model with the correct in_channels
    model = SubtleDetailCNN(in_channels=in_channels, num_classes=num_classes)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the best saved model unless fresh_start is True
    best_model_path = os.path.join(dataset_dir, "models", "best_model.pth")
    if not args.fresh_start and os.path.exists(best_model_path):
        print(f"Loading the best saved model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

    # Determine mode from config if not specified in command line
    if args.mode is None:
        # Ensure the execution_flags dictionary has the required keys
        execution_flags = config.get("execution_flags", {})
        train_flag = execution_flags.get("train", False)
        train_only_flag = execution_flags.get("train_only", False)
        predict_flag = execution_flags.get("predict", False)

        # Determine the mode based on the flags
        if train_flag and predict_flag:
            args.mode = "train"  # Train and predict
        elif train_only_flag:
            args.mode = "train"  # Train only
        elif predict_flag:
            args.mode = "predict"  # Predict only
        else:
            args.mode = "train"  # Default to train if no flags are set

        print(f"Using mode from config: {args.mode}")

    if args.mode == "train":
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        train_model(model, train_dataset, criterion, optimizer, num_epochs=args.epochs,
                    dataset_name=args.dataset, invert_DBNN=args.invert_DBNN)

        # If invert_DBNN is True, reconstruct images from the generated CSV file
        if args.invert_DBNN:
            csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
            output_dir = os.path.join(dataset_dir, "reconstructed_images")
            os.makedirs(output_dir, exist_ok=True)
            reconstruct_images(csv_path, model, output_dir, device)

    elif args.mode == "test":
        if args.model_path is None:
            raise ValueError("Model path must be provided for testing.")

        # Load the saved model
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        # Test the model
        test_dataset = CustomImageDataset(img_dir=test_dir, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loss, test_accuracy = test_model(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    elif args.mode == "predict":
        if args.model_path is None:
            raise ValueError("Model path must be provided for prediction.")

        # Load the saved model
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        # Perform prediction
        predict_dataset = CustomImageDataset(img_dir=test_dir, transform=None)
        predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False, num_workers=4)
        predictions = predict_model(model, predict_loader, device)

        # Save predictions to a CSV file
        predictions_path = os.path.join(dataset_dir, "predictions.csv")
        pd.DataFrame(predictions, columns=["image_path", "predicted_class"]).to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        # If invert_DBNN is True, reconstruct images from the generated CSV file
        if args.invert_DBNN:
            csv_path = predictions_path
            output_dir = os.path.join(dataset_dir, "reconstructed_images")
            os.makedirs(output_dir, exist_ok=True)
            reconstruct_images(csv_path, model, output_dir, device)

if __name__ == "__main__":
    main()
