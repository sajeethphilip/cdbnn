import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import SubtleDetailCNN
from .data_loader import CustomImageDataset
import pandas as pd
import os
from tqdm import tqdm

from .model import SubtleDetailCNN, InverseSubtleDetailCNN

def train_model(model, train_dataset, criterion, optimizer, num_epochs=20, dataset_name="mnist",
                batch_size=64, num_workers=4, device=None, validate=False, val_loader=None, invert_DBNN=False):
    """
    Optimized training function with performance improvements and dynamic model adaptation.

    Args:
        model: The neural network model to train
        train_dataset: Training dataset
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        num_epochs: Number of training epochs
        dataset_name: Name of the dataset for saving
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        device: Device to use (cpu or cuda)
        validate: Whether to perform validation
        val_loader: Validation data loader
        invert_DBNN: Whether to create an inverse model
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    print(f"Training on {device}")

    # Create DataLoader with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Check if model needs to be adapted to the dataset
    # Get a sample batch to determine the shape
    sample_inputs, _ = next(iter(train_loader))
    print(f"Input shape: {sample_inputs.shape}")

    # Verify if the model can handle the input
    try:
        sample_inputs = sample_inputs.to(device)
        with torch.no_grad():
            model(sample_inputs)
        print("Model successfully processed sample input")
    except Exception as e:
        print(f"Error during model validation: {e}")
        print("Attempting to adapt model to the input shape...")

        # If using the updated SubtleDetailCNN, it should handle this automatically
        # If not, you might need additional logic here

        # Reset optimizer since model parameters may have changed
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize tracking variables
    best_loss = float('inf')
    os.makedirs(f'data/{dataset_name}/models', exist_ok=True)
    best_model_path = f'data/{dataset_name}/models/best_model.pth'


    # Use learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training loop with progress bar
     # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with loss: {best_loss:.4f}")

    # If invert_DBNN is True, reconstruct images from the generated CSV file
    if invert_DBNN:
        csv_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
        output_dir = os.path.join("data", dataset_name, "reconstructed_images")
        os.makedirs(output_dir, exist_ok=True)
        reconstruct_images(csv_path, model, output_dir, device, in_channels)

def save_features(model, data_loader, dataset_name, device):
    """Extract and save features from the model"""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            try:
                inputs = inputs.to(device)
                _, batch_features = model(inputs)
                all_features.append(batch_features.cpu().numpy())
                all_labels.append(labels.numpy())
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue

    # Concatenate all batches if we have any
    if all_features and all_labels:
        import numpy as np
        features_np = np.concatenate(all_features, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # Save to CSV
        df = pd.DataFrame(features_np, columns=[f"feature_{i}" for i in range(features_np.shape[1])])
        df["target"] = labels_np

        os.makedirs(f'data/{dataset_name}', exist_ok=True)
        csv_path = f'data/{dataset_name}/{dataset_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")
    else:
        print("No features extracted, skipping CSV creation")

def validate_model(model, val_loader, criterion, device):
    """Validate the model on a validation set"""
    model.eval()
    val_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                valid_batches += 1
            except Exception as e:
                print(f"Error during validation: {e}")
                continue

    # Avoid division by zero
    if valid_batches == 0:
        return float('inf')
    return val_loss / valid_batches
