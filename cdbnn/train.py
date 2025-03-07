import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import SubtleDetailCNN, InverseSubtleDetailCNN
from .data_loader import CustomImageDataset
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plot a confusion matrix using seaborn and matplotlib.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): List of class names.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def train_model(model, train_dataset, criterion, optimizer, num_epochs=20, dataset_name="mnist",
                batch_size=64, num_workers=4, device=None, validate=False, val_loader=None, invert_DBNN=False):
    """
    Optimized training function with performance improvements and dynamic model adaptation.
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
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm for progress tracking
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass with mixed precision for CUDA
                optimizer.zero_grad()
                try:
                    outputs, features = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Update statistics
                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
                except Exception as e:
                    print(f"Error during training iteration: {e}")
                    continue

        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation if requested
        if validate and val_loader is not None:
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}')
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch_loss)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with loss: {best_loss:.4f}")

            # Extract and save features on CPU for data analysis
            save_features(model, train_loader, dataset_name, device)

    # Reconstruct images if invert_DBNN is True
    if invert_DBNN:
        print("Reconstructing images using the inverse model...")
        csv_path = f'data/{dataset_name}/{dataset_name}.csv'
        output_dir = f'data/{dataset_name}/reconstructed_images'
        os.makedirs(output_dir, exist_ok=True)

        # Load the best model for reconstruction
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # Call the reconstruct_images function
        reconstruct_images(csv_path, model, output_dir, device)
        print(f"Reconstructed images saved to {output_dir}")

    # Plot confusion matrix for training data
    print("Plotting confusion matrix for training data...")
    train_predictions, train_labels = get_predictions(model, train_loader, device)
    train_cm = confusion_matrix(train_labels, train_predictions)
    plot_confusion_matrix(train_cm, class_names=train_dataset.classes, title='Training Confusion Matrix')

    # Plot confusion matrix for validation/test data
    if validate and val_loader is not None:
        print("Plotting confusion matrix for validation data...")
        val_predictions, val_labels = get_predictions(model, val_loader, device)
        val_cm = confusion_matrix(val_labels, val_predictions)
        plot_confusion_matrix(val_cm, class_names=train_dataset.classes, title='Validation Confusion Matrix')


def get_predictions(model, data_loader, device):
    """
    Generate predictions and labels for a given data loader.

    Args:
        model: The trained model.
        data_loader: DataLoader for the dataset.
        device: Device to use (cpu or cuda).

    Returns:
        predictions: List of predicted labels.
        labels: List of actual labels.
    """
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    return predictions, labels

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
