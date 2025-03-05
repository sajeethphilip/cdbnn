import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import SubtleDetailCNN
from .data_loader import CustomImageDataset
import pandas as pd

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, dataset_name="mnist"):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, features = model(inputs)
            
            # Debug shapes
            print(f"Outputs shape: {outputs.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Ensure labels are a tensor
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'data/{dataset_name}/models/best_model.pth')
            # Save features to CSV
            features_np = features.detach().numpy()
            df = pd.DataFrame(features_np, columns=[f"feature_{i}" for i in range(128)])
            df["target"] = labels.numpy()
            df.to_csv(f'data/{dataset_name}/{dataset_name}.csv', index=False)
            
