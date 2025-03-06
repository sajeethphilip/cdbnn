import torch
import torch.nn as nn

class SubtleDetailCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=(224, 224)):
        super(SubtleDetailCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the size of the feature map after convolutions and pooling
        # For 224x224 input, after 6 pooling layers: 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3
        # For 28x28 input (MNIST), after 3 pooling layers: 28 -> 14 -> 7 -> 3
        h, w = input_size
        for _ in range(6):  # 6 pooling layers
            h = (h + 1) // 2  # Integer division with ceiling
            w = (w + 1) // 2

            # Break if dimensions become too small
            if h < 1 or w < 1:
                raise ValueError(f"Input size {input_size} is too small for this model architecture")

        # Final feature map size
        self.feature_size = 128 * h * w

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Track the shape for debugging
        original_shape = x.shape

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = torch.relu(self.conv7(x))

        # Current shape before flattening
        conv_shape = x.shape

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        flat_shape = x.shape

        # Check if the flattened shape matches our expectation
        if x.shape[1] != self.feature_size:
            print(f"Warning: Expected feature size {self.feature_size}, got {x.shape[1]}")
            print(f"Original input shape: {original_shape}, After convolutions: {conv_shape}")
            # Dynamically adjust the first fully connected layer
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
            self.feature_size = x.shape[1]

        features = torch.relu(self.fc1(x))
        x = self.fc2(features)
        return x, features


class InverseSubtleDetailCNN(nn.Module):
    def __init__(self, in_channels, num_classes, output_size=(224, 224)):
        super(InverseSubtleDetailCNN, self).__init__()
        # Calculate feature size based on output dimensions
        h, w = output_size
        for _ in range(6):  # 6 pooling layers in the original model
            h = (h + 1) // 2
            w = (w + 1) // 2

        # Break if dimensions become too small
        if h < 1 or w < 1:
            raise ValueError(f"Output size {output_size} is too small for this model architecture")

        self.feature_size = 128 * h * w

        self.fc1 = nn.Linear(num_classes, self.feature_size)
        self.deconv1 = nn.ConvTranspose2d(128, 1024, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv7 = nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=1, padding=1)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.output_h, self.output_w = h, w

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.view(-1, 128, self.output_h, self.output_w)
        x = self.unpool(torch.relu(self.deconv1(x)))
        x = self.unpool(torch.relu(self.deconv2(x)))
        x = self.unpool(torch.relu(self.deconv3(x)))
        x = self.unpool(torch.relu(self.deconv4(x)))
        x = self.unpool(torch.relu(self.deconv5(x)))
        x = self.unpool(torch.relu(self.deconv6(x)))
        x = torch.sigmoid(self.deconv7(x))  # Use sigmoid to ensure pixel values are between 0 and 1
        return x
