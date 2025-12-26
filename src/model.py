# -*- coding: utf-8 -*-
"""Neural network model architectures for MNIST digit recognition.

This module defines CNN architectures for handwritten digit classification,
including the baseline ImprovedNet and an enhanced ResidualCNN.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedNet(nn.Module):
    """Improved CNN with batch normalization and dropout.
    
    A convolutional neural network with three convolutional layers,
    batch normalization, max pooling, and fully connected layers.
    
    Attributes:
        conv1: First convolutional layer (1 -> 32 channels).
        bn1: Batch normalization for first conv layer.
        conv2: Second convolutional layer (32 -> 64 channels).
        bn2: Batch normalization for second conv layer.
        conv3: Third convolutional layer (64 -> 128 channels).
        bn3: Batch normalization for third conv layer.
        pool: Max pooling layer.
        dropout: Dropout layer for regularization.
        fc1: First fully connected layer.
        fc2: Second fully connected layer.
        fc3: Output layer (10 classes).
    """
    
    def __init__(self, dropout_rate: float = 0.25) -> None:
        """Initialize the ImprovedNet architecture.
        
        Args:
            dropout_rate: Probability of dropout for regularization.
        """
        super(ImprovedNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fully connected layers
        # After 3 pooling operations: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            Output tensor of shape (batch_size, 10) with logits.
        """
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection.
    
    A basic residual block that adds the input to the output,
    enabling better gradient flow in deeper networks.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the first convolution.
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor with skip connection added.
        """
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class ResidualCNN(nn.Module):
    """Enhanced CNN with residual connections.
    
    A deeper architecture using residual blocks for better
    gradient flow and potentially higher accuracy.
    """
    
    def __init__(self, dropout_rate: float = 0.25) -> None:
        """Initialize ResidualCNN architecture.
        
        Args:
            dropout_rate: Probability of dropout for regularization.
        """
        super(ResidualCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 64, stride=2)   # 28 -> 14
        self.layer2 = self._make_layer(64, 128, stride=2)  # 14 -> 7
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128, 10)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    stride: int) -> nn.Sequential:
        """Create a layer with residual blocks.
        
        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            stride: Stride for downsampling.
        
        Returns:
            Sequential container with residual blocks.
        """
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).
        
        Returns:
            Output tensor of shape (batch_size, 10) with logits.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_model(model_name: str = "improved", dropout_rate: float = 0.25) -> nn.Module:
    """Factory function to create a model by name.
    
    Args:
        model_name: Name of the model architecture ("improved" or "residual").
        dropout_rate: Dropout probability for regularization.
    
    Returns:
        Instantiated neural network model.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    models = {
        "improved": ImprovedNet,
        "residual": ResidualCNN,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](dropout_rate=dropout_rate)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
