# -*- coding: utf-8 -*-
"""Configuration module for MNIST digit recognition.

This module contains all hyperparameters and configuration settings
for the MNIST digit recognition project.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class Config:
    """Configuration dataclass containing all hyperparameters.
    
    Attributes:
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for optimizer.
        epochs: Number of training epochs.
        train_split: Fraction of data used for training.
        val_split: Fraction of data used for validation.
        dropout_rate: Dropout probability for regularization.
        seed: Random seed for reproducibility.
        device: Device to run training on (cuda/cpu).
        data_dir: Directory for MNIST data storage.
        output_dir: Directory for saving outputs.
        model_dir: Directory for saving model checkpoints.
    """
    
    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 20
    train_split: float = 0.8
    val_split: float = 0.1
    dropout_rate: float = 0.25
    seed: int = 42
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models")
    
    # Data augmentation
    rotation_degrees: float = 10.0
    translate_range: Tuple[float, float] = (0.1, 0.1)
    
    # Model selection
    model_name: str = "improved"  # Options: "improved", "residual"
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.model_dir = Path(self.model_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = Config()
