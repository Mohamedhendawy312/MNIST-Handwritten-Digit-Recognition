# -*- coding: utf-8 -*-
"""Utility functions for data loading and reproducibility.

This module provides helper functions for loading the MNIST dataset,
creating data loaders, and ensuring reproducibility.
"""

import random
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from .config import Config, DEFAULT_CONFIG


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(augment: bool = True, config: Config = DEFAULT_CONFIG) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get data transforms for training and evaluation.
    
    Args:
        augment: Whether to apply data augmentation for training.
        config: Configuration object with augmentation parameters.
    
    Returns:
        Tuple of (train_transform, eval_transform).
    """
    # Base transforms for evaluation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=config.rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=config.translate_range,
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = eval_transform
    
    return train_transform, eval_transform


def get_data_loaders(config: Config = DEFAULT_CONFIG, 
                     augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object with data parameters.
        augment: Whether to apply data augmentation.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform, eval_transform = get_transforms(augment=augment, config=config)
    
    # Load full training dataset with training transform
    full_train_dataset = torchvision.datasets.MNIST(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test dataset separately
    test_dataset = torchvision.datasets.MNIST(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=eval_transform
    )
    
    # Split training data into train/val
    total_size = len(full_train_dataset)
    train_size = int(config.train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_device(config: Config = DEFAULT_CONFIG) -> torch.device:
    """Get the device for training/inference.
    
    Args:
        config: Configuration object.
    
    Returns:
        torch.device object.
    """
    return torch.device(config.device)
