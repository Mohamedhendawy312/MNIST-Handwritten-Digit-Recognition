# -*- coding: utf-8 -*-
"""Training pipeline for MNIST digit recognition.

This module provides the training loop, model checkpointing,
and training history tracking.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import Config, DEFAULT_CONFIG
from .model import get_model, count_parameters
from .utils import set_seed, get_data_loaders, get_device
from .evaluate import evaluate_model


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """Train the model for one epoch.
    
    Args:
        model: Neural network model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{running_loss / (pbar.n + 1):.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_model(
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model with the given configuration.
    
    Args:
        config: Configuration object with hyperparameters.
        verbose: Whether to print training progress.
    
    Returns:
        Tuple of (trained_model, training_history).
        Training history contains: train_loss, train_acc, val_loss, val_acc.
    """
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config)
    if verbose:
        print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders(config, augment=True)
    if verbose:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = get_model(config.model_name, config.dropout_rate)
    model = model.to(device)
    if verbose:
        print(f"Model: {config.model_name} ({count_parameters(model):,} parameters)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler (OneCycleLR for faster convergence)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate * 10,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Training history
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_path = config.model_dir / "best_model.pth"
    
    # Training loop
    if verbose:
        print(f"\nStarting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.epochs
        )
        
        # Update scheduler after each batch (handled in train_epoch via closure)
        scheduler.step()
        
        # Validate
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device, criterion)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "model_name": config.model_name,
                    "dropout_rate": config.dropout_rate,
                }
            }, best_model_path)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    if verbose:
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best model saved to: {best_model_path}")
    
    # Load best model for return
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, history


def load_model(model_path: Path, device: torch.device = None) -> nn.Module:
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint.
        device: Device to load model to.
    
    Returns:
        Loaded model in evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get("config", {})
    model_name = model_config.get("model_name", "improved")
    dropout_rate = model_config.get("dropout_rate", 0.25)
    
    # Create and load model
    model = get_model(model_name, dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def main() -> None:
    """Main entry point for training."""
    import json
    
    config = Config()
    
    print("=" * 60)
    print("MNIST Digit Recognition - Training")
    print("=" * 60)
    
    # Train the model
    model, history = train_model(config, verbose=True)
    
    # Save training history for visualization
    history_path = config.output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    device = get_device(config)
    _, _, test_loader = get_data_loaders(config, augment=False)
    
    from .evaluate import evaluate_model, print_classification_report
    
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model, test_loader, device
    )
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print_classification_report(all_labels, all_preds)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    from .visualize import save_all_visualizations
    save_all_visualizations(model, history, config)


if __name__ == "__main__":
    main()

