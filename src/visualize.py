# -*- coding: utf-8 -*-
"""Visualization utilities for MNIST digit recognition.

This module provides functions for creating publication-quality
visualizations of model results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from .config import Config, DEFAULT_CONFIG
from .evaluate import get_confusion_matrix, evaluate_model
from .utils import get_data_loaders, get_device


# Set style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: Optional path to save the figure.
        figsize: Figure size in inches.
    
    Returns:
        Matplotlib figure object.
    """
    cm = get_confusion_matrix(y_true, y_pred)
    
    # Normalize for percentage display
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        cbar_kws={"label": "Percentage (%)"}
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized %)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot training and validation curves.
    
    Args:
        history: Dictionary with train_loss, train_acc, val_loss, val_acc.
        save_path: Optional path to save the figure.
        figsize: Figure size in inches.
    
    Returns:
        Matplotlib figure object.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-s", label="Validation Loss", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-s", label="Validation Accuracy", markersize=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([90, 100])  # Focus on high accuracy range
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to: {save_path}")
    
    return fig


def plot_sample_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 25,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """Plot grid of sample predictions with correct/incorrect highlighting.
    
    Args:
        model: Trained neural network model.
        data_loader: Data loader to sample from.
        device: Device for inference.
        num_samples: Number of samples to display (should be a perfect square).
        save_path: Optional path to save the figure.
        figsize: Figure size in inches.
    
    Returns:
        Matplotlib figure object.
    """
    model.eval()
    
    # Collect samples
    images_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            images_list.extend(images.cpu())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(predicted.cpu().numpy())
            
            if len(images_list) >= num_samples:
                break
    
    # Select subset
    images_list = images_list[:num_samples]
    labels_list = labels_list[:num_samples]
    preds_list = preds_list[:num_samples]
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_list):
            img = images_list[idx].squeeze().numpy()
            true_label = labels_list[idx]
            pred_label = preds_list[idx]
            
            # Denormalize for display
            img = img * 0.3081 + 0.1307
            img = np.clip(img, 0, 1)
            
            ax.imshow(img, cmap="gray")
            
            # Color code: green for correct, red for incorrect
            is_correct = true_label == pred_label
            color = "green" if is_correct else "red"
            
            ax.set_title(f"True: {true_label} | Pred: {pred_label}", 
                        fontsize=9, color=color, fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample predictions to: {save_path}")
    
    return fig


def plot_misclassified_samples(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """Plot grid of misclassified samples.
    
    Args:
        model: Trained neural network model.
        data_loader: Data loader to sample from.
        device: Device for inference.
        num_samples: Maximum number of misclassified samples to display.
        save_path: Optional path to save the figure.
        figsize: Figure size in inches.
    
    Returns:
        Matplotlib figure object.
    """
    model.eval()
    
    # Collect misclassified samples
    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Find misclassified
            mask = predicted != labels
            
            misclassified_images.extend(images[mask].cpu())
            misclassified_true.extend(labels[mask].cpu().numpy())
            misclassified_pred.extend(predicted[mask].cpu().numpy())
            
            if len(misclassified_images) >= num_samples:
                break
    
    # Limit to num_samples
    misclassified_images = misclassified_images[:num_samples]
    misclassified_true = misclassified_true[:num_samples]
    misclassified_pred = misclassified_pred[:num_samples]
    
    if len(misclassified_images) == 0:
        print("No misclassified samples found!")
        return None
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(misclassified_images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    if grid_size == 1:
        axes = np.array([[axes]])
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(misclassified_images):
            img = misclassified_images[idx].squeeze().numpy()
            true_label = misclassified_true[idx]
            pred_label = misclassified_pred[idx]
            
            # Denormalize
            img = img * 0.3081 + 0.1307
            img = np.clip(img, 0, 1)
            
            ax.imshow(img, cmap="gray")
            ax.set_title(f"True: {true_label} → Pred: {pred_label}", 
                        fontsize=9, color="red", fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")
    
    plt.suptitle(f"Misclassified Samples ({len(misclassified_images)} shown)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved misclassified samples to: {save_path}")
    
    return fig


def save_all_visualizations(
    model: nn.Module,
    history: Dict[str, List[float]],
    config: Config = DEFAULT_CONFIG
) -> None:
    """Generate and save all visualizations.
    
    Args:
        model: Trained neural network model.
        history: Training history dictionary.
        config: Configuration object.
    """
    device = get_device(config)
    _, _, test_loader = get_data_loaders(config, augment=False)
    
    # Evaluate to get predictions
    _, _, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    output_dir = config.output_dir
    
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=output_dir / "confusion_matrix.png"
    )
    plt.close()
    
    # Training curves
    plot_training_curves(
        history,
        save_path=output_dir / "training_curves.png"
    )
    plt.close()
    
    # Sample predictions
    plot_sample_predictions(
        model, test_loader, device,
        save_path=output_dir / "sample_predictions.png"
    )
    plt.close()
    
    # Misclassified samples
    plot_misclassified_samples(
        model, test_loader, device,
        save_path=output_dir / "misclassified_samples.png"
    )
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main() -> None:
    """Main entry point for generating visualizations."""
    import json
    from .train import load_model
    
    config = Config()
    device = get_device(config)
    
    print("=" * 60)
    print("MNIST Digit Recognition - Visualization")
    print("=" * 60)
    
    # Load model
    model_path = config.model_dir / "best_model.pth"
    if not model_path.exists():
        print(f"Error: No model found at {model_path}")
        print("Please run training first: python -m src.train")
        return
    
    model = load_model(model_path, device)
    
    # Try to load training history
    history_path = config.output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        print("Warning: No training history found. Skipping training curves.")
        history = None
    
    # Get test data
    _, _, test_loader = get_data_loaders(config, augment=False)
    
    # Generate visualizations
    _, _, test_preds, test_labels = evaluate_model(model, test_loader, device)
    
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=config.output_dir / "confusion_matrix.png"
    )
    plt.close()
    
    # Training curves (if history available)
    if history:
        plot_training_curves(
            history,
            save_path=config.output_dir / "training_curves.png"
        )
        plt.close()
    
    # Sample predictions
    plot_sample_predictions(
        model, test_loader, device,
        save_path=config.output_dir / "sample_predictions.png"
    )
    plt.close()
    
    # Misclassified samples
    plot_misclassified_samples(
        model, test_loader, device,
        save_path=config.output_dir / "misclassified_samples.png"
    )
    plt.close()
    
    print(f"\nAll visualizations saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
