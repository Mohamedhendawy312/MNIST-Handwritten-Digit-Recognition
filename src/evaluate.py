# -*- coding: utf-8 -*-
"""Evaluation module for MNIST digit recognition.

This module provides functions for model evaluation, computing metrics,
and generating classification reports.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from .config import Config, DEFAULT_CONFIG
from .utils import get_data_loaders, get_device


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate model on a dataset.
    
    Args:
        model: Neural network model.
        data_loader: Data loader for evaluation.
        device: Device for inference.
        criterion: Optional loss function. If None, loss is not computed.
    
    Returns:
        Tuple of (average_loss, accuracy, predictions, labels).
        If criterion is None, average_loss will be 0.0.
    """
    model.eval()
    running_loss = 0.0
    all_predictions: List[int] = []
    all_labels: List[int] = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = running_loss / len(data_loader) if criterion else 0.0
    accuracy = 100 * accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels


def get_metrics(
    y_true: List[int],
    y_pred: List[int]
) -> Tuple[float, float, float, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        Tuple of (accuracy, precision, recall, f1_score) as percentages.
    """
    accuracy = 100 * accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    return accuracy, precision * 100, recall * 100, f1 * 100


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        Confusion matrix as numpy array of shape (10, 10).
    """
    return confusion_matrix(y_true, y_pred)


def get_per_class_accuracy(y_true: List[int], y_pred: List[int]) -> List[float]:
    """Compute per-class accuracy.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        List of accuracy percentages for each class (0-9).
    """
    cm = get_confusion_matrix(y_true, y_pred)
    per_class_acc = []
    
    for i in range(10):
        if cm[i].sum() > 0:
            acc = 100 * cm[i, i] / cm[i].sum()
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    return per_class_acc


def print_classification_report(y_true: List[int], y_pred: List[int]) -> str:
    """Print and return sklearn classification report.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        Classification report string.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=[str(i) for i in range(10)],
        digits=4
    )
    print(report)
    return report


def main() -> None:
    """Main entry point for evaluation."""
    from .train import load_model
    
    config = Config()
    device = get_device(config)
    
    print("=" * 60)
    print("MNIST Digit Recognition - Evaluation")
    print("=" * 60)
    
    # Load model
    model_path = config.model_dir / "best_model.pth"
    if not model_path.exists():
        print(f"Error: No model found at {model_path}")
        print("Please run training first: python -m src.train")
        return
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)
    
    # Get data loaders
    _, val_loader, test_loader = get_data_loaders(config, augment=False)
    
    # Evaluate on validation set
    print("\n--- Validation Set ---")
    val_loss, val_acc, val_preds, val_labels = evaluate_model(
        model, val_loader, device, nn.CrossEntropyLoss()
    )
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.2f}%")
    
    accuracy, precision, recall, f1 = get_metrics(val_labels, val_preds)
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
    # Evaluate on test set
    print("\n--- Test Set ---")
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model, test_loader, device, nn.CrossEntropyLoss()
    )
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    accuracy, precision, recall, f1 = get_metrics(test_labels, test_preds)
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
    # Print detailed classification report
    print("\n--- Detailed Classification Report (Test Set) ---")
    print_classification_report(test_labels, test_preds)
    
    # Print per-class accuracy
    print("\n--- Per-Class Accuracy ---")
    per_class = get_per_class_accuracy(test_labels, test_preds)
    for digit, acc in enumerate(per_class):
        print(f"  Digit {digit}: {acc:.2f}%")


if __name__ == "__main__":
    main()
