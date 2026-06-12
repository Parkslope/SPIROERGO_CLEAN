"""
Visualization utilities for model evaluation

Includes:
- ROC curve plotting
- Confusion matrix plotting  
- Performance metrics visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import wandb
from pathlib import Path


def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """
    Plot ROC curve and return figure
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
        roc_auc: AUC score
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, roc_auc


def plot_confusion_matrix(y_true, y_pred, class_names=['No PH', 'PH'],
                         title="Confusion Matrix", save_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_path: Optional path to save figure
        normalize: Whether to normalize counts
        
    Returns:
        fig: Matplotlib figure
        cm: Confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = title + " (Normalized)"
        display_cm = cm_normalized
    else:
        fmt = 'd'
        display_cm = cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(display_cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True, ax=ax, 
                annot_kws={"size": 14, "weight": "bold"})
    
    # Styling
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add counts in cells if normalized
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]}',
                             ha="center", va="center", color="gray", 
                             fontsize=9)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, cm


def log_plots_to_wandb(y_true, y_pred, y_prob, prefix="test", epoch=None):
    """
    Log ROC curve and confusion matrix to W&B
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities for positive class
        prefix: Prefix for logging (e.g., 'test', 'val', 'fold_0')
        epoch: Optional epoch number
    """
    # Create plots
    roc_fig, roc_auc = plot_roc_curve(y_true, y_prob, 
                                       title=f"{prefix.upper()} ROC Curve")
    
    cm_fig, cm = plot_confusion_matrix(y_true, y_pred, 
                                       title=f"{prefix.upper()} Confusion Matrix")
    
    cm_norm_fig, cm_norm = plot_confusion_matrix(y_true, y_pred,
                                                  title=f"{prefix.upper()} Confusion Matrix",
                                                  normalize=True)
    
    # Convert figures to wandb Images
    log_dict = {
        f'{prefix}/roc_curve': wandb.Image(roc_fig),
        f'{prefix}/confusion_matrix': wandb.Image(cm_fig),
        f'{prefix}/confusion_matrix_normalized': wandb.Image(cm_norm_fig),
        f'{prefix}/roc_auc': roc_auc
    }
    
    if epoch is not None:
        log_dict['epoch'] = epoch
    
    # Log to wandb
    wandb.log(log_dict)
    
    # Close figures to free memory
    plt.close(roc_fig)
    plt.close(cm_fig)
    plt.close(cm_norm_fig)
    
    return roc_auc


def save_plots_locally(y_true, y_pred, y_prob, save_dir, prefix="test"):
    """
    Save ROC curve and confusion matrix locally
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class
        save_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ROC curve
    roc_path = save_dir / f"{prefix}_roc_curve.png"
    roc_fig, roc_auc = plot_roc_curve(y_true, y_prob,
                                       title=f"{prefix.upper()} ROC Curve",
                                       save_path=roc_path)
    plt.close(roc_fig)
    
    # Save confusion matrix
    cm_path = save_dir / f"{prefix}_confusion_matrix.png"
    cm_fig, cm = plot_confusion_matrix(y_true, y_pred,
                                       title=f"{prefix.upper()} Confusion Matrix",
                                       save_path=cm_path)
    plt.close(cm_fig)
    
    # Save normalized confusion matrix
    cm_norm_path = save_dir / f"{prefix}_confusion_matrix_normalized.png"
    cm_norm_fig, cm_norm = plot_confusion_matrix(y_true, y_pred,
                                                  title=f"{prefix.upper()} Confusion Matrix",
                                                  save_path=cm_norm_path,
                                                  normalize=True)
    plt.close(cm_norm_fig)
    
    return roc_auc
