"""
Create Combined ROC Curves and Confusion Matrices for Model Comparison

Generates:
1. Combined ROC curve with all 3 models (PH vs All)
2. Combined ROC curve with all 3 models (PAH vs CTEPH Subgroup)
3. Combined confusion matrices (PH vs All)
4. Combined confusion matrices (PAH vs CTEPH Subgroup)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_fold_predictions(logs_dir: Path, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions from a single fold's logs.
    
    Returns:
        labels: True labels
        predictions: Predicted probabilities
        sample_ids: Sample identifiers
    """
    fold_dir = logs_dir / f"fold_{fold}"
    csv_file = fold_dir / "test_predictions.csv"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        labels = df['true_label'].values.astype(int)
        predictions = df['predicted_probability'].values.astype(float)
        sample_ids = df['sample_id'].values
        return labels, predictions, sample_ids
    
    raise FileNotFoundError(f"Could not find test_predictions.csv in {fold_dir}")


def load_model_predictions(logs_dir: Path, n_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and aggregate predictions from all folds for a single model.
    
    Returns:
        all_labels: Aggregated true labels
        all_predictions: Aggregated predicted probabilities
    """
    all_labels = []
    all_predictions = []
    all_indices = []
    
    for fold in range(n_folds):
        try:
            labels, predictions, indices = load_fold_predictions(logs_dir, fold)
            all_labels.append(labels)
            all_predictions.append(predictions)
            all_indices.append(indices)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    if len(all_labels) == 0:
        raise ValueError(f"No prediction data found for {logs_dir}")
    
    # Concatenate all folds
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_indices = np.concatenate(all_indices)
    
    # Sort by indices to ensure alignment
    sort_order = np.argsort(all_indices)
    all_labels = all_labels[sort_order]
    all_predictions = all_predictions[sort_order]
    
    return all_labels, all_predictions


def plot_combined_roc_curves(models_data: Dict, title: str, output_path: Path):
    """
    Plot combined ROC curves for multiple models on the same plot.
    
    Args:
        models_data: Dict mapping model names to (y_true, y_pred_proba) tuples
        title: Plot title
        output_path: Where to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot ROC curve for each model
    for idx, (model_name, (y_true, y_pred_proba)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2.5, color=colors[idx % len(colors)],
                label=f'{model_name}')
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    
    # Formatting
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add text box with sample info
    n_samples = len(y_true)
    n_positive = np.sum(y_true)
    n_negative = n_samples - n_positive
    textstr = f'n = {n_samples}\nPositive: {n_positive}\nNegative: {n_negative}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Saved: {pdf_path}")


def plot_combined_confusion_matrices(models_data: Dict, title: str, output_path: Path):
    """
    Plot confusion matrices for multiple models in a grid.
    
    Args:
        models_data: Dict mapping model names to (y_true, y_pred_proba) tuples
        title: Overall figure title
        output_path: Where to save the figure
    """
    n_models = len(models_data)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # Plot confusion matrix for each model
    for idx, (model_name, (y_true, y_pred_proba)) in enumerate(models_data.items()):
        # Convert probabilities to predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for display (show percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percent = cm_normalized[i, j]
                row.append(f'{count}\n({percent:.1f}%)')
            annotations.append(row)
        
        # Plot heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   square=True, ax=axes[idx],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'},
                   vmin=0, vmax=cm.max())
        
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm) * 100
        
        axes[idx].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.1f}%', 
                           fontsize=13, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Saved: {pdf_path}")


def main():
    """Main execution function."""
    print("="*80)
    print("CREATING COMBINED ROC CURVES AND CONFUSION MATRICES")
    print("="*80)
    
    # Configuration
    logs_base = Path("logs")
    output_dir = Path("evaluation/combined_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations
    ph_vs_all_models = {
        "ConvNeXt-Tiny": "ConvNeXt_Baseline",
        "ResNet-50": "ResNet50_Baseline",
        "CLIP ViT-L": "CLIP_ViT_Large_Baseline"
    }

    subgroup_models = {
        "ConvNeXt-Tiny": "ConvNeXt_Baseline",
        "ResNet-50": "ResNet50_Baseline",
        "CLIP ViT-L": "CLIP_ViT_Large_Baseline"
    }
    
    n_folds = 5
    
    # ========================================================================
    # Process PH vs All Models
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING: PH vs All Models")
    print("="*80)
    
    ph_models_data = {}
    
    for model_name, job_dir in ph_vs_all_models.items():
        print(f"\nLoading {model_name} ({job_dir})...")
        logs_dir = logs_base / job_dir
        
        if not logs_dir.exists():
            print(f"  ⚠️  Directory not found: {logs_dir}")
            continue
        
        try:
            labels, predictions = load_model_predictions(logs_dir, n_folds)
            print(f"  ✓ Loaded {len(labels)} samples")
            
            # Compute AUROC
            fpr, tpr, _ = roc_curve(labels, predictions)
            roc_auc = auc(fpr, tpr)
            print(f"  ✓ AUROC: {roc_auc:.4f}")
            
            ph_models_data[model_name] = (labels, predictions)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    if len(ph_models_data) >= 2:
        print("\nGenerating PH vs All figures...")
        
        # ROC curves
        plot_combined_roc_curves(
            ph_models_data,
            "ROC Curves: PH vs All Classification",
            output_dir / "roc_curve_ph_vs_all.png"
        )
        
        # Confusion matrices
        plot_combined_confusion_matrices(
            ph_models_data,
            "Confusion Matrices: PH vs All Classification",
            output_dir / "confusion_matrices_ph_vs_all.png"
        )
    else:
        print("\n⚠️  Not enough models loaded for PH vs All comparison")
    
    # ========================================================================
    # Process Subgroup Models (PAH vs CTEPH)
    # ========================================================================
    print("\n" + "="*80)
    print("PROCESSING: PAH vs CTEPH Subgroup Models")
    print("="*80)
    
    subgroup_models_data = {}
    
    for model_name, job_dir in subgroup_models.items():
        print(f"\nLoading {model_name} ({job_dir})...")
        logs_dir = logs_base / job_dir
        
        if not logs_dir.exists():
            print(f"  ⚠️  Directory not found: {logs_dir}")
            continue
        
        try:
            labels, predictions = load_model_predictions(logs_dir, n_folds)
            print(f"  ✓ Loaded {len(labels)} samples")
            
            # Compute AUROC
            fpr, tpr, _ = roc_curve(labels, predictions)
            roc_auc = auc(fpr, tpr)
            print(f"  ✓ AUROC: {roc_auc:.4f}")
            
            subgroup_models_data[model_name] = (labels, predictions)
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    if len(subgroup_models_data) >= 2:
        print("\nGenerating PAH vs CTEPH subgroup figures...")
        
        # ROC curves
        plot_combined_roc_curves(
            subgroup_models_data,
            "ROC Curves: PAH vs CTEPH Subgroup Analysis",
            output_dir / "roc_curve_pah_vs_cteph_subgroup.png"
        )
        
        # Confusion matrices
        plot_combined_confusion_matrices(
            subgroup_models_data,
            "Confusion Matrices: PAH vs CTEPH Subgroup Analysis",
            output_dir / "confusion_matrices_pah_vs_cteph_subgroup.png"
        )
    else:
        print("\n⚠️  Not enough models loaded for subgroup comparison")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPH vs All Models: {len(ph_models_data)} loaded")
    for model_name in ph_models_data:
        y_true, y_pred = ph_models_data[model_name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print(f"  • {model_name:15s}: AUROC = {roc_auc:.4f}, n = {len(y_true)}")
    
    print(f"\nPAH vs CTEPH Subgroup Models: {len(subgroup_models_data)} loaded")
    for model_name in subgroup_models_data:
        y_true, y_pred = subgroup_models_data[model_name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print(f"  • {model_name:15s}: AUROC = {roc_auc:.4f}, n = {len(y_true)}")
    
    print(f"\n✓ All results saved to: {output_dir}")
    
    print("\nGenerated Files:")
    print("  • roc_curve_ph_vs_all.png")
    print("  • confusion_matrices_ph_vs_all.png")
    print("  • roc_curve_pah_vs_cteph_subgroup.png")
    print("  • confusion_matrices_pah_vs_cteph_subgroup.png")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
