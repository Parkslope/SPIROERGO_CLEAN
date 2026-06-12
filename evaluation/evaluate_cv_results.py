"""
Cross-Validation Evaluation Script

Evaluates model performance across all folds:
- ROC curves with confidence intervals
- Medical metrics (sensitivity, specificity, PPV, NPV, etc.)
- Confusion matrices
- Statistical analysis with standard deviations

Usage:
    python evaluation/evaluate_cv_results.py --results_dir results/ConvNeXt_Baseline
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score
)
from scipy import stats


def bootstrap_ci(y_true, y_pred_proba, n_bootstraps=1000, ci=95):
    """
    Compute confidence intervals for AUC using bootstrapping
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bootstraps: Number of bootstrap iterations
        ci: Confidence interval percentage
    
    Returns:
        mean_auc, lower_ci, upper_ci
    """
    np.random.seed(42)
    bootstrapped_aucs = []
    
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        if len(np.unique(y_true[indices])) < 2:
            # Need at least one positive and one negative sample
            continue
        
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))
    
    sorted_aucs = np.array(bootstrapped_aucs)
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile
    
    ci_lower = np.percentile(sorted_aucs, lower_percentile)
    ci_upper = np.percentile(sorted_aucs, upper_percentile)
    mean_auc = np.mean(sorted_aucs)
    
    return mean_auc, ci_lower, ci_upper


def calculate_medical_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate medical/clinical metrics
    
    Returns:
        Dictionary of metrics
    """
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Sensitivity (Recall, True Positive Rate)
    sensitivity = recall_score(y_true, y_pred)
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Positive Predictive Value (Precision)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # Likelihood Ratios
    lr_positive = sensitivity / (1 - specificity) if specificity < 1 else np.inf
    lr_negative = (1 - sensitivity) / specificity if specificity > 0 else np.inf
    
    # Youden's Index
    youden = sensitivity + specificity - 1
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap CI for AUC
    auc_mean, auc_ci_lower, auc_ci_upper = bootstrap_ci(y_true, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'lr_positive': lr_positive,
        'lr_negative': lr_negative,
        'youden_index': youden,
        'roc_auc': roc_auc,
        'auc_ci_lower': auc_ci_lower,
        'auc_ci_upper': auc_ci_upper,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def plot_roc_curve_with_ci(fold_results: List[Dict], output_path: Path):
    """
    Plot ROC curves for all folds with mean and confidence interval
    """
    plt.figure(figsize=(10, 8))
    
    # Plot individual fold ROC curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, result in enumerate(fold_results):
        y_true = result['y_true']
        y_pred_proba = result['y_pred_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                label=f'Fold {i} (AUC = {roc_auc:.3f})')
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', 
            label='Chance', alpha=.8)
    
    # Calculate mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b', 
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
            lw=2, alpha=.8)
    
    # Calculate std of TPR and plot confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                    alpha=.2, label='± 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Cross-Validation Results', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")


def plot_confusion_matrices(fold_results: List[Dict], output_path: Path):
    """
    Plot confusion matrices for all folds in a grid
    """
    n_folds = len(fold_results)
    n_cols = min(3, n_folds)
    n_rows = (n_folds + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(fold_results):
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   square=True, ax=axes[i],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_title(f'Fold {i} Confusion Matrix', fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_folds, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to: {output_path}")


def save_metrics_report(fold_metrics: List[Dict], output_path: Path):
    """
    Save comprehensive metrics report with statistics
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Convert to DataFrame for easy statistics
        df_metrics = pd.DataFrame(fold_metrics)
        
        f.write("SUMMARY STATISTICS (Mean ± Std Dev)\n")
        f.write("-"*80 + "\n\n")
        
        # Key metrics for medical evaluation
        key_metrics = [
            ('Accuracy', 'accuracy'),
            ('Balanced Accuracy', 'balanced_accuracy'),
            ('Sensitivity (Recall/TPR)', 'sensitivity'),
            ('Specificity (TNR)', 'specificity'),
            ('Positive Predictive Value (Precision)', 'ppv'),
            ('Negative Predictive Value', 'npv'),
            ('F1 Score', 'f1_score'),
            ('ROC AUC', 'roc_auc'),
            ("Youden's Index", 'youden_index'),
            ('Positive Likelihood Ratio', 'lr_positive'),
            ('Negative Likelihood Ratio', 'lr_negative'),
        ]
        
        for name, key in key_metrics:
            values = df_metrics[key].values
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample std
            
            # Format based on metric type
            if key in ['lr_positive', 'lr_negative']:
                # Some LRs might be inf
                finite_values = values[np.isfinite(values)]
                if len(finite_values) > 0:
                    mean = np.mean(finite_values)
                    std = np.std(finite_values, ddof=1)
                    f.write(f"{name:40s}: {mean:6.3f} ± {std:6.3f}\n")
                else:
                    f.write(f"{name:40s}: N/A (infinite values)\n")
            else:
                f.write(f"{name:40s}: {mean:6.3f} ± {std:6.3f}\n")
        
        # AUC with confidence interval
        f.write("\n")
        auc_values = df_metrics['roc_auc'].values
        auc_mean = np.mean(auc_values)
        auc_ci_lower = np.mean(df_metrics['auc_ci_lower'].values)
        auc_ci_upper = np.mean(df_metrics['auc_ci_upper'].values)
        f.write(f"ROC AUC with 95% CI                     : {auc_mean:.3f} [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]\n")
        
        # Confusion matrix totals
        f.write("\n" + "="*80 + "\n")
        f.write("CONFUSION MATRIX (Aggregated Across All Folds)\n")
        f.write("-"*80 + "\n\n")
        
        total_tp = df_metrics['tp'].sum()
        total_tn = df_metrics['tn'].sum()
        total_fp = df_metrics['fp'].sum()
        total_fn = df_metrics['fn'].sum()
        
        f.write(f"True Positives  (TP): {total_tp:4d}\n")
        f.write(f"True Negatives  (TN): {total_tn:4d}\n")
        f.write(f"False Positives (FP): {total_fp:4d}\n")
        f.write(f"False Negatives (FN): {total_fn:4d}\n")
        f.write(f"\nTotal Samples: {total_tp + total_tn + total_fp + total_fn}\n")
        
        # Per-fold details
        f.write("\n" + "="*80 + "\n")
        f.write("PER-FOLD RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for i, metrics in enumerate(fold_metrics):
            f.write(f"Fold {i}:\n")
            f.write(f"  Accuracy:       {metrics['accuracy']:.3f}\n")
            f.write(f"  Sensitivity:    {metrics['sensitivity']:.3f}\n")
            f.write(f"  Specificity:    {metrics['specificity']:.3f}\n")
            f.write(f"  PPV:            {metrics['ppv']:.3f}\n")
            f.write(f"  NPV:            {metrics['npv']:.3f}\n")
            f.write(f"  F1 Score:       {metrics['f1_score']:.3f}\n")
            f.write(f"  ROC AUC:        {metrics['roc_auc']:.3f} [{metrics['auc_ci_lower']:.3f}, {metrics['auc_ci_upper']:.3f}]\n")
            f.write(f"  TP/TN/FP/FN:    {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Metrics report saved to: {output_path}")


def load_fold_predictions(results_dir: Path) -> List[Dict]:
    """
    Load predictions from all folds
    
    Expected file structure:
    results_dir/
        fold_0/
            predictions.csv or test_predictions.csv
        fold_1/
            ...
    
    Returns:
        List of dicts with y_true, y_pred, y_pred_proba for each fold
    """
    fold_results = []
    fold_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        print(f"No fold directories found in {results_dir}")
        return []
    
    for fold_dir in fold_dirs:
        # Try different prediction file names
        pred_files = ['predictions.csv', 'test_predictions.csv', 'val_predictions.csv']
        pred_file = None
        
        for pf in pred_files:
            potential_file = fold_dir / pf
            if potential_file.exists():
                pred_file = potential_file
                break
        
        if pred_file is None:
            print(f"Warning: No predictions file found in {fold_dir}")
            continue
        
        # Load predictions
        df = pd.read_csv(pred_file)
        
        # Expected columns: true_label, predicted_label, predicted_proba
        # Handle different column names
        label_col = None
        pred_col = None
        proba_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'true' in col_lower or col_lower == 'label':
                label_col = col
            elif 'predicted_label' in col_lower or col_lower == 'prediction':
                pred_col = col
            elif 'proba' in col_lower or 'probability' in col_lower:
                proba_col = col
        
        if label_col is None or proba_col is None:
            print(f"Warning: Could not find required columns in {pred_file}")
            continue
        
        y_true = df[label_col].values
        y_pred_proba = df[proba_col].values
        
        # Get predicted labels (use provided or threshold at 0.5)
        if pred_col is not None:
            y_pred = df[pred_col].values
        else:
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        fold_results.append({
            'fold': fold_dir.name,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
    
    return fold_results


def evaluate_cv_results(results_dir: str, output_dir: str = None):
    """
    Main evaluation function
    
    Args:
        results_dir: Directory containing fold results
        output_dir: Where to save evaluation outputs (defaults to results_dir/evaluation)
    """
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / "evaluation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"EVALUATING CROSS-VALIDATION RESULTS")
    print(f"{'='*80}\n")
    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}\n")
    
    # Load predictions from all folds
    print("Loading predictions from folds...")
    fold_results = load_fold_predictions(results_dir)
    
    if not fold_results:
        print("ERROR: No fold results found!")
        return
    
    print(f"Found {len(fold_results)} folds\n")
    
    # Calculate metrics for each fold
    print("Calculating metrics...")
    fold_metrics = []
    for result in fold_results:
        metrics = calculate_medical_metrics(
            result['y_true'],
            result['y_pred'],
            result['y_pred_proba']
        )
        metrics['fold'] = result['fold']
        fold_metrics.append(metrics)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # ROC curve with CI
    plot_roc_curve_with_ci(
        fold_results,
        output_dir / "roc_curve_with_ci.png"
    )
    
    # Confusion matrices
    plot_confusion_matrices(
        fold_results,
        output_dir / "confusion_matrices.png"
    )
    
    # Save metrics report
    print("\nSaving metrics report...")
    save_metrics_report(
        fold_metrics,
        output_dir / "metrics_report.txt"
    )
    
    # Save metrics as JSON for programmatic access
    metrics_json = output_dir / "metrics_summary.json"
    with open(metrics_json, 'w') as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json}")
    
    # Save metrics as CSV
    metrics_csv = output_dir / "metrics_summary.csv"
    pd.DataFrame(fold_metrics).to_csv(metrics_csv, index=False)
    print(f"Metrics CSV saved to: {metrics_csv}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"All results saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cross-validation results with medical metrics"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing fold results (e.g., results/ConvNeXt_Baseline)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: results_dir/evaluation)'
    )
    
    args = parser.parse_args()
    
    evaluate_cv_results(args.results_dir, args.output_dir)
