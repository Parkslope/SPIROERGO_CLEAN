"""
Statistical Model Comparison using DeLong's Test for PAH vs. CTEPH Subgroup

Performs pairwise comparisons of AUROC between CLIP, ResNet-50, and Swin models
for the PAH vs. CTEPH subgroup analysis using aggregated out-of-fold predictions 
with Holm-Bonferroni correction.
"""

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def compute_midrank(x):
    """Computes midranks for DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast implementation of DeLong's algorithm for AUROC comparison."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes the p-value for two correlated ROC curves using DeLong's method.
    
    Args:
        ground_truth: Array of true binary labels
        predictions_one: Predicted probabilities from model 1
        predictions_two: Predicted probabilities from model 2
    
    Returns:
        p_value: Two-tailed p-value for the hypothesis that AUROC_1 = AUROC_2
    """
    preds = np.vstack([predictions_one, predictions_two])
    ground_truth = np.array(ground_truth)
    order = np.argsort(-ground_truth)
    label_1_count = np.sum(ground_truth)
    preds_sorted = preds[:, order]
    
    aucs, sigma = fast_delong(preds_sorted, label_1_count)
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    p_value = 2 * scipy.stats.norm.sf(z[0][0])
    return p_value, aucs


def load_fold_predictions(logs_dir: Path, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions from a single fold's logs.
    
    Returns:
        labels: True labels
        predictions: Predicted probabilities
        sample_ids: Sample identifiers (for alignment)
    """
    fold_dir = logs_dir / f"fold_{fold}"
    
    # Load from CSV format
    csv_file = fold_dir / "test_predictions.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        labels = df['true_label'].values.astype(int)
        predictions = df['predicted_probability'].values.astype(float)
        sample_ids = df['sample_id'].values
        return labels, predictions, sample_ids
    
    raise FileNotFoundError(f"Could not find test_predictions.csv in {fold_dir}")


def load_model_predictions(results_dir: Path, n_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
            labels, predictions, indices = load_fold_predictions(results_dir, fold)
            all_labels.append(labels)
            all_predictions.append(predictions)
            all_indices.append(indices)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
    
    if len(all_labels) == 0:
        raise ValueError(f"No prediction data found for {results_dir}")
    
    # Concatenate all folds
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    all_indices = np.concatenate(all_indices)
    
    # Sort by indices to ensure alignment
    sort_order = np.argsort(all_indices)
    all_labels = all_labels[sort_order]
    all_predictions = all_predictions[sort_order]
    
    return all_labels, all_predictions


def compute_auroc(y_true, y_pred):
    """Compute AUROC using sklearn."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return np.nan


def main():
    """Main execution function."""
    print("="*80)
    print("PAH vs. CTEPH SUBGROUP: DeLong's Test with Holm-Bonferroni Correction")
    print("="*80)
    
    # Configuration for SUBGROUP models
    logs_base = Path("logs")
    models_config = {
        "CLIP ViT-L": "CLIP_ViT_Large_Baseline",
        "ResNet-50": "ResNet50_Baseline",
        "ConvNeXt-Tiny": "ConvNeXt_Baseline"
    }
    
    n_folds = 5
    
    # Load predictions for each model
    print("\n1. Loading Model Predictions...")
    print("-" * 80)
    
    models_data = {}
    
    for model_name, job_dir in models_config.items():
        print(f"\nLoading {model_name} predictions from {job_dir}...")
        logs_dir = logs_base / job_dir
        
        if not logs_dir.exists():
            print(f"  ⚠️  Directory not found: {logs_dir}")
            continue
        
        try:
            # Load from fold logs
            labels, predictions = load_model_predictions(logs_dir, n_folds)
            print(f"  ✓ Loaded {len(labels)} samples")
            print(f"  ✓ AUROC: {compute_auroc(labels, predictions):.4f}")
            models_data[model_name] = (labels, predictions)
        except Exception as e:
            print(f"  ⚠️  Error loading predictions: {e}")
            continue
    
    if len(models_data) < 2:
        print("\n❌ ERROR: Need at least 2 models for comparison")
        return
    
    # Verify label alignment
    print("\n2. Verifying Label Alignment...")
    print("-" * 80)
    
    first_model = list(models_data.keys())[0]
    y_true = models_data[first_model][0]
    
    aligned = True
    for model_name in models_data:
        if not np.array_equal(models_data[model_name][0], y_true):
            print(f"  ⚠️  Warning: {model_name} labels do not match!")
            aligned = False
    
    if aligned:
        print(f"  ✓ All models have aligned labels (n={len(y_true)})")
        print(f"  ✓ Positive samples (CTEPH): {np.sum(y_true)} ({100*np.mean(y_true):.1f}%)")
        print(f"  ✓ Negative samples (PAH): {len(y_true) - np.sum(y_true)} ({100*(1-np.mean(y_true)):.1f}%)")
    else:
        print("  ❌ ERROR: Label mismatch between models!")
        return
    
    # Extract predictions
    models_predictions = {name: data[1] for name, data in models_data.items()}
    
    # Compute AUROCs
    print("\n3. Individual Model Performance...")
    print("-" * 80)
    
    aurocs = {}
    for model_name, predictions in models_predictions.items():
        auroc = compute_auroc(y_true, predictions)
        aurocs[model_name] = auroc
        print(f"  {model_name:15s}: AUROC = {auroc:.4f}")
    
    # Pairwise comparisons
    print("\n4. Pairwise DeLong Tests...")
    print("-" * 80)
    
    model_names = list(models_predictions.keys())
    comparisons = []
    raw_p_values = []
    auc_differences = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            comparisons.append((m1, m2))
            
            p_value, aucs = delong_roc_test(
                y_true,
                models_predictions[m1],
                models_predictions[m2]
            )
            
            raw_p_values.append(p_value)
            auc_diff = aurocs[m1] - aurocs[m2]
            auc_differences.append(auc_diff)
            
            print(f"\n  {m1} vs {m2}:")
            print(f"    AUROC difference: {auc_diff:+.4f}")
            print(f"    Raw p-value: {p_value:.6f}")
    
    # Multiple comparison correction
    print("\n5. Multiple Comparison Correction (Holm-Bonferroni)...")
    print("-" * 80)
    
    if len(raw_p_values) > 1:
        reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            raw_p_values, alpha=0.05, method='holm'
        )
        
        print(f"\n  Number of comparisons: {len(comparisons)}")
        print(f"  Family-wise error rate (FWER): α = 0.05")
        print(f"  Correction method: Holm-Bonferroni")
        
        print("\n  Results:")
        print("  " + "="*76)
        print(f"  {'Comparison':<25} {'ΔAUROC':<12} {'Raw p':<12} {'Adj. p':<12} {'Significant'}")
        print("  " + "-"*76)
        
        for i, (m1, m2) in enumerate(comparisons):
            status = "✓ YES" if reject[i] else "✗ NO"
            print(f"  {m1:>12} vs {m2:<10} {auc_differences[i]:>+10.4f}  "
                  f"{raw_p_values[i]:>10.6f}  {p_corrected[i]:>10.6f}  {status}")
        print("  " + "="*76)
    else:
        print("  Only one comparison - no correction needed")
        print(f"  {comparisons[0][0]} vs {comparisons[0][1]}: p = {raw_p_values[0]:.6f}")
        reject = [raw_p_values[0] < 0.05]
        p_corrected = raw_p_values
    
    # Summary
    print("\n6. Summary...")
    print("-" * 80)
    
    print("\n  Model Ranking (by AUROC):")
    sorted_models = sorted(aurocs.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, auroc) in enumerate(sorted_models, 1):
        print(f"    {rank}. {model_name:15s}: {auroc:.4f}")
    
    print("\n  Significant Differences (α=0.05, Holm-Bonferroni corrected):")
    sig_count = sum(reject)
    if sig_count > 0:
        for i, (m1, m2) in enumerate(comparisons):
            if reject[i]:
                better = m1 if auc_differences[i] > 0 else m2
                print(f"    • {m1} vs {m2}: {better} is significantly better (p={p_corrected[i]:.4f})")
    else:
        print("    • No significant differences detected")
    
    # Save results
    output_file = Path("evaluation/results/subgroup_pah_cteph_delong.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "analysis": "PAH vs CTEPH Subgroup",
        "n_samples": int(len(y_true)),
        "n_positive_cteph": int(np.sum(y_true)),
        "n_negative_pah": int(len(y_true) - np.sum(y_true)),
        "aurocs": aurocs,
        "comparisons": [{"model1": m1, "model2": m2, "auc_diff": float(d), 
                        "raw_p": float(p), "adjusted_p": float(pc), "significant": bool(r)}
                       for (m1, m2), d, p, pc, r in zip(comparisons, auc_differences, 
                                                         raw_p_values, p_corrected, reject)],
        "alpha": 0.05,
        "correction_method": "holm"
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
