"""
Attention Weight Analysis for ResNet-50 Multi-Panel Aggregator - PAH vs. CTEPH Subgroup

Extracts CLS token attention to each of the 9 panels and creates:
A) Panel attention summary heatmaps (PAH vs CTEPH)
B) Example attention overlay visualizations

Modified for PAH vs. CTEPH subgroup analysis.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_chartclip import MultiPlotResNet, MultiPlotConvNeXt
from data_loader_multiplot import MultiPlotDataset
from torch.utils.data import DataLoader


def extract_attention_with_hooks(model: nn.Module, panel_images: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Extract attention weights using similarity-based approach
    
    Returns:
        attention: [B, 9] - average attention from CLS to each of 9 panels
    """
    model.eval()
    
    with torch.no_grad():
        B, num_panels, C, H, W = panel_images.shape
        
        # Encode panels
        panels_flat = panel_images.view(B * num_panels, C, H, W)
        panel_embeddings = model.encoder(panels_flat)  # [B*9, D]
        D = panel_embeddings.size(-1)
        panel_embeddings = panel_embeddings.view(B, num_panels, D)  # [B, 9, D]
        
        # Add position embeddings
        panel_embeddings = model.position_embedding(panel_embeddings)
        
        # Get final CLS output
        cls_output = model.aggregator(panel_embeddings)  # [B, D]
        
        # Compute attention as normalized similarity
        # between final CLS representation and original panel embeddings
        similarities = torch.matmul(
            cls_output.unsqueeze(1),  # [B, 1, D]
            panel_embeddings.transpose(1, 2)  # [B, D, 9]
        ).squeeze(1)  # [B, 9]
        
        # Normalize to [0, 1]
        attention = torch.softmax(similarities, dim=1)
    
    return attention.cpu().numpy()


def compute_average_attention_per_class(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Compute average attention weights per class
    
    Returns:
        results: dict with keys 'pah', 'cteph', 'correct', 'incorrect'
                each containing average attention across 9 panels
    """
    model.eval()
    
    attention_pah = []  # Class 0
    attention_cteph = []  # Class 1
    attention_correct = []
    attention_incorrect = []
    
    with torch.no_grad():
        for panel_images, labels, _ in tqdm(test_loader, desc="Computing attention"):
            panel_images = panel_images.to(device)
            labels = labels.cpu().numpy()
            
            # Get predictions
            logits = model(panel_images)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Extract attention
            attention = extract_attention_with_hooks(model, panel_images, device)  # [B, 9]
            
            # Separate by class
            for i in range(len(labels)):
                label = labels[i]
                pred = preds[i]
                att = attention[i]
                
                if label == 1:
                    attention_cteph.append(att)
                else:
                    attention_pah.append(att)
                
                if label == pred:
                    attention_correct.append(att)
                else:
                    attention_incorrect.append(att)
    
    return {
        'pah': np.array(attention_pah),
        'cteph': np.array(attention_cteph),
        'correct': np.array(attention_correct),
        'incorrect': np.array(attention_incorrect)
    }


def plot_attention_heatmap(attention_matrix: np.ndarray, title: str, output_path: Path):
    """
    Plot 3x3 attention heatmap aligned with Wasserman panel grid
    
    Args:
        attention_matrix: [9] array of attention weights
        title: plot title
        output_path: where to save the figure
    """
    # Reshape to 3x3 grid
    attention_grid = attention_matrix.reshape(3, 3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot heatmap
    sns.heatmap(
        attention_grid,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=attention_matrix.max() * 1.1,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=ax
    )
    
    # Labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Panel labels (1-9)
    for i in range(3):
        for j in range(3):
            panel_num = i * 3 + j + 1
            ax.text(j + 0.5, i + 0.15, f'Panel {panel_num}', 
                   ha='center', va='top', fontsize=9, color='black', weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_comparison_heatmaps(attention_pah: np.ndarray, attention_cteph: np.ndarray, output_path: Path):
    """
    Plot side-by-side comparison of PAH vs CTEPH attention
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PAH cases (Class 0)
    attention_grid_pah = attention_pah.reshape(3, 3)
    sns.heatmap(
        attention_grid_pah,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=max(attention_pah.max(), attention_cteph.max()) * 1.1,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=axes[0]
    )
    axes[0].set_title('Pulmonary Arterial Hypertension (PAH)\nClass 0', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Column', fontsize=12)
    axes[0].set_ylabel('Row', fontsize=12)
    
    # CTEPH cases (Class 1)
    attention_grid_cteph = attention_cteph.reshape(3, 3)
    sns.heatmap(
        attention_grid_cteph,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=max(attention_pah.max(), attention_cteph.max()) * 1.1,
        square=True,
        linewidths=1,
        linecolor='gray',
        ax=axes[1]
    )
    axes[1].set_title('Chronic Thromboembolic PH (CTEPH)\nClass 1', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Column', fontsize=12)
    axes[1].set_ylabel('Row', fontsize=12)
    
    # Add panel numbers
    for ax in axes:
        for i in range(3):
            for j in range(3):
                panel_num = i * 3 + j + 1
                ax.text(j + 0.5, i + 0.15, f'P{panel_num}', 
                       ha='center', va='top', fontsize=9, color='black', weight='bold')
    
    plt.suptitle('Average Panel Attention: PAH vs CTEPH', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_attention_overlay(
    plot_dir: Path,
    sample_id: str,
    attention_weights: np.ndarray,
    output_path: Path
):
    """
    Create visualization with attention overlay on the 9-panel grid
    
    Args:
        plot_dir: directory containing plots
        sample_id: sample identifier
        attention_weights: [9] array
        output_path: where to save
    """
    sample_dir = plot_dir / sample_id
    
    if not sample_dir.exists():
        print(f"Warning: Sample directory not found: {sample_dir}")
        return
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Sample: {sample_id}\nPanel Attention Weights', fontsize=16, fontweight='bold')
    
    # Normalize attention for color mapping
    att_normalized = attention_weights / attention_weights.max()
    
    for idx in range(9):
        i, j = idx // 3, idx % 3
        ax = axes[i, j]
        
        # Load plot image
        plot_path = sample_dir / f"plot_{idx+1:02d}.png"
        
        if plot_path.exists():
            img = Image.open(plot_path)
            ax.imshow(img, alpha=0.7)
            
            # Add colored overlay based on attention
            attention_value = att_normalized[idx]
            overlay = np.ones((*img.size[::-1], 3))
            overlay[:, :, 0] = 1.0  # Red channel
            overlay[:, :, 1] = 1.0 - attention_value  # Less green = more attention
            overlay[:, :, 2] = 1.0 - attention_value  # Less blue = more attention
            
            ax.imshow(overlay, alpha=attention_value * 0.4)
        
        # Title with attention weight
        ax.set_title(f'Panel {idx+1}\nAttention: {attention_weights[idx]:.3f}', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Created overlay: {output_path}")


def load_model_checkpoint(checkpoint_path: str, device: torch.device, model_type: str = "resnet") -> nn.Module:
    """Load trained model from checkpoint"""
    if model_type == "convnext":
        model = MultiPlotConvNeXt(
            convnext_model="facebook/convnext-tiny-224",
            pretrained=False,
            freeze_encoder=False,
            projection_dim=None,
            num_panels=9,
            num_heads=8,
            num_transformer_layers=3,
            dropout=0.15,
            mlp_hidden_dim=256,
            mlp_ratio=4.0
        )
    else:  # resnet
        model = MultiPlotResNet(
            resnet_model="resnet50",
            pretrained=False,
            freeze_encoder=False,
            projection_dim=None,
            num_panels=9,
            num_heads=8,
            num_transformer_layers=3,
            dropout=0.1,
            mlp_hidden_dim=256,
            mlp_ratio=4.0
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def create_test_loader(csv_file: str, plot_dir: str, fold: int, batch_size: int = 16) -> DataLoader:
    """Create test data loader for a specific fold"""
    import tempfile
    
    df = pd.read_csv(csv_file, sep=';')
    test_df = df[df['fold_test'] == fold].copy()
    test_df['split'] = 'test'
    
    temp_dir = Path(tempfile.gettempdir()) / "attention_eval_subgroup"
    temp_dir.mkdir(exist_ok=True)
    test_csv = temp_dir / f"fold_{fold}_test.csv"
    test_df.to_csv(test_csv, sep=';', index=False)
    
    test_dataset = MultiPlotDataset(
        csv_file=str(test_csv),
        plot_dir=plot_dir,
        split='test',
        image_size=336,
        transform=None,
        check_files=True,
        plot_dropout_prob=0.0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_df


def main():
    """Main analysis function"""
    import argparse
    parser = argparse.ArgumentParser(description="Subgroup (PAH vs CTEPH) panel attention analysis")
    parser.add_argument("--checkpoint_dir", default="checkpoints/ConvNeXt_Baseline")
    parser.add_argument("--csv_file",       default="newV2_subgroup_classified.csv")
    parser.add_argument("--plot_dir",       required=True, help="Directory of preprocessed plot PNGs")
    parser.add_argument("--output_dir",     default="evaluation/results/subgroup_pah_cteph_attention_convnext")
    parser.add_argument("--model_type",     default="convnext", choices=["convnext", "resnet"])
    parser.add_argument("--n_folds",        type=int, default=5)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--n_examples",     type=int, default=50)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    csv_file       = args.csv_file
    plot_dir       = Path(args.plot_dir)
    output_dir     = Path(args.output_dir)
    model_type     = args.model_type
    n_folds        = args.n_folds
    batch_size     = args.batch_size
    n_examples     = args.n_examples

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nAttention Weight Analysis for ConvNeXt-Tiny - PAH vs. CTEPH Subgroup")
    print(f"=" * 80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"CSV file: {csv_file}")
    print(f"Plot directory: {plot_dir}")
    print(f"=" * 80)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heatmaps").mkdir(exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    
    # Aggregate attention across all folds
    all_attention_pah = []
    all_attention_cteph = []
    all_attention_correct = []
    all_attention_incorrect = []
    example_samples = []
    
    # Process each fold
    for fold in range(n_folds):
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold}")
        print(f"{'='*80}")
        
        fold_checkpoint = checkpoint_dir / f"fold_{fold}" / "best_model.pth"
        
        if not fold_checkpoint.exists():
            print(f"⚠️ Checkpoint not found: {fold_checkpoint}")
            continue
        
        # Load model
        print(f"Loading checkpoint...")
        model = load_model_checkpoint(str(fold_checkpoint), device, model_type=model_type)
        print(f"✓ Model loaded")
        
        # Create test loader
        print(f"Creating test data loader...")
        test_loader, test_df = create_test_loader(csv_file, plot_dir, fold, batch_size)
        print(f"✓ Test loader created ({len(test_df)} samples)")
        
        # Compute attention per class
        print(f"Computing attention weights...")
        attention_results = compute_average_attention_per_class(model, test_loader, device)
        
        all_attention_pah.append(attention_results['pah'])
        all_attention_cteph.append(attention_results['cteph'])
        all_attention_correct.append(attention_results['correct'])
        all_attention_incorrect.append(attention_results['incorrect'])
        
        print(f"  PAH samples: {len(attention_results['pah'])}")
        print(f"  CTEPH samples: {len(attention_results['cteph'])}")
        
        # Collect example samples for overlays
        if fold == 0:  # Only from first fold
            # Get some correctly classified samples
            with torch.no_grad():
                for panel_images, labels, filenames in test_loader:
                    panel_images = panel_images.to(device)
                    logits = model(panel_images)
                    probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    labels_np = labels.numpy()
                    
                    # Extract attention
                    attention = extract_attention_with_hooks(model, panel_images, device)
                    
                    for i in range(len(labels)):
                        if labels_np[i] == preds[i]:  # Correct prediction
                            example_samples.append({
                                'filename': filenames[i],
                                'label': labels_np[i],
                                'prob': probs[i],
                                'attention': attention[i]
                            })
                    
                    if len(example_samples) >= n_examples:
                        break
        
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    all_attention_pah = np.concatenate(all_attention_pah, axis=0)
    all_attention_cteph = np.concatenate(all_attention_cteph, axis=0)
    all_attention_correct = np.concatenate(all_attention_correct, axis=0)
    all_attention_incorrect = np.concatenate(all_attention_incorrect, axis=0)
    
    # Compute means
    mean_attention_pah = all_attention_pah.mean(axis=0)
    mean_attention_cteph = all_attention_cteph.mean(axis=0)
    mean_attention_correct = all_attention_correct.mean(axis=0)
    mean_attention_incorrect = all_attention_incorrect.mean(axis=0)
    
    print(f"\nAverage attention weights:")
    print(f"  PAH (n={len(all_attention_pah)}): {mean_attention_pah}")
    print(f"  CTEPH (n={len(all_attention_cteph)}): {mean_attention_cteph}")
    
    # Save numerical results
    results_dict = {
        'mean_attention_pah': mean_attention_pah.tolist(),
        'mean_attention_cteph': mean_attention_cteph.tolist(),
        'mean_attention_correct': mean_attention_correct.tolist(),
        'mean_attention_incorrect': mean_attention_incorrect.tolist(),
        'n_pah': len(all_attention_pah),
        'n_cteph': len(all_attention_cteph)
    }
    
    with open(output_dir / "attention_summary.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Saved: {output_dir / 'attention_summary.json'}")
    
    # Create heatmaps
    print(f"\nGenerating heatmaps...")
    
    # Individual heatmaps
    plot_attention_heatmap(
        mean_attention_pah,
        "Average Panel Attention: PAH Cases (Class 0)",
        output_dir / "heatmaps" / "attention_pah.png"
    )
    
    plot_attention_heatmap(
        mean_attention_cteph,
        "Average Panel Attention: CTEPH Cases (Class 1)",
        output_dir / "heatmaps" / "attention_cteph.png"
    )
    
    # Comparison heatmap
    plot_comparison_heatmaps(
        mean_attention_pah,
        mean_attention_cteph,
        output_dir / "heatmaps" / "attention_pah_vs_cteph_comparison.png"
    )
    
    # Generate example overlays
    print(f"\nGenerating example overlays...")
    for i, example in enumerate(example_samples[:n_examples]):
        sample_id = Path(example['filename']).stem
        label_str = "CTEPH" if example['label'] == 1 else "PAH"
        create_attention_overlay(
            plot_dir,
            sample_id,
            example['attention'],
            output_dir / "overlays" / f"example_{i+1}_{label_str}_{sample_id}.png"
        )
    
    print(f"\n{'='*80}")
    print(f"PAH vs. CTEPH Subgroup Attention Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
