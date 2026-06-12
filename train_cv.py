"""
Cross-validation training script for ChartCLIP Multi-Plot model

Runs 5-fold cross-validation with proper model saving and logging.
Each fold trains independently with its own W&B run.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import torch
from omegaconf import OmegaConf
import wandb

# Import trainers - dynamically select based on config
from train_chartclip import ChartCLIPTrainer
from data_loader_multiplot import create_multiplot_data_loaders_cv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _detect_model_type(config):
    """
    Detect which model type based on config
    
    Returns:
        str: 'chartclip', 'resnet', 'swin', 'convnext', or 'clip'
    """
    # Check if convnext_model exists in config
    if OmegaConf.select(config, "model.convnext_model"):
        return "convnext"
    
    # Check if swin_model exists in config
    if OmegaConf.select(config, "model.swin_model"):
        return "swin"
    
    # Check if chartclip_model exists
    if OmegaConf.select(config, "model.chartclip_model"):
        chartclip_model = config.model.chartclip_model
        if "Chart_CLIP" in chartclip_model or "chartclip" in chartclip_model.lower():
            return "chartclip"
        return "clip"
    
    # Check model name in experiment_name
    experiment_name = config.get('experiment_name', '').lower()
    if 'convnext' in experiment_name:
        return "convnext"
    elif 'swin' in experiment_name:
        return "swin"
    elif 'resnet' in experiment_name:
        return "resnet"
    elif 'chart' in experiment_name:
        return "chartclip"
    
    # Default to chartclip (most versatile)
    return "chartclip"


def run_fold(fold, config_path, model_name, csv_file, overrides=None, shuffle_panels=False):
    """
    Run training for a single fold
    
    Args:
        fold: Fold number (0-4)
        config_path: Path to base config file
        model_name: Name of model (for experiment tracking)
        csv_file: Path to CSV with fold columns
        overrides: Dict of config overrides
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Fold {fold}")
    logger.info(f"{'='*70}\n")
    
    # Load base config
    config = OmegaConf.load(config_path)
    
    # Resolve Hydra defaults if not already resolved
    if 'defaults' in config:
        from hydra import compose, initialize_config_dir
        from pathlib import Path
        import os
        
        # Get config directory
        config_dir = Path(config_path).parent.absolute()
        
        # Initialize Hydra and compose config
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            config = compose(config_name=Path(config_path).stem)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(config, key, value, merge=False)
            logger.info(f"Applied override: {key} = {value}")
    
    # Ensure image_size is present (default to 336 for ChartCLIP)
    if not OmegaConf.select(config, "data.image_size"):
        OmegaConf.update(config, "data.image_size", 336, merge=False)
        logger.info("Set default image_size = 336")
    
    # Disable struct mode to allow adding new keys
    OmegaConf.set_struct(config, False)
    
    # Update config for this fold
    config.data.csv_file = csv_file
    config.data.fold = fold  # Add fold number
    
    # Update output directories for this fold
    config.output_dir = f"outputs/{model_name}/fold_{fold}"
    config.checkpoint_dir = f"checkpoints/{model_name}/fold_{fold}"
    config.log_dir = f"logs/{model_name}/fold_{fold}"
    config.results_dir = f"results/{model_name}/fold_{fold}"
    
    # Update W&B config
    config.wandb.name = f"{model_name}_fold{fold}"
    config.wandb.tags.append(f"fold_{fold}")
    config.wandb.notes = f"5-fold CV - Fold {fold}/4"
    
    # Detect model type and create appropriate trainer
    model_type = _detect_model_type(config)
    logger.info(f"Detected model type: {model_type}")
    
    # Use ChartCLIPTrainer (works for ChartCLIP, CLIP, ResNet, Swin, ConvNeXt)
    trainer = ChartCLIPTrainer(config)
    
    # Need to modify data loader to use fold
    # Override the create_data_loaders method
    from data_loader_multiplot import create_multiplot_data_loaders_cv
    trainer.train_loader, trainer.val_loader, trainer.test_loader = \
        create_multiplot_data_loaders_cv(
            csv_file=csv_file,
            plot_dir=config.data.plot_dir,
            fold=fold,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            image_size=config.data.image_size,
            pin_memory=config.data.pin_memory,
            drop_last_train=config.data.drop_last,
            shuffle_panels=shuffle_panels
        )
    
    # Train
    trainer.train()
    
    # Get best val AUC
    best_auc = trainer.best_val_auc
    
    logger.info(f"\nFold {fold} Complete - Best Val AUC: {best_auc:.4f}\n")
    
    return best_auc


def run_cross_validation(config_path, model_name, csv_file='spiro_binary_labels.csv', n_folds=5,
                        plot_dir_override=None, batch_size_override=None, num_workers_override=None,
                        shuffle_panels=False):
    """
    Run complete cross-validation
    
    Args:
        config_path: Path to config file
        model_name: Name for experiment (e.g., 'CLIP_base', 'Chart_CLIP')
        csv_file: CSV file with fold columns
        n_folds: Number of folds
        plot_dir_override: Override plot directory from config
        batch_size_override: Override batch size from config
        num_workers_override: Override num workers from config
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting {n_folds}-Fold Cross-Validation")
    logger.info(f"Model: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'='*70}\n")
    
    # Check CSV has fold columns
    df = pd.read_csv(csv_file, sep=';')
    if 'fold_test' not in df.columns or 'fold_val' not in df.columns:
        raise ValueError(f"CSV {csv_file} missing fold columns. Run create_nested_folds.py first.")
    
    # Create override dict for parameters
    overrides = {}
    if plot_dir_override:
        overrides['data.plot_dir'] = plot_dir_override
        logger.info(f"Override: plot_dir = {plot_dir_override}")
    if batch_size_override:
        overrides['data.batch_size'] = batch_size_override
        logger.info(f"Override: batch_size = {batch_size_override}")
    if num_workers_override:
        overrides['data.num_workers'] = num_workers_override
        logger.info(f"Override: num_workers = {num_workers_override}")
    
    config_to_use = config_path
    
    # Run each fold
    fold_results = {}
    for fold in range(n_folds):
        try:
            best_auc = run_fold(fold, config_to_use, model_name, csv_file, overrides=overrides,
                                shuffle_panels=shuffle_panels)
            fold_results[fold] = best_auc
        except Exception as e:
            logger.error(f"Fold {fold} failed with error: {e}")
            import traceback
            traceback.print_exc()
            fold_results[fold] = None
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Cross-Validation Complete!")
    logger.info(f"{'='*70}")
    
    valid_results = [v for v in fold_results.values() if v is not None]
    
    if valid_results:
        mean_auc = sum(valid_results) / len(valid_results)
        logger.info(f"\nResults Summary:")
        for fold, auc in fold_results.items():
            status = f"{auc:.4f}" if auc is not None else "FAILED"
            logger.info(f"  Fold {fold}: {status}")
        logger.info(f"\nMean AUC: {mean_auc:.4f}")
        logger.info(f"Std AUC: {torch.tensor(valid_results).std().item():.4f}")
    else:
        logger.error("All folds failed!")
    
    # Save results
    results_file = Path(f"results/{model_name}/cv_results.txt")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"{model_name} - {n_folds}-Fold Cross-Validation Results\n")
        f.write("="*70 + "\n\n")
        for fold, auc in fold_results.items():
            f.write(f"Fold {fold}: {auc if auc else 'FAILED'}\n")
        if valid_results:
            f.write(f"\nMean AUC: {mean_auc:.4f}\n")
            f.write(f"Std AUC: {torch.tensor(valid_results).std().item():.4f}\n")
    
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation training")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for experiment tracking')
    parser.add_argument('--csv', type=str, default='spiro_binary_labels.csv',
                       help='CSV file with fold columns')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--plot_dir', type=str, default=None,
                       help='Override plot directory (for cluster)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Override num workers')
    parser.add_argument('--shuffle_panels', action='store_true', default=False,
                       help='Randomly permute panel order on every sample load (train/val/test)')

    args = parser.parse_args()
    
    # Override config if cluster arguments provided
    if args.plot_dir or args.batch_size or args.num_workers:
        logger.info("Overriding config with cluster arguments:")
        if args.plot_dir:
            logger.info(f"  plot_dir: {args.plot_dir}")
        if args.batch_size:
            logger.info(f"  batch_size: {args.batch_size}")
        if args.num_workers:
            logger.info(f"  num_workers: {args.num_workers}")
    
    run_cross_validation(
        config_path=args.config,
        model_name=args.model_name,
        csv_file=args.csv,
        n_folds=args.n_folds,
        plot_dir_override=args.plot_dir,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        shuffle_panels=args.shuffle_panels
    )
