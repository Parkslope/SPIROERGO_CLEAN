"""
Data Loader for Multi-Plot ChartCLIP Model

Loads 9 preprocessed plot images per sample from a user-specified directory.
Each sample has a folder with plot_01.png through plot_09.png
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class MultiPlotDataset(Dataset):
    """
    Dataset for loading 9 preprocessed plot images per sample
    
    Directory structure:
    plot_dir/
        sample_1/
            plot_01.png
            plot_02.png
            ...
            plot_09.png
        sample_2/
            ...
    """
    def __init__(
        self,
        csv_file: str,
        plot_dir: str,
        split: str = 'train',
        image_size: int = 336,  # ChartCLIP uses 336x336
        transform: Optional[transforms.Compose] = None,
        normalize_type: str = 'clip',  # 'clip', 'imagenet', or 'none'
        check_files: bool = True,
        plot_dropout_prob: float = 0.0,
        shuffle_panels: bool = False
    ):
        """
        Args:
            csv_file: Path to CSV with columns [filename, label, split]
            plot_dir: Path to preprocessed plots directory
            split: 'train', 'val', or 'test'
            image_size: Target size for plots (default 336 for ChartCLIP)
            transform: Optional additional transforms
            normalize_type: Normalization type - 'clip', 'imagenet', or 'none'
                          - 'clip': CLIP normalization for ChartCLIP/CLIP models
                          - 'imagenet': ImageNet normalization for ResNet/ConvNeXt/Swin
                          - 'none': No normalization (keep [0,1] range)
            check_files: Check if all plot files exist on initialization
            plot_dropout_prob: Probability of dropping each plot during training (0.0-1.0).
                              Only applied during training split. Dropped plots are replaced with black images.
            shuffle_panels: Randomly permute the order of the 9 panels each time a sample is loaded.
                           Applied on all splits (train/val/test) so the model is always tested without
                           positional cues. Use to probe whether the transformer learns position-invariant
                           features.
        """
        self.plot_dir = Path(plot_dir)
        self.split = split
        self.image_size = image_size
        self.num_plots = 9
        
        # Load CSV (handle both comma and semicolon delimiters)
        try:
            df = pd.read_csv(csv_file, sep=';')
        except:
            df = pd.read_csv(csv_file)
        
        # Rename columns if needed
        if 'binary_class' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'binary_class': 'label'})
        
        # Filter by split
        if 'split' in df.columns:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            raise ValueError("CSV must have 'split' column")
        
        logger.info(f"Loaded {len(self.df)} samples for {split} split")
        
        # Filter available samples (check if folders exist)
        if check_files:
            self.df = self.filter_available_samples()
            logger.info(f"Found {len(self.df)} samples with all 9 plots")
        
        # Base transform: resize and convert to tensor
        base_transform = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ]
        
        # Add normalization based on type
        if normalize_type == 'clip':
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            base_transform.append(transforms.Normalize(mean=clip_mean, std=clip_std))
            logger.info("Using CLIP normalization")
        elif normalize_type == 'imagenet':
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            base_transform.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std))
            logger.info("Using ImageNet normalization")
        elif normalize_type == 'none':
            logger.info("No normalization applied (values in [0,1])")
        else:
            raise ValueError(f"Unknown normalize_type: {normalize_type}. Use 'clip', 'imagenet', or 'none'")
        
        self.base_transform = transforms.Compose(base_transform)
        self.additional_transform = transform
        
        # Plot dropout (only for training)
        self.plot_dropout_prob = plot_dropout_prob if split == 'train' else 0.0
        if self.plot_dropout_prob > 0:
            logger.info(f"Plot dropout enabled: {self.plot_dropout_prob:.2f} probability per plot")

        # Panel shuffling (applied on all splits when enabled)
        self.shuffle_panels = shuffle_panels
        if self.shuffle_panels:
            logger.info("Panel shuffling enabled: panel order randomized on every sample load")
        
    def filter_available_samples(self) -> pd.DataFrame:
        """Filter samples that have all 9 plot files"""
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            filename = row['filename']
            # Remove .pdf extension if present
            sample_name = Path(filename).stem
            sample_dir = self.plot_dir / sample_name
            
            if not sample_dir.exists():
                continue
            
            # Check all 9 plots exist
            all_plots_exist = all(
                (sample_dir / f"plot_{i:02d}.png").exists()
                for i in range(1, 10)
            )
            
            if all_plots_exist:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            logger.warning(f"No valid samples found in {self.plot_dir}")
        
        return self.df.loc[valid_indices].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def load_plot_image(self, plot_path: Path) -> Image.Image:
        """Load a single plot image"""
        try:
            img = Image.open(plot_path).convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Error loading {plot_path}: {e}")
            # Return white image as fallback
            return Image.new('RGB', (self.image_size, self.image_size), color='white')
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            plots: [num_plots, C, H, W] - 9 plot images
            label: int - binary label (0 or 1)
            filename: str - sample identifier
        """
        row = self.df.iloc[idx]
        filename = row['filename']
        label = int(row['label'])
        
        # Get sample directory
        sample_name = Path(filename).stem
        sample_dir = self.plot_dir / sample_name
        
        # Load all 9 plots
        plot_tensors = []
        for i in range(1, 10):
            plot_path = sample_dir / f"plot_{i:02d}.png"
            
            # Load image
            img = self.load_plot_image(plot_path)
            
            # Apply transforms
            if self.additional_transform is not None:
                img = self.additional_transform(img)
            
            img_tensor = self.base_transform(img)
            plot_tensors.append(img_tensor)
        
        # Stack into [num_plots, C, H, W]
        plots = torch.stack(plot_tensors, dim=0)

        # Randomly permute panel order (applied on all splits when enabled)
        if self.shuffle_panels:
            perm = torch.randperm(self.num_plots)
            plots = plots[perm]

        # Apply plot dropout (only during training)
        if self.plot_dropout_prob > 0:
            # Randomly drop plots by replacing with zeros (black image)
            dropout_mask = torch.rand(self.num_plots) > self.plot_dropout_prob
            # Ensure at least one plot remains
            if not dropout_mask.any():
                # Keep a random plot
                random_idx = torch.randint(0, self.num_plots, (1,)).item()
                dropout_mask[random_idx] = True
            
            # Zero out dropped plots
            plots = plots * dropout_mask.view(-1, 1, 1, 1)
        
        return plots, label, filename


def get_train_transforms(image_size: int = 336) -> transforms.Compose:
    """
    Training augmentations
    Note: These are applied BEFORE the base transform (resize + normalize)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=5),
    ])


def get_val_test_transforms() -> Optional[transforms.Compose]:
    """
    Validation/test transforms (typically no augmentation)
    """
    return None


def create_multiplot_data_loaders_cv(
    csv_file: str,
    plot_dir: str,
    fold: int,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 336,
    pin_memory: bool = True,
    drop_last_train: bool = True,
    plot_dropout_prob: float = 0.0,
    shuffle_panels: bool = False
):
    """
    Create data loaders for cross-validation based on fold columns
    
    Args:
        csv_file: Path to CSV with fold_test and fold_val columns
        plot_dir: Directory containing preprocessed plots
        fold: Fold number (0-4)
        plot_dropout_prob: Probability of dropping each plot during training (0.0-1.0)
        shuffle_panels: Randomly permute panel order on every sample load (all splits)
        Other args same as create_multiplot_data_loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    df = pd.read_csv(csv_file, sep=';')
    
    # Split based on fold columns
    # Train: samples where fold_test != fold AND fold_val != fold
    # Val: samples where fold_val == fold
    # Test: samples where fold_test == fold
    train_df = df[(df['fold_test'] != fold) & (df['fold_val'] != fold)]
    val_df = df[df['fold_val'] == fold]
    test_df = df[df['fold_test'] == fold]
    
    print(f"Fold {fold} - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save filtered dataframes to temp CSVs
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "cv_folds"
    temp_dir.mkdir(exist_ok=True)
    
    train_csv = temp_dir / f"fold_{fold}_train.csv"
    val_csv = temp_dir / f"fold_{fold}_val.csv"
    test_csv = temp_dir / f"fold_{fold}_test.csv"
    
    train_df.to_csv(train_csv, sep=';', index=False)
    val_df.to_csv(val_csv, sep=';', index=False)
    test_df.to_csv(test_csv, sep=';', index=False)
    
    # Create datasets (they need split column)
    # Add split column to each
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    train_df.to_csv(train_csv, sep=';', index=False)
    val_df.to_csv(val_csv, sep=';', index=False)
    test_df.to_csv(test_csv, sep=';', index=False)
    
    train_dataset = MultiPlotDataset(
        csv_file=str(train_csv), plot_dir=plot_dir, split='train',
        image_size=image_size, transform=get_train_transforms(image_size),
        normalize_type='imagenet', check_files=True, plot_dropout_prob=plot_dropout_prob,
        shuffle_panels=shuffle_panels
    )
    val_dataset = MultiPlotDataset(
        csv_file=str(val_csv), plot_dir=plot_dir, split='val',
        image_size=image_size, transform=get_val_test_transforms(),
        normalize_type='imagenet', check_files=True, plot_dropout_prob=0.0,
        shuffle_panels=shuffle_panels
    )
    test_dataset = MultiPlotDataset(
        csv_file=str(test_csv), plot_dir=plot_dir, split='test',
        image_size=image_size, transform=get_val_test_transforms(),
        normalize_type='imagenet', check_files=True, plot_dropout_prob=0.0,
        shuffle_panels=shuffle_panels
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def create_multiplot_data_loaders(
    csv_file: str,
    plot_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 336,
    pin_memory: bool = True,
    drop_last_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        csv_file: Path to CSV file with labels and splits
        plot_dir: Path to preprocessed plots directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        image_size: Image size (default 336 for ChartCLIP)
        pin_memory: Pin memory for faster GPU transfer
        drop_last_train: Drop last incomplete batch in training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Training dataset with augmentation
    train_dataset = MultiPlotDataset(
        csv_file=csv_file,
        plot_dir=plot_dir,
        split='train',
        image_size=image_size,
        transform=get_train_transforms(image_size),
        normalize_type='imagenet',  # Default to ImageNet for ResNet/ConvNeXt/Swin
        check_files=True
    )
    
    # Validation dataset (no augmentation)
    val_dataset = MultiPlotDataset(
        csv_file=csv_file,
        plot_dir=plot_dir,
        split='val',
        image_size=image_size,
        transform=get_val_test_transforms(),
        normalize_type='imagenet',
        check_files=True
    )
    
    # Test dataset (no augmentation)
    test_dataset = MultiPlotDataset(
        csv_file=csv_file,
        plot_dir=plot_dir,
        split='test',
        image_size=image_size,
        transform=get_val_test_transforms(),
        normalize_type='imagenet',
        check_files=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Log dataset statistics
    logger.info("="*60)
    logger.info("Dataset Statistics")
    logger.info("="*60)
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples:   {len(val_dataset)}")
    logger.info(f"Test samples:  {len(test_dataset)}")
    logger.info(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    logger.info(f"Image size:    {image_size}x{image_size}")
    logger.info(f"Batch size:    {batch_size}")
    logger.info(f"Num workers:   {num_workers}")
    logger.info("="*60)
    
    return train_loader, val_loader, test_loader


def analyze_dataset(csv_file: str, plot_dir: str):
    """Analyze dataset and print statistics"""
    df = pd.read_csv(csv_file, sep=';')
    plot_dir = Path(plot_dir)
    
    print("="*60)
    print("Dataset Analysis")
    print("="*60)
    
    # Overall statistics
    print(f"\nTotal samples in CSV: {len(df)}")
    
    # Split statistics
    if 'split' in df.columns:
        print("\nSamples per split:")
        for split in ['train', 'val', 'test']:
            count = len(df[df['split'] == split])
            print(f"  {split:5s}: {count}")
    
    # Label statistics
    if 'label' in df.columns:
        print("\nLabel distribution:")
        print(df['label'].value_counts().sort_index())
        
        # Per-split label distribution
        if 'split' in df.columns:
            print("\nLabel distribution per split:")
            for split in ['train', 'val', 'test']:
                split_df = df[df['split'] == split]
                print(f"\n{split}:")
                print(split_df['label'].value_counts().sort_index())
    
    # Check file availability
    print("\n" + "="*60)
    print("File Availability Check")
    print("="*60)
    
    missing_samples = []
    incomplete_samples = []
    
    for idx, row in df.iterrows():
        filename = row['filename']
        sample_name = Path(filename).stem
        sample_dir = plot_dir / sample_name
        
        if not sample_dir.exists():
            missing_samples.append(sample_name)
            continue
        
        # Check all 9 plots
        missing_plots = []
        for i in range(1, 10):
            plot_path = sample_dir / f"plot_{i:02d}.png"
            if not plot_path.exists():
                missing_plots.append(f"plot_{i:02d}.png")
        
        if missing_plots:
            incomplete_samples.append((sample_name, missing_plots))
    
    print(f"\nSamples with missing directories: {len(missing_samples)}")
    print(f"Samples with incomplete plots: {len(incomplete_samples)}")
    print(f"Valid samples: {len(df) - len(missing_samples) - len(incomplete_samples)}")
    
    if missing_samples:
        print(f"\nFirst 5 missing samples: {missing_samples[:5]}")
    
    if incomplete_samples:
        print(f"\nFirst 5 incomplete samples:")
        for sample, plots in incomplete_samples[:5]:
            print(f"  {sample}: missing {plots}")
    
    print("="*60)


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)
    
    csv_file = "sample_data/sample_labels.csv"
    plot_dir = "sample_data"
    
    print("Analyzing dataset...")
    analyze_dataset(csv_file, plot_dir)
    
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_multiplot_data_loaders(
        csv_file=csv_file,
        plot_dir=plot_dir,
        batch_size=4,
        num_workers=0,  # Use 0 for testing on Windows
        image_size=336
    )
    
    print("\nTesting data loader...")
    for plots, labels, filenames in train_loader:
        print(f"Batch shape: {plots.shape}")  # Should be [B, 9, 3, 336, 336]
        print(f"Labels shape: {labels.shape}")  # Should be [B]
        print(f"Filenames: {filenames[:2]}")
        break
