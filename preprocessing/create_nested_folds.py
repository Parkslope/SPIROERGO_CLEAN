"""
Create nested 5-fold cross-validation with separate val and test sets

For each fold k (0-4):
- Test set: fold_test == k
- Validation set: fold_val == k  
- Training set: all other samples

This ensures non-overlapping train/val/test within each fold.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_nested_folds(csv_path='spiro_binary_labels.csv', n_folds=5, random_state=42):
    """
    Create nested folds with separate validation and test sets
    
    Strategy:
    - Divide data into K folds
    - For each experiment on fold k:
      - fold_test == k: Test set (~20% for 5-fold)
      - fold_val == k: Validation set (~20% for 5-fold)  
      - Rest: Training set (~60% for 5-fold)
    
    Args:
        csv_path: Path to CSV
        n_folds: Number of folds
        random_state: Random seed
    """
    # Load data
    df = pd.read_csv(csv_path, sep=';')
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution:\n{df['binary_class'].value_counts()}")
    
    # Initialize fold columns
    df['fold_test'] = -1
    df['fold_val'] = -1
    
    # Strategy: Create K mutually exclusive groups
    # Then rotate: fold k uses group k as test, group (k+1)%K as val, rest as train
    
    # First, divide all data into K stratified groups
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Assign each sample to a base group (0 to K-1)
    df['base_group'] = -1
    for group, (_, group_idx) in enumerate(skf.split(df.index, df['binary_class'])):
        df.loc[group_idx, 'base_group'] = group
    
    # Now for each fold, assign test and val based on rotation
    for fold in range(n_folds):
        test_group = fold
        val_group = (fold + 1) % n_folds  # Next group becomes validation
        
        test_mask = df['base_group'] == test_group
        val_mask = df['base_group'] == val_group
        
        df.loc[test_mask, 'fold_test'] = fold
        df.loc[val_mask, 'fold_val'] = fold
    
    # Drop temporary column
    df.drop('base_group', axis=1, inplace=True)
    
    # Verify no overlap between val and test in each fold
    print(f"\n{'='*70}")
    print("Fold Statistics:")
    print(f"{'='*70}")
    
    for fold in range(n_folds):
        test_mask = df['fold_test'] == fold
        val_mask = df['fold_val'] == fold
        train_mask = (~test_mask) & (~val_mask)
        
        # Check overlap
        overlap = (test_mask & val_mask).sum()
        
        test_samples = test_mask.sum()
        val_samples = val_mask.sum()
        train_samples = train_mask.sum()
        
        # Class distribution
        test_dist = df[test_mask]['binary_class'].value_counts()
        val_dist = df[val_mask]['binary_class'].value_counts()
        train_dist = df[train_mask]['binary_class'].value_counts()
        
        print(f"\nFold {fold}:")
        print(f"  Train: {train_samples} ({train_samples/len(df)*100:.1f}%) - "
              f"Class 0: {train_dist.get(0,0)}, Class 1: {train_dist.get(1,0)}")
        print(f"  Val:   {val_samples} ({val_samples/len(df)*100:.1f}%) - "
              f"Class 0: {val_dist.get(0,0)}, Class 1: {val_dist.get(1,0)}")
        print(f"  Test:  {test_samples} ({test_samples/len(df)*100:.1f}%) - "
              f"Class 0: {test_dist.get(0,0)}, Class 1: {test_dist.get(1,0)}")
        print(f"  Overlap (should be 0): {overlap}")
    
    # Save
    backup_path = csv_path.replace('.csv', '_backup.csv')
    df_original = pd.read_csv(csv_path, sep=';')
    df_original.to_csv(backup_path, sep=';', index=False)
    
    df.to_csv(csv_path, sep=';', index=False)
    
    print(f"\n{'='*70}")
    print(f"Backup created: {backup_path}")
    print(f"Original file updated: {csv_path}")
    print(f"{'='*70}")
    
    return df

if __name__ == "__main__":
    df = create_nested_folds()
    
    print("\nSample rows:")
    print(df[['id', 'filename', 'binary_class', 'split', 'fold_test', 'fold_val']].head(20))
