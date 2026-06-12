"""
Training script for ChartCLIP Multi-Plot classification model

Uses:
- ChartCLIP encoder for per-plot feature extraction
- Transformer for multi-plot aggregation
- Binary classification for PH detection
- WandB logging
- Hydra configuration
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

# Import custom modules
from model_chartclip import create_chartclip_model
from data_loader_multiplot import create_multiplot_data_loaders
from training import EarlyStopping, ModelCheckpoint, MedicalMetrics
from utils.visualization import log_plots_to_wandb, save_plots_locally

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChartCLIPTrainer:
    """Trainer for ChartCLIP Multi-Plot model"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create output directories
        self._create_directories()
        
        # Initialize WandB
        self._init_wandb()
        
        # Create data loaders
        logger.info("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Create model
        logger.info("Creating model...")
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Mixed precision training
        self.use_amp = config.mixed_precision and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            restore_best_weights=config.training.early_stopping.restore_best_weights
        )
        
        self.checkpoint = ModelCheckpoint(
            filepath=Path(config.checkpoint_dir) / "best_model.pth",
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )
        
        logger.info("Trainer initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Trainable parameters: {self.model.count_parameters(trainable_only=True):,}")
        
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        return device
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_directories(self):
        """Create output directories"""
        for dir_name in ['output_dir', 'checkpoint_dir', 'log_dir', 'results_dir']:
            path = Path(self.config[dir_name])
            path.mkdir(parents=True, exist_ok=True)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if self.config.wandb.mode != 'disabled':
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=OmegaConf.to_container(self.config, resolve=True),
                mode=self.config.wandb.mode
            )
            logger.info("WandB initialized")
    
    def _create_data_loaders(self):
        """Create data loaders"""
        return create_multiplot_data_loaders(
            csv_file=self.config.data.csv_file,
            plot_dir=self.config.data.plot_dir,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            pin_memory=self.config.data.pin_memory,
            drop_last_train=self.config.data.drop_last
        )
    
    def _create_model(self):
        """Create model (ChartCLIP, ResNet, Swin, or ConvNeXt)"""
        # Check which encoder backbone is specified
        resnet_model = getattr(self.config.model, 'resnet_model', None)
        swin_model = getattr(self.config.model, 'swin_model', None)
        convnext_model = getattr(self.config.model, 'convnext_model', None)
        pretrained = getattr(self.config.model, 'pretrained', True)
        
        return create_chartclip_model(
            chartclip_model=getattr(self.config.model, 'chartclip_model', 'Junteng/Chart_CLIP'),
            freeze_encoder=self.config.model.freeze_encoder,
            projection_dim=self.config.model.projection_dim,
            num_heads=self.config.model.num_heads,
            num_transformer_layers=self.config.model.num_transformer_layers,
            dropout=self.config.model.dropout,
            mlp_hidden_dim=self.config.model.mlp_hidden_dim,
            mlp_ratio=getattr(self.config.model, 'mlp_ratio', 4.0),
            resnet_model=resnet_model,
            swin_model=swin_model,
            convnext_model=convnext_model,
            pretrained=pretrained
        )
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config.training.optimizer.name.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.training.scheduler.name.lower()
        
        if scheduler_name == 'cosine' or scheduler_name == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step' or scheduler_name == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_name == 'plateau' or scheduler_name == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_criterion(self):
        """Create loss function"""
        # Binary classification with BCEWithLogitsLoss
        # Compute class weights if specified
        if self.config.data.class_weights == 'auto':
            # Count labels in training set
            labels = []
            for _, label, _ in self.train_loader.dataset:
                labels.append(label)
            labels = np.array(labels)
            
            # Compute inverse frequency weights
            pos_count = (labels == 1).sum()
            neg_count = (labels == 0).sum()
            pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            pos_weight = torch.tensor([pos_weight], device=self.device)
            
            logger.info(f"Using class weight (positive): {pos_weight.item():.2f}")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (plots, labels, _) in enumerate(pbar):
            # Move to device
            plots = plots.to(self.device)  # [B, 9, 3, 336, 336]
            labels = labels.to(self.device).float().unsqueeze(1)  # [B, 1]
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits = self.model(plots)  # [B, 1]
                    loss = self.criterion(logits, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(plots)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = MedicalMetrics.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for plots, labels, _ in pbar:
                plots = plots.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(plots)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(plots)
                    loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.flatten())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = MedicalMetrics.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """Test the model"""
        logger.info("Testing model...")
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_sample_ids = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for plots, labels, sample_ids in pbar:
                plots = plots.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(plots)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(plots)
                    loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.flatten())
                all_sample_ids.extend(sample_ids)
        
        # Compute metrics
        avg_loss = running_loss / len(self.test_loader)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        metrics = MedicalMetrics.compute_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = avg_loss
        
        MedicalMetrics.print_metrics(metrics, prefix="Test")
        
        # Save predictions as CSV
        logger.info("Saving test predictions...")
        predictions_df = pd.DataFrame({
            'sample_id': all_sample_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'predicted_probability': y_prob
        })
        
        # Save to logs directory
        predictions_path = Path(self.config.log_dir) / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
        
        # Generate and log plots
        logger.info("Generating ROC curve and confusion matrix...")
        
        # Save plots locally
        plot_dir = Path(self.config.results_dir) / "plots"
        save_plots_locally(y_true, y_pred, y_prob, plot_dir, prefix="test")
        
        # Log to WandB
        if self.config.wandb.mode != 'disabled':
            log_plots_to_wandb(y_true, y_pred, y_prob, prefix="test")
            logger.info("Plots logged to W&B")
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.training.epochs}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Checkpoint
            self.checkpoint(val_metrics['auc'], self.model, self.optimizer, epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['auc'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Track best
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
        
        # Test with best model
        logger.info("\n" + "="*60)
        logger.info("Training complete. Testing best model...")
        test_metrics = self.test_model()
        
        # Log test metrics to WandB
        if self.config.wandb.mode != 'disabled':
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Finish WandB
        if self.config.wandb.mode != 'disabled':
            wandb.finish()
        
        logger.info("Training pipeline complete!")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console and WandB"""
        # Console logging
        logger.info(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"AUC: {train_metrics['auc']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}")
        
        # WandB logging
        if self.config.wandb.mode != 'disabled':
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/auc': train_metrics['auc'],
                'train/sensitivity': train_metrics['sensitivity'],
                'train/specificity': train_metrics['specificity'],
                'train/ppv': train_metrics['ppv'],
                'train/f1': train_metrics['f1_score'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/auc': val_metrics['auc'],
                'val/sensitivity': val_metrics['sensitivity'],
                'val/specificity': val_metrics['specificity'],
                'val/ppv': val_metrics['ppv'],
                'val/f1': val_metrics['f1_score'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)


@hydra.main(version_base=None, config_path="configs", config_name="config_chartclip")
def main(config: DictConfig) -> None:
    """Main training function"""
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Create trainer and train
    trainer = ChartCLIPTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
