"""
Training pipeline for CNN-ViT Hybrid CPET classification model
Includes transfer learning, early stopping, and comprehensive evaluation
Uses Hydra for configuration management and W&B for logging
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

# from model_architecture import create_model  # Not needed for ChartCLIP
# from data_utils import create_data_loaders  # Not needed for ChartCLIP


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, filepath, monitor='val_auc', mode='max', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, score, model, optimizer, epoch, metrics):
        save_checkpoint = False
        
        if not self.save_best_only:
            save_checkpoint = True
        else:
            if self.best_score is None:
                save_checkpoint = True
            elif ((self.mode == 'max' and score > self.best_score) or 
                  (self.mode == 'min' and score < self.best_score)):
                save_checkpoint = True
        
        if save_checkpoint:
            self.best_score = score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': score,
                'metrics': metrics
            }
            torch.save(checkpoint, self.filepath)
            print(f"Checkpoint saved: {self.filepath}")


class MedicalMetrics:
    """Medical-specific evaluation metrics"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive medical evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        # Handle both 1D (binary) and 2D (multiclass) probability arrays
        if y_prob.ndim == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob)
        
        # Clinical metrics
        sensitivity = recall_score(y_true, y_pred, pos_label=1)  # Recall for PH class
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        
        # Positive and Negative Predictive Values
        # Note: confusion matrix already computed above
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # Balanced Accuracy (important for medical applications)
        balanced_acc = (sensitivity + specificity) / 2
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Print metrics in a formatted way"""
        print(f"\n{prefix} Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV (Precision): {metrics['ppv']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")


class CPETTrainer:
    """Main trainer class for CPET classification"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create model
        self.model = create_model(
            model_name=config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            backbone_name=config.model.backbone_name,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        ).to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            csv_file=config.data.csv_file,
            plot_dir=config.data.plot_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            image_size=tuple(config.data.image_size)
        )
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create loss function with class weighting for balanced learning
        self.criterion = self._create_criterion()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize utilities
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            restore_best_weights=config.training.early_stopping.restore_best_weights
        )
        
        # Create checkpoint path
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        self.checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.training.checkpoint.monitor,
            mode=config.training.checkpoint.mode,
            save_best_only=config.training.checkpoint.save_best_only
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training history
        self.history = {'train': [], 'val': []}
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components"""
        # Different learning rates for CNN backbone and ViT
        cnn_params = []
        vit_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'cnn_backbone' in name:
                cnn_params.append(param)
            elif 'vit' in name or 'patch_embedding' in name:
                vit_params.append(param)
            else:
                classifier_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        base_lr = self.config.training.learning_rate
        
        if cnn_params:
            param_groups.append({
                'params': cnn_params,
                'lr': base_lr * self.config.training.learning_rates.cnn_backbone,
                'name': 'cnn_backbone'
            })
        if vit_params:
            param_groups.append({
                'params': vit_params,
                'lr': base_lr * self.config.training.learning_rates.vit_components,
                'name': 'vit'
            })
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': base_lr * self.config.training.learning_rates.classifier,
                'name': 'classifier'
            })
        
        optimizer_name = self.config.training.optimizer.name.lower()
        if optimizer_name == "adamw":
            return optim.AdamW(
                param_groups,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.optimizer.betas,
                eps=self.config.training.optimizer.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_criterion(self):
        """Create loss function with class weighting"""
        if self.config.training.loss.use_class_weights:
            # Calculate class weights from training data
            train_labels = []
            for batch in self.train_loader:
                train_labels.extend(batch['label'].numpy())
            
            class_counts = np.bincount(train_labels)
            total_samples = len(train_labels)
            class_weights = total_samples / (len(class_counts) * class_counts)
            
            print(f"Class weights: {class_weights}")
            weight = torch.FloatTensor(class_weights).to(self.device)
        else:
            weight = None
        
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.config.training.loss.label_smoothing
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.training.scheduler.name.lower()
        
        if scheduler_name == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=self.config.training.scheduler.factor,
                patience=self.config.training.scheduler.patience,
                min_lr=self.config.training.scheduler.min_lr,
                verbose=self.config.training.scheduler.verbose
            )
        elif scheduler_name == 'cosine_annealing':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.epochs
            )
        elif scheduler_name == 'step_lr':
            return StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler.get('step_size', 30),
                gamma=self.config.training.scheduler.get('gamma', 0.1)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        # Calculate metrics
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
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = MedicalMetrics.compute_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """Test the model on test set"""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        all_filenames = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                all_filenames.extend(batch['filename'])
        
        # Calculate metrics
        metrics = MedicalMetrics.compute_metrics(
            np.array(all_labels), 
            np.array(all_preds), 
            np.array(all_probs)
        )
        
        # Save predictions
        results_df = pd.DataFrame({
            'filename': all_filenames,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'prob_class_0': [p[0] for p in all_probs],
            'prob_class_1': [p[1] for p in all_probs]
        })
        
        results_path = os.path.join(self.config['training']['output_dir'], 'test_predictions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Test predictions saved to: {results_path}")
        
        return metrics
    
    def train(self):
        """Main training loop with W&B logging"""
        print(f"Starting training for {self.config.training.epochs} epochs...")
        
        best_val_auc = 0.0
        
        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Print metrics
            MedicalMetrics.print_metrics(train_metrics, "Train")
            MedicalMetrics.print_metrics(val_metrics, "Validation")
            
            # Log to W&B
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # Check for improvement
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                print(f"New best validation AUC: {best_val_auc:.4f}")
            
            # Checkpoint
            self.checkpoint(val_metrics['auc'], self.model, self.optimizer, epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics['auc'], self.model):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL EVALUATION ON TEST SET")
        print("="*50)
        
        test_metrics = self.test_model()
        MedicalMetrics.print_metrics(test_metrics, "Test")
        
        # Save training history
        self._save_history()
        self._plot_training_curves()
        
        return test_metrics
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard"""
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', value, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    def _save_history(self):
        """Save training history"""
        history_path = os.path.join(self.config['training']['output_dir'], 'training_history.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        history_serializable = convert_types(self.history)
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.history['train']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, [m['loss'] for m in self.history['train']], 'b-', label='Train')
        axes[0, 0].plot(epochs, [m['loss'] for m in self.history['val']], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC
        axes[0, 1].plot(epochs, [m['auc'] for m in self.history['train']], 'b-', label='Train')
        axes[0, 1].plot(epochs, [m['auc'] for m in self.history['val']], 'r-', label='Validation')
        axes[0, 1].set_title('AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(epochs, [m['accuracy'] for m in self.history['train']], 'b-', label='Train')
        axes[1, 0].plot(epochs, [m['accuracy'] for m in self.history['val']], 'r-', label='Validation')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Sensitivity
        axes[1, 1].plot(epochs, [m['sensitivity'] for m in self.history['train']], 'b-', label='Train')
        axes[1, 1].plot(epochs, [m['sensitivity'] for m in self.history['val']], 'r-', label='Validation')
        axes[1, 1].set_title('Sensitivity (Recall)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sensitivity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['training']['output_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")


def load_config(config_path: str = None) -> Dict:
    """Load training configuration"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'model': {
            'name': 'cnn_vit_hybrid',
            'num_classes': 2,
            'pretrained': True,
            'kwargs': {
                'backbone_name': 'resnet50',
                'embed_dim': 768,
                'num_heads': 12,
                'num_layers': 6,
                'dropout': 0.2
            }
        },
        'data': {
            'csv_file': 'spiro_binary_labels.csv',
            'plot_dir': 'data/preprocessed_plots',
            'image_size': [512, 512]
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'reduce_on_plateau',
            'early_stopping_patience': 15,
            'early_stopping_delta': 0.001,
            'num_workers': 4,
            'checkpoint_path': 'checkpoints/best_model.pth',
            'log_dir': 'logs',
            'output_dir': 'results'
        }
    }


def main():
    """Main training function"""
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Save configuration
    config_path = os.path.join(config['training']['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer and start training
    trainer = CPETTrainer(config)
    test_metrics = trainer.train()
    
    print(f"\nTraining completed! Final test metrics:")
    MedicalMetrics.print_metrics(test_metrics, "Final Test")


if __name__ == "__main__":
    main()
