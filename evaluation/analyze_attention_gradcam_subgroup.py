"""
Attention Weight + Grad-CAM Analysis for ConvNeXt Multi-Panel Aggregator
PAH vs. CTEPH Subgroup Analysis

Generates:
1. Panel attention weights (similarity-based)
2. Grad-CAM heatmaps for each of the 9 panels
3. Original panel images
4. Combined visualizations

Output structure:
results/
  ├── heatmaps/
  │   ├── attention_pah_vs_cteph_comparison.png
  │   └── ...
  ├── gradcam_samples/
  │   ├── sample_001_PAH/
  │   │   ├── panel_01_original.png
  │   │   ├── panel_01_gradcam.png
  │   │   ├── panel_01_overlay.png
  │   │   ├── ...
  │   │   ├── panel_09_overlay.png
  │   │   ├── attention_weights.json
  │   │   └── combined_view.png
  │   └── ...
  └── attention_summary.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_chartclip import MultiPlotConvNeXt, MultiPlotResNet
from data_loader_multiplot import MultiPlotDataset
from torch.utils.data import DataLoader


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt encoder
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: [B, C, H, W]
            target_class: if None, uses predicted class
        Returns:
            cam: [B, H, W] numpy array with values in [0, 1]
        """
        # Forward pass
        output = input_tensor
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        
        # For binary classification with sigmoid, we use the logit directly
        # We want to maximize the output for positive class
        output.backward(gradient=torch.ones_like(output), retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)  # [B, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        for i in range(cam_np.shape[0]):
            cam_np[i] = (cam_np[i] - cam_np[i].min()) / (cam_np[i].max() - cam_np[i].min() + 1e-8)
        
        return cam_np


def get_gradcam_target_layer(model, model_type="convnext"):
    """
    Get the target layer for Grad-CAM based on model type
    
    For ConvNeXt: Last convolutional layer before global pooling
    For ResNet: layer4 (last residual block)
    """
    if model_type == "convnext":
        # ConvNeXt structure: encoder.convnext_model.encoder.stages[-1].layers[-1]
        # We want the last stage's last layer
        try:
            target_layer = model.encoder.convnext_model.encoder.stages[-1].layers[-1]
            return target_layer
        except:
            print("Warning: Could not find ConvNeXt target layer, using stages[-1]")
            return model.encoder.convnext_model.encoder.stages[-1]
    
    elif model_type == "resnet":
        # ResNet structure: encoder.encoder[7] is layer4
        return model.encoder.encoder[7]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_gradcam_for_panels(
    model: nn.Module,
    panel_images: torch.Tensor,
    target_class: int,
    model_type: str = "convnext"
) -> np.ndarray:
    """
    Compute Grad-CAM for each of the 9 panels
    
    Args:
        model: trained model
        panel_images: [B, 9, C, H, W]
        target_class: target class for Grad-CAM (0 or 1)
        model_type: "convnext" or "resnet"
    
    Returns:
        gradcams: [B, 9, H, W] numpy array of heatmaps
    """
    model.eval()
    B, num_panels, C, H, W = panel_images.shape
    
    # Get target layer
    target_layer = get_gradcam_target_layer(model, model_type)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    all_cams = []
    
    for panel_idx in range(num_panels):
        # Get single panel
        single_panel = panel_images[:, panel_idx, :, :, :]  # [B, C, H, W]
        single_panel.requires_grad = True
        
        # Forward through encoder only
        panel_embedding = model.encoder(single_panel)  # [B, D]
        
        # To get proper gradients, we need to do a full forward pass
        # but we'll focus on this panel's contribution
        
        # Create a batch with only this panel repeated 9 times
        repeated_panels = single_panel.unsqueeze(1).repeat(1, num_panels, 1, 1, 1)  # [B, 9, C, H, W]
        
        # Forward pass
        logits = model(repeated_panels)  # [B, 1]
        
        # Generate CAM
        cam = gradcam.generate_cam(logits, target_class)  # [B, H_feat, W_feat]
        
        # Resize to input size
        cam_resized = np.zeros((B, H, W))
        for b in range(B):
            cam_resized[b] = cv2.resize(cam[b], (W, H))
        
        all_cams.append(cam_resized)
        
        # Clear gradients
        model.zero_grad()
    
    # Stack: [9, B, H, W] -> [B, 9, H, W]
    all_cams = np.stack(all_cams, axis=1)
    
    return all_cams


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
        similarities = torch.matmul(
            cls_output.unsqueeze(1),  # [B, 1, D]
            panel_embeddings.transpose(1, 2)  # [B, D, 9]
        ).squeeze(1)  # [B, 9]
        
        # Normalize to [0, 1]
        attention = torch.softmax(similarities, dim=1)
    
    return attention.cpu().numpy()


def save_gradcam_visualization(
    original_img: np.ndarray,
    gradcam_heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.4
):
    """
    Save Grad-CAM overlay visualization
    
    Args:
        original_img: [H, W, 3] RGB image (0-255)
        gradcam_heatmap: [H, W] heatmap (0-1)
        output_path: where to save
        alpha: overlay transparency
    """
    # Ensure original_img is uint8
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)
    
    # Create heatmap colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    
    # Save
    Image.fromarray(overlay).save(output_path)


def process_sample_with_gradcam(
    model: nn.Module,
    panel_images: torch.Tensor,
    label: int,
    sample_id: str,
    plot_dir: Path,
    output_dir: Path,
    device: torch.device,
    model_type: str = "convnext"
):
    """
    Process a single sample: generate Grad-CAMs and save all visualizations
    
    Args:
        model: trained model
        panel_images: [1, 9, C, H, W]
        label: ground truth label
        sample_id: sample identifier
        plot_dir: directory with original plots
        output_dir: where to save results
        device: torch device
        model_type: "convnext" or "resnet"
    """
    model.eval()
    
    # Create sample directory
    label_str = "CTEPH" if label == 1 else "PAH"
    sample_dir = output_dir / f"{sample_id}_{label_str}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prediction
    with torch.no_grad():
        logits = model(panel_images)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)
        pred_str = "CTEPH" if pred == 1 else "PAH"
    
    # Get attention weights
    attention = extract_attention_with_hooks(model, panel_images, device)[0]  # [9]
    
    # Get target layer for Grad-CAM
    target_layer = get_gradcam_target_layer(model, model_type)
    
    # Compute Grad-CAM for each panel
    gradcams = []
    
    for panel_idx in range(9):
        # Initialize Grad-CAM for this panel
        gradcam = GradCAM(model.encoder, target_layer)
        
        # Get single panel and enable gradients
        single_panel = panel_images[:, panel_idx, :, :, :].clone().detach().to(device)  # [1, C, H, W]
        single_panel.requires_grad = True
        
        # Forward through encoder to get features
        model.zero_grad()
        features = model.encoder(single_panel)  # [1, D]
        
        # We need to backprop through the whole model to get proper gradients
        # Create full input with this panel repeated
        full_input = panel_images.clone().to(device)
        full_input.requires_grad = True
        
        # Forward pass
        model.zero_grad()
        logits_full = model(full_input)
        
        # Backward through the network
        logits_full.backward()
        
        # Now process just this panel's encoder to get Grad-CAM
        model.zero_grad()
        single_panel_grad = panel_images[:, panel_idx, :, :, :].clone().to(device)
        single_panel_grad.requires_grad = True
        
        # Reinitialize Grad-CAM hooks
        gradcam = GradCAM(model.encoder, target_layer)
        
        # Forward
        enc_output = model.encoder(single_panel_grad)
        
        # Backward (simulate importance by using logit as target)
        enc_output.backward(gradient=torch.ones_like(enc_output))
        
        # Generate CAM
        if gradcam.gradients is not None and gradcam.activations is not None:
            gradients = gradcam.gradients
            activations = gradcam.activations
            
            # Compute weights
            if gradients.ndim == 4:  # [B, C, H, W]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1)  # [B, H, W]
            elif gradients.ndim == 3:  # [B, H, W, C] or [B, C, H]
                if gradients.shape[-1] == activations.shape[-1]:  # [..., C]
                    weights = gradients.mean(dim=(1, 2), keepdim=True)
                    cam = (weights * activations).sum(dim=-1)
                else:
                    weights = gradients.mean(dim=-1, keepdim=True)
                    cam = (weights * activations).sum(dim=1)
            else:
                # Fallback: use attention-based visualization
                cam = torch.ones(1, 7, 7) * attention[panel_idx]
            
            cam = F.relu(cam)
            cam_np = cam[0].detach().cpu().numpy()
            
            # Normalize
            if cam_np.max() > cam_np.min():
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
            else:
                cam_np = cam_np * 0 + attention[panel_idx]
        else:
            # Fallback: use attention weight
            cam_np = np.ones((7, 7)) * attention[panel_idx]
        
        gradcams.append(cam_np)
    
    # Load original images and save visualizations
    if isinstance(sample_id, str):
        sample_id_path = sample_id
    else:
        sample_id_path = str(sample_id)
    
    source_dir = plot_dir / Path(sample_id_path).stem
    
    if not source_dir.exists():
        print(f"Warning: Source directory not found: {source_dir}")
        return
    
    # Create combined figure
    fig, axes = plt.subplots(3, 9, figsize=(27, 9))
    fig.suptitle(f'Sample: {sample_id}\nTrue: {label_str} | Pred: {pred_str} (prob={prob:.3f})', 
                 fontsize=16, fontweight='bold')
    
    for panel_idx in range(9):
        i, j = panel_idx // 3, panel_idx % 3
        
        # Load original image
        plot_path = source_dir / f"plot_{panel_idx+1:02d}.png"
        
        if plot_path.exists():
            original_img = np.array(Image.open(plot_path).convert('RGB'))
            H, W = original_img.shape[:2]
            
            # Resize Grad-CAM to match image
            gradcam_resized = cv2.resize(gradcams[panel_idx], (W, H))
            
            # Save individual files
            # 1. Original
            Image.fromarray(original_img).save(sample_dir / f"panel_{panel_idx+1:02d}_original.png")
            
            # 2. Grad-CAM heatmap
            plt.imsave(sample_dir / f"panel_{panel_idx+1:02d}_gradcam.png", 
                      gradcam_resized, cmap='jet')
            
            # 3. Overlay
            save_gradcam_visualization(
                original_img,
                gradcam_resized,
                sample_dir / f"panel_{panel_idx+1:02d}_overlay.png",
                alpha=0.4
            )
            
            # Add to combined figure
            # Row 1: Original
            axes[0, panel_idx].imshow(original_img)
            axes[0, panel_idx].set_title(f'P{panel_idx+1}\nAtt: {attention[panel_idx]:.3f}', fontsize=9)
            axes[0, panel_idx].axis('off')
            
            # Row 2: Grad-CAM
            axes[1, panel_idx].imshow(gradcam_resized, cmap='jet')
            axes[1, panel_idx].set_title(f'Grad-CAM', fontsize=9)
            axes[1, panel_idx].axis('off')
            
            # Row 3: Overlay
            heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
            axes[2, panel_idx].imshow(overlay)
            axes[2, panel_idx].set_title(f'Overlay', fontsize=9)
            axes[2, panel_idx].axis('off')
        else:
            for row in range(3):
                axes[row, panel_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row, panel_idx].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Overlay', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(sample_dir / "combined_view.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save attention weights JSON
    with open(sample_dir / "attention_weights.json", 'w') as f:
        json.dump({
            'sample_id': sample_id,
            'true_label': label,
            'predicted_label': pred,
            'probability': float(prob),
            'attention_weights': attention.tolist()
        }, f, indent=2)
    
    # =========================================================================
    # CREATE 3 SEPARATE 3x3 GRIDS
    # =========================================================================
    
    # Normalize attention for overlay
    att_normalized = attention / attention.max()
    
    # Grid 1: Clean Original Panels (3x3)
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 15))
    fig1.suptitle(f'Original Panels: {sample_id}\nTrue: {label_str} | Pred: {pred_str} (prob={prob:.3f})', 
                  fontsize=16, fontweight='bold')
    
    # Grid 2: Attention Overlay (3x3)
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
    fig2.suptitle(f'Attention Overlay: {sample_id}\nTrue: {label_str} | Pred: {pred_str} (prob={prob:.3f})', 
                  fontsize=16, fontweight='bold')
    
    # Grid 3: Grad-CAM Overlay (3x3)
    fig3, axes3 = plt.subplots(3, 3, figsize=(15, 15))
    fig3.suptitle(f'Grad-CAM Overlay: {sample_id}\nTrue: {label_str} | Pred: {pred_str} (prob={prob:.3f})', 
                  fontsize=16, fontweight='bold')
    
    for idx in range(9):
        i, j = idx // 3, idx % 3
        
        plot_path = source_dir / f"plot_{idx+1:02d}.png"
        
        if plot_path.exists():
            # Load original image
            img = Image.open(plot_path)
            img_array = np.array(img.convert('RGB'))
            H, W = img_array.shape[:2]
            
            # Resize Grad-CAM to match image
            gradcam_resized = cv2.resize(gradcams[idx], (W, H))
            
            # Grid 1: Clean original
            axes1[i, j].imshow(img)
            axes1[i, j].set_title(f'Panel {idx+1}', fontsize=10, fontweight='bold')
            axes1[i, j].axis('off')
            
            # Grid 2: Attention overlay
            axes2[i, j].imshow(img, alpha=0.7)
            
            # Create attention-based colored overlay
            attention_value = att_normalized[idx]
            overlay_att = np.ones((*img.size[::-1], 3))
            overlay_att[:, :, 0] = 1.0  # Red channel
            overlay_att[:, :, 1] = 1.0 - attention_value  # Less green = more attention
            overlay_att[:, :, 2] = 1.0 - attention_value  # Less blue = more attention
            
            axes2[i, j].imshow(overlay_att, alpha=attention_value * 0.4)
            axes2[i, j].set_title(f'Panel {idx+1}\nAttention: {attention[idx]:.3f}', 
                                 fontsize=10, fontweight='bold')
            axes2[i, j].axis('off')
            
            # Grid 3: Grad-CAM overlay
            heatmap_gradcam = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
            heatmap_gradcam = cv2.cvtColor(heatmap_gradcam, cv2.COLOR_BGR2RGB)
            overlay_gradcam = cv2.addWeighted(img_array, 0.6, heatmap_gradcam, 0.4, 0)
            
            axes3[i, j].imshow(overlay_gradcam)
            axes3[i, j].set_title(f'Panel {idx+1}\nAttention: {attention[idx]:.3f}', 
                                 fontsize=10, fontweight='bold')
            axes3[i, j].axis('off')
        else:
            # Missing panel
            for ax in [axes1[i, j], axes2[i, j], axes3[i, j]]:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.axis('off')
    
    # Save the 3 grid images
    fig1.tight_layout()
    fig1.savefig(sample_dir / "grid_1_original.png", dpi=200, bbox_inches='tight')
    plt.close(fig1)
    
    fig2.tight_layout()
    fig2.savefig(sample_dir / "grid_2_attention_overlay.png", dpi=200, bbox_inches='tight')
    plt.close(fig2)
    
    fig3.tight_layout()
    fig3.savefig(sample_dir / "grid_3_gradcam_overlay.png", dpi=200, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"  Saved: {sample_dir.name}")


def compute_average_attention_per_class(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Compute average attention weights per class
    
    Returns:
        results: dict with keys 'pah', 'cteph', 'correct', 'incorrect'
    """
    model.eval()
    
    attention_pah = []
    attention_cteph = []
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
            attention = extract_attention_with_hooks(model, panel_images, device)
            
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


def plot_comparison_heatmaps(attention_pah: np.ndarray, attention_cteph: np.ndarray, output_path: Path):
    """Plot side-by-side comparison of PAH vs CTEPH attention"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PAH cases
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
    
    # CTEPH cases
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


def load_model_checkpoint(checkpoint_path: str, device: torch.device, model_type: str = "convnext") -> nn.Module:
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


def create_test_loader(csv_file: str, plot_dir: str, fold: int, batch_size: int = 1) -> DataLoader:
    """Create test data loader for a specific fold"""
    import tempfile
    
    df = pd.read_csv(csv_file, sep=';')
    test_df = df[df['fold_test'] == fold].copy()
    test_df['split'] = 'test'
    
    temp_dir = Path(tempfile.gettempdir()) / "gradcam_eval_subgroup"
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
        num_workers=0,  # Set to 0 for Grad-CAM computation
        pin_memory=True
    )
    
    return test_loader, test_df


def main():
    """Main analysis function with Grad-CAM"""
    import argparse
    parser = argparse.ArgumentParser(description="Grad-CAM + attention analysis for PAH vs CTEPH subgroup")
    parser.add_argument("--checkpoint_dir",      default="checkpoints/ConvNeXt_Baseline")
    parser.add_argument("--csv_file",            default="newV2_subgroup_classified.csv")
    parser.add_argument("--plot_dir",            required=True, help="Directory of preprocessed plot PNGs")
    parser.add_argument("--output_dir",          default="evaluation/results/subgroup_gradcam_convnext")
    parser.add_argument("--model_type",          default="convnext", choices=["convnext", "resnet"])
    parser.add_argument("--n_folds",             type=int, default=5)
    parser.add_argument("--target_pah_correct",  type=int, default=40)
    parser.add_argument("--target_cteph_correct", type=int, default=40)
    args = parser.parse_args()

    checkpoint_dir       = Path(args.checkpoint_dir)
    csv_file             = args.csv_file
    plot_dir             = Path(args.plot_dir)
    output_dir           = Path(args.output_dir)
    model_type           = args.model_type
    n_folds              = args.n_folds
    target_pah_correct   = args.target_pah_correct
    target_cteph_correct = args.target_cteph_correct

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nGrad-CAM + Attention Analysis - PAH vs. CTEPH Subgroup")
    print(f"=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"CSV file: {csv_file}")
    print(f"Plot directory: {plot_dir}")
    print(f"=" * 80)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heatmaps").mkdir(exist_ok=True)
    (output_dir / "gradcam_samples").mkdir(exist_ok=True)
    
    # Aggregate attention across all folds
    all_attention_pah = []
    all_attention_cteph = []
    
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
        
        # Create test loader (batch_size=1 for Grad-CAM)
        print(f"Creating test data loader...")
        test_loader, test_df = create_test_loader(csv_file, str(plot_dir), fold, batch_size=1)
        print(f"✓ Test loader created ({len(test_df)} samples)")
        
        # Load test predictions to find correctly classified samples
        pred_file = checkpoint_dir / f"fold_{fold}" / "test_predictions.csv"
        
        if not pred_file.exists():
            print(f"⚠️ Predictions file not found: {pred_file}")
            print(f"Falling back to computing predictions...")
            
            # Fallback: compute predictions
            print(f"\nGenerating Grad-CAM visualizations for correctly classified samples...")
            pah_correct_count = 0
            cteph_correct_count = 0
            
            for panel_images, labels, filenames in test_loader:
                if pah_correct_count >= target_pah_correct and cteph_correct_count >= target_cteph_correct:
                    break
                
                panel_images = panel_images.to(device)
                label = labels.item()
                
                with torch.no_grad():
                    logits = model(panel_images)
                    prob = torch.sigmoid(logits).item()
                    pred = int(prob >= 0.5)
                
                if label != pred:
                    continue
                
                if label == 0 and pah_correct_count >= target_pah_correct:
                    continue
                if label == 1 and cteph_correct_count >= target_cteph_correct:
                    continue
                
                sample_id = Path(filenames[0]).stem
                
                process_sample_with_gradcam(
                    model,
                    panel_images,
                    label,
                    sample_id,
                    plot_dir,
                    output_dir / "gradcam_samples",
                    device,
                    model_type=model_type
                )
                
                if label == 0:
                    pah_correct_count += 1
                else:
                    cteph_correct_count += 1
                
                print(f"  Progress: PAH={pah_correct_count}/{target_pah_correct}, CTEPH={cteph_correct_count}/{target_cteph_correct}")
        else:
            # Efficient approach: use pre-computed predictions
            print(f"Loading predictions from: {pred_file.name}")
            pred_df = pd.read_csv(pred_file)
            
            # Filter for correctly classified samples
            pred_df['correct'] = pred_df['true_label'] == pred_df['predicted_label']
            correct_df = pred_df[pred_df['correct']].copy()
            
            # Separate by class
            pah_correct = correct_df[correct_df['true_label'] == 0].head(target_pah_correct)
            cteph_correct = correct_df[correct_df['true_label'] == 1].head(target_cteph_correct)
            
            selected_samples = pd.concat([pah_correct, cteph_correct])
            
            print(f"  Selected {len(pah_correct)} PAH and {len(cteph_correct)} CTEPH samples")
            print(f"\nGenerating Grad-CAM visualizations...")
            
            # Process only selected samples
            pah_count = 0
            cteph_count = 0
            
            for panel_images, labels, filenames in test_loader:
                sample_filename = Path(filenames[0]).name
                
                # Check if this sample is in our selected list
                if sample_filename not in selected_samples['sample_id'].values:
                    continue
                
                panel_images = panel_images.to(device)
                label = labels.item()
                sample_id = Path(filenames[0]).stem
                
                # Get attention and process
                process_sample_with_gradcam(
                    model,
                    panel_images,
                    label,
                    sample_id,
                    plot_dir,
                    output_dir / "gradcam_samples",
                    device,
                    model_type=model_type
                )
                
                if label == 0:
                    pah_count += 1
                else:
                    cteph_count += 1
                
                print(f"  Progress: PAH={pah_count}/{len(pah_correct)}, CTEPH={cteph_count}/{len(cteph_correct)}")
                
                # Stop if we've processed all selected samples
                if pah_count >= len(pah_correct) and cteph_count >= len(cteph_correct):
                    break
            
            # Compute attention for statistics (only for selected samples)
            print(f"Computing attention statistics for selected samples...")
            selected_attention_pah = []
            selected_attention_cteph = []
            
            for panel_images, labels, filenames in test_loader:
                sample_filename = Path(filenames[0]).name
                
                if sample_filename not in selected_samples['sample_id'].values:
                    continue
                
                panel_images = panel_images.to(device)
                label = labels.item()
                
                attention = extract_attention_with_hooks(model, panel_images, device)[0]
                
                if label == 0:
                    selected_attention_pah.append(attention)
                else:
                    selected_attention_cteph.append(attention)
            
            all_attention_pah.append(np.array(selected_attention_pah))
            all_attention_cteph.append(np.array(selected_attention_cteph))
        
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    all_attention_pah = np.concatenate(all_attention_pah, axis=0)
    all_attention_cteph = np.concatenate(all_attention_cteph, axis=0)
    
    mean_attention_pah = all_attention_pah.mean(axis=0)
    mean_attention_cteph = all_attention_cteph.mean(axis=0)
    
    print(f"\nAverage attention weights:")
    print(f"  PAH (n={len(all_attention_pah)}): {mean_attention_pah}")
    print(f"  CTEPH (n={len(all_attention_cteph)}): {mean_attention_cteph}")
    
    # Save numerical results
    results_dict = {
        'mean_attention_pah': mean_attention_pah.tolist(),
        'mean_attention_cteph': mean_attention_cteph.tolist(),
        'n_pah': len(all_attention_pah),
        'n_cteph': len(all_attention_cteph)
    }
    
    with open(output_dir / "attention_summary.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Saved: {output_dir / 'attention_summary.json'}")
    
    # Create comparison heatmap
    print(f"\nGenerating comparison heatmap...")
    plot_comparison_heatmaps(
        mean_attention_pah,
        mean_attention_cteph,
        output_dir / "heatmaps" / "attention_pah_vs_cteph_comparison.png"
    )
    
    print(f"\n{'='*80}")
    print(f"Grad-CAM + Attention Analysis Complete!")
    print(f"Total correctly classified samples processed:")
    print(f"  PAH: {target_pah_correct}")
    print(f"  CTEPH: {target_cteph_correct}")
    print(f"  Total: {target_pah_correct + target_cteph_correct}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
