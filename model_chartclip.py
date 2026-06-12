"""
ChartCLIP-based Multi-Plot Classification Model

Architecture:
1. Per-plot encoder: ChartCLIP (shared across all 9 plots)
2. Aggregation: Transformer encoder over panel embeddings
3. Classification head: MLP with dropout
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModel, AutoImageProcessor
import torchvision.models as models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChartCLIPEncoder(nn.Module):
    """
    Wrapper for ChartCLIP image encoder.
    Uses ikatsuno/chartclip or Junteng/Chart_CLIP
    """
    def __init__(
        self,
        model_name: str = "Junteng/Chart_CLIP",
        freeze_encoder: bool = True,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        logger.info(f"Loading ChartCLIP model: {model_name}")
        
        # Load CLIP vision model
        # Note: Junteng/Chart_CLIP has model files in a 'model/' subfolder
        if "Junteng/Chart_CLIP" in model_name or "Chart_CLIP" in model_name:
            logger.info("Loading from model/ subfolder...")
            self.vision_model = CLIPVisionModel.from_pretrained(model_name, subfolder="model")
            self.image_processor = CLIPImageProcessor.from_pretrained(model_name, subfolder="model")
        else:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Get embedding dimension
        self.embed_dim = self.vision_model.config.hidden_size
        logger.info(f"ChartCLIP embedding dimension: {self.embed_dim}")
        
        # Optional projection layer
        if projection_dim is not None and projection_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.embed_dim
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing ChartCLIP encoder")
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] - preprocessed images
        Returns:
            embeddings: [B, D] - L2-normalized embeddings
        """
        # Get CLIP vision features
        outputs = self.vision_model(pixel_values=pixel_values)
        
        # Use pooled output (CLS token embedding)
        embeddings = outputs.pooler_output  # [B, embed_dim]
        
        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def get_image_processor_params(self):
        """Get preprocessing parameters for the image processor"""
        return {
            'size': self.image_processor.size,
            'mean': self.image_processor.image_mean,
            'std': self.image_processor.image_std
        }


class ResNetEncoder(nn.Module):
    """
    Wrapper for pretrained ResNet models from torchvision
    """
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        logger.info(f"Loading ResNet model: {model_name}, pretrained={pretrained}")
        
        # Load pretrained ResNet from torchvision
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.embed_dim = 512
        elif model_name == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.embed_dim = 512
        elif model_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.embed_dim = 2048
        elif model_name == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.embed_dim = 2048
        elif model_name == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
            self.embed_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        logger.info(f"ResNet embedding dimension: {self.embed_dim}")
        
        # Optional projection layer
        if projection_dim is not None and projection_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.embed_dim
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing ResNet encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] - images
        Returns:
            embeddings: [B, D] - L2-normalized embeddings
        """
        # Get ResNet features
        features = self.encoder(images)  # [B, embed_dim, 1, 1]
        embeddings = features.squeeze(-1).squeeze(-1)  # [B, embed_dim]
        
        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class SwinEncoder(nn.Module):
    """
    Wrapper for pretrained Swin Transformer models from HuggingFace
    """
    def __init__(
        self,
        model_name: str = "microsoft/swin-base-patch4-window7-224",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        logger.info(f"Loading Swin model: {model_name}, pretrained={pretrained}")
        
        # Load pretrained Swin from HuggingFace (use backbone model, not classification)
        if pretrained:
            self.swin_model = AutoModel.from_pretrained(model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.swin_model = AutoModel.from_config(config)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from config
        self.embed_dim = self.swin_model.config.hidden_size
        logger.info(f"Swin embedding dimension: {self.embed_dim}")
        logger.info(f"Swin expected input size: {self.image_processor.size}")
        
        # Optional projection layer
        if projection_dim is not None and projection_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.embed_dim
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing Swin encoder")
            for param in self.swin_model.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] - images (should match expected input size)
        Returns:
            embeddings: [B, D] - L2-normalized embeddings
        """
        # Get Swin features
        outputs = self.swin_model(pixel_values=images)
        
        # Robust pooling: handle different output formats
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            # Preferred: use pooled output if available
            embeddings = outputs.pooler_output  # [B, embed_dim]
        elif hasattr(outputs, "last_hidden_state"):
            # Fallback: pool last_hidden_state
            h = outputs.last_hidden_state
            if h.ndim == 3:  # [B, N, C] - typical for ViT/Swin
                embeddings = h.mean(dim=1)  # Average over tokens
            elif h.ndim == 4:  # [B, C, H, W] - unlikely but handle it
                embeddings = h.mean(dim=(2, 3))  # Global average pooling
            else:
                raise ValueError(f"Unexpected last_hidden_state shape: {h.shape}")
        else:
            raise ValueError(f"Cannot extract features from Swin output: {outputs.keys()}")
        
        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class ConvNeXtEncoder(nn.Module):
    """
    Wrapper for pretrained ConvNeXt models from HuggingFace
    ConvNeXt: A modern pure ConvNet architecture
    """
    def __init__(
        self,
        model_name: str = "facebook/convnext-base-224-22k",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None
    ):
        super().__init__()
        
        logger.info(f"Loading ConvNeXt model: {model_name}, pretrained={pretrained}")
        
        # Load pretrained ConvNeXt from HuggingFace
        if pretrained:
            self.convnext_model = AutoModel.from_pretrained(model_name)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.convnext_model = AutoModel.from_config(config)
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from config
        self.embed_dim = self.convnext_model.config.hidden_sizes[-1]
        logger.info(f"ConvNeXt embedding dimension: {self.embed_dim}")
        
        # Optional projection layer
        if projection_dim is not None and projection_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.embed_dim
        
        # Freeze encoder if specified
        if freeze_encoder:
            logger.info("Freezing ConvNeXt encoder")
            for param in self.convnext_model.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] - images
        Returns:
            embeddings: [B, D] - L2-normalized embeddings
        """
        # Get ConvNeXt features
        outputs = self.convnext_model(pixel_values=images)
        
        # Robust pooling: handle different output formats
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            # Preferred: use pooled output if available
            embeddings = outputs.pooler_output  # [B, embed_dim]
        elif hasattr(outputs, "last_hidden_state"):
            # Fallback: pool last_hidden_state
            h = outputs.last_hidden_state
            if h.ndim == 4:  # [B, C, H, W]
                embeddings = h.mean(dim=(2, 3))  # Global average pooling
            elif h.ndim == 3:  # [B, N, C]
                embeddings = h.mean(dim=1)  # Average over tokens
            else:
                raise ValueError(f"Unexpected last_hidden_state shape: {h.shape}")
        else:
            raise ValueError(f"Cannot extract features from ConvNeXt output: {outputs.keys()}")
        
        # Optional projection
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class PanelPositionEmbedding(nn.Module):
    """
    Learnable position embeddings for the 9 panels (3x3 grid)
    """
    def __init__(self, num_panels: int = 9, embed_dim: int = 768):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.randn(num_panels, embed_dim) * 0.02)
        
    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_embeddings: [B, num_panels, D]
        Returns:
            embeddings with position info: [B, num_panels, D]
        """
        return panel_embeddings + self.position_embeddings.unsqueeze(0)


class TransformerAggregator(nn.Module):
    """
    Transformer encoder to aggregate information from 9 panel embeddings
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, panel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_embeddings: [B, num_panels, D]
        Returns:
            cls_output: [B, D] - aggregated representation
        """
        B = panel_embeddings.size(0)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        
        # Concatenate CLS token with panel embeddings
        x = torch.cat([cls_tokens, panel_embeddings], dim=1)  # [B, num_panels+1, D]
        
        # Apply transformer
        x = self.transformer(x)  # [B, num_panels+1, D]
        
        # Extract CLS token output
        cls_output = x[:, 0]  # [B, D]
        
        # Layer norm
        cls_output = self.norm(cls_output)
        
        return cls_output


class ClassificationHead(nn.Module):
    """
    MLP classification head for binary classification
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 1  # Binary classification: output 1 logit
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]
        Returns:
            logits: [B, 1] for binary classification
        """
        return self.mlp(x)


class MultiPlotChartCLIP(nn.Module):
    """
    Complete model: ChartCLIP encoder + Transformer aggregation + Classification head
    
    Pipeline:
    1. Encode each of 9 plots with shared ChartCLIP encoder
    2. Add position embeddings to distinguish plots
    3. Aggregate with transformer (using CLS token)
    4. Classify with MLP head
    """
    def __init__(
        self,
        chartclip_model: str = "Junteng/Chart_CLIP",
        freeze_encoder: bool = True,
        projection_dim: Optional[int] = None,
        num_panels: int = 9,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        # 1. Per-plot encoder
        self.encoder = ChartCLIPEncoder(
            model_name=chartclip_model,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim
        )
        
        embed_dim = self.encoder.output_dim
        logger.info(f"Using embedding dimension: {embed_dim}")
        
        # 2. Position embeddings
        self.position_embedding = PanelPositionEmbedding(
            num_panels=num_panels,
            embed_dim=embed_dim
        )
        
        # 3. Transformer aggregator
        self.aggregator = TransformerAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        
        # 4. Classification head
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            num_classes=1  # Binary classification
        )
        
        self.num_panels = num_panels
        
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"  Trainable: {self.count_parameters(trainable_only=True)}")
        
    def forward(self, panel_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_images: [B, num_panels, C, H, W] - 9 plot images per sample
        Returns:
            logits: [B, 1] - binary classification logits
        """
        B, num_panels, C, H, W = panel_images.shape
        assert num_panels == self.num_panels, f"Expected {self.num_panels} panels, got {num_panels}"
        
        # Reshape to process all panels together
        panels_flat = panel_images.view(B * num_panels, C, H, W)  # [B*9, C, H, W]
        
        # Encode all panels (shared encoder)
        panel_embeddings = self.encoder(panels_flat)  # [B*9, D]
        
        # Reshape back to batch x panels
        D = panel_embeddings.size(-1)
        panel_embeddings = panel_embeddings.view(B, num_panels, D)  # [B, 9, D]
        
        # Add position information
        panel_embeddings = self.position_embedding(panel_embeddings)  # [B, 9, D]
        
        # Aggregate with transformer
        aggregated = self.aggregator(panel_embeddings)  # [B, D]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, 1]
        
        return logits
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_attention_weights(self, panel_images: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from transformer (for interpretability)
        
        Args:
            panel_images: [B, num_panels, C, H, W]
        Returns:
            attention_weights: [B, num_layers, num_heads, num_panels+1, num_panels+1]
        """
        # This would require modifying TransformerAggregator to return attention weights
        # Left as future enhancement
        raise NotImplementedError("Attention weight extraction not yet implemented")


class MultiPlotResNet(nn.Module):
    """
    Complete model: ResNet encoder + Transformer aggregation + Classification head
    
    Pipeline:
    1. Encode each of 9 plots with shared ResNet encoder
    2. Add position embeddings to distinguish plots
    3. Aggregate with transformer (using CLS token)
    4. Classify with MLP head
    """
    def __init__(
        self,
        resnet_model: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None,
        num_panels: int = 9,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        # 1. Per-plot encoder
        self.encoder = ResNetEncoder(
            model_name=resnet_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim
        )
        
        embed_dim = self.encoder.output_dim
        logger.info(f"Using embedding dimension: {embed_dim}")
        
        # 2. Position embeddings
        self.position_embedding = PanelPositionEmbedding(
            num_panels=num_panels,
            embed_dim=embed_dim
        )
        
        # 3. Transformer aggregator
        self.aggregator = TransformerAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        
        # 4. Classification head
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            num_classes=1  # Binary classification
        )
        
        self.num_panels = num_panels
        
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"  Trainable: {self.count_parameters(trainable_only=True)}")
        
    def forward(self, panel_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_images: [B, num_panels, C, H, W] - 9 plot images per sample
        Returns:
            logits: [B, 1] - binary classification logits
        """
        B, num_panels, C, H, W = panel_images.shape
        assert num_panels == self.num_panels, f"Expected {self.num_panels} panels, got {num_panels}"
        
        # Reshape to process all panels together
        panels_flat = panel_images.view(B * num_panels, C, H, W)  # [B*9, C, H, W]
        
        # Encode all panels (shared encoder)
        panel_embeddings = self.encoder(panels_flat)  # [B*9, D]
        
        # Reshape back to batch x panels
        D = panel_embeddings.size(-1)
        panel_embeddings = panel_embeddings.view(B, num_panels, D)  # [B, 9, D]
        
        # Add position information
        panel_embeddings = self.position_embedding(panel_embeddings)  # [B, 9, D]
        
        # Aggregate with transformer
        aggregated = self.aggregator(panel_embeddings)  # [B, D]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, 1]
        
        return logits
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class MultiPlotSwin(nn.Module):
    """
    Complete model: Swin Transformer encoder + Transformer aggregation + Classification head
    
    Pipeline:
    1. Encode each of 9 plots with shared Swin Transformer encoder
    2. Add position embeddings to distinguish plots
    3. Aggregate with transformer (using CLS token)
    4. Classify with MLP head
    """
    def __init__(
        self,
        swin_model: str = "microsoft/swin-base-patch4-window7-224",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None,
        num_panels: int = 9,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        # 1. Per-plot encoder
        self.encoder = SwinEncoder(
            model_name=swin_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim
        )
        
        embed_dim = self.encoder.output_dim
        logger.info(f"Using embedding dimension: {embed_dim}")
        
        # 2. Position embeddings
        self.position_embedding = PanelPositionEmbedding(
            num_panels=num_panels,
            embed_dim=embed_dim
        )
        
        # 3. Transformer aggregator
        self.aggregator = TransformerAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        
        # 4. Classification head
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            num_classes=1  # Binary classification
        )
        
        self.num_panels = num_panels
        
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"  Trainable: {self.count_parameters(trainable_only=True)}")
        
    def forward(self, panel_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_images: [B, num_panels, C, H, W] - 9 plot images per sample
        Returns:
            logits: [B, 1] - binary classification logits
        """
        B, num_panels, C, H, W = panel_images.shape
        assert num_panels == self.num_panels, f"Expected {self.num_panels} panels, got {num_panels}"
        
        # Reshape to process all panels together
        panels_flat = panel_images.view(B * num_panels, C, H, W)  # [B*9, C, H, W]
        
        # Encode all panels (shared encoder)
        panel_embeddings = self.encoder(panels_flat)  # [B*9, D]
        
        # Reshape back to batch x panels
        D = panel_embeddings.size(-1)
        panel_embeddings = panel_embeddings.view(B, num_panels, D)  # [B, 9, D]
        
        # Add position information
        panel_embeddings = self.position_embedding(panel_embeddings)  # [B, 9, D]
        
        # Aggregate with transformer
        aggregated = self.aggregator(panel_embeddings)  # [B, D]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, 1]
        
        return logits
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class MultiPlotConvNeXt(nn.Module):
    """
    Complete model: ConvNeXt encoder + Transformer aggregation + Classification head
    
    Pipeline:
    1. Encode each of 9 plots with shared ConvNeXt encoder
    2. Add position embeddings to distinguish plots
    3. Aggregate with transformer (using CLS token)
    4. Classify with MLP head
    """
    def __init__(
        self,
        convnext_model: str = "facebook/convnext-base-224-22k",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        projection_dim: Optional[int] = None,
        num_panels: int = 9,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        # 1. Per-plot encoder
        self.encoder = ConvNeXtEncoder(
            model_name=convnext_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim
        )
        
        embed_dim = self.encoder.output_dim
        logger.info(f"Using embedding dimension: {embed_dim}")
        
        # 2. Position embeddings
        self.position_embedding = PanelPositionEmbedding(
            num_panels=num_panels,
            embed_dim=embed_dim
        )
        
        # 3. Transformer aggregator
        self.aggregator = TransformerAggregator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio
        )
        
        # 4. Classification head
        self.classifier = ClassificationHead(
            input_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            num_classes=1  # Binary classification
        )
        
        self.num_panels = num_panels
        
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        logger.info(f"  Trainable: {self.count_parameters(trainable_only=True)}")
        
    def forward(self, panel_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            panel_images: [B, num_panels, C, H, W] - 9 plot images per sample
        Returns:
            logits: [B, 1] - binary classification logits
        """
        B, num_panels, C, H, W = panel_images.shape
        assert num_panels == self.num_panels, f"Expected {self.num_panels} panels, got {num_panels}"
        
        # Reshape to process all panels together
        panels_flat = panel_images.view(B * num_panels, C, H, W)  # [B*9, C, H, W]
        
        # Encode all panels (shared encoder)
        panel_embeddings = self.encoder(panels_flat)  # [B*9, D]
        
        # Reshape back to batch x panels
        D = panel_embeddings.size(-1)
        panel_embeddings = panel_embeddings.view(B, num_panels, D)  # [B, 9, D]
        
        # Add position information
        panel_embeddings = self.position_embedding(panel_embeddings)  # [B, 9, D]
        
        # Aggregate with transformer
        aggregated = self.aggregator(panel_embeddings)  # [B, D]
        
        # Classify
        logits = self.classifier(aggregated)  # [B, 1]
        
        return logits
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_chartclip_model(
    chartclip_model: str = "Junteng/Chart_CLIP",
    freeze_encoder: bool = True,
    projection_dim: Optional[int] = None,
    num_heads: int = 8,
    num_transformer_layers: int = 3,
    dropout: float = 0.1,
    mlp_hidden_dim: int = 256,
    resnet_model: Optional[str] = None,
    swin_model: Optional[str] = None,
    convnext_model: Optional[str] = None,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create the model
    
    If convnext_model is specified, creates MultiPlotConvNeXt
    If swin_model is specified, creates MultiPlotSwin
    If resnet_model is specified, creates MultiPlotResNet
    Otherwise creates MultiPlotChartCLIP
    """
    if convnext_model is not None:
        logger.info(f"Creating ConvNeXt model: {convnext_model}")
        return MultiPlotConvNeXt(
            convnext_model=convnext_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            mlp_hidden_dim=mlp_hidden_dim,
            **kwargs
        )
    elif swin_model is not None:
        logger.info(f"Creating Swin model: {swin_model}")
        return MultiPlotSwin(
            swin_model=swin_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            mlp_hidden_dim=mlp_hidden_dim,
            **kwargs
        )
    elif resnet_model is not None:
        logger.info(f"Creating ResNet model: {resnet_model}")
        return MultiPlotResNet(
            resnet_model=resnet_model,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            mlp_hidden_dim=mlp_hidden_dim,
            **kwargs
        )
    else:
        logger.info(f"Creating ChartCLIP model: {chartclip_model}")
        return MultiPlotChartCLIP(
            chartclip_model=chartclip_model,
            freeze_encoder=freeze_encoder,
            projection_dim=projection_dim,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            mlp_hidden_dim=mlp_hidden_dim,
            **kwargs
        )


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MultiPlotChartCLIP model...")
    model = create_chartclip_model(
        chartclip_model="Junteng/Chart_CLIP",
        freeze_encoder=True,
        num_heads=8,
        num_transformer_layers=3
    )
    
    # Test forward pass
    B = 2
    dummy_input = torch.randn(B, 9, 3, 336, 336)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Trainable parameters: {model.count_parameters(trainable_only=True):,}")
