"""
Advanced Model Architectures for Bone Fracture Detection
Implements EfficientNet, DenseNet, and Ensemble models with state-of-the-art techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import timm
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import logging

from configs.config import get_config


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature enhancement"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        return self.out_linear(attn_output)


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant image regions"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module using squeeze-and-excitation"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pooling
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1)
        
        # Max pooling
        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc(y2)
        
        # Combine and apply attention
        attention = self.sigmoid(y1 + y2).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EnhancedEfficientNet(nn.Module):
    """Enhanced EfficientNet with attention mechanisms and custom head"""
    
    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.4,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            spatial_size = features.shape[2]
        
        # Add attention modules
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * 2, 512),  # *2 for avg + max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        if self.use_attention:
            features = self.cbam(features)
        
        # Global pooling
        avg_pool = self.global_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        
        # Combine pooled features
        combined_features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for visualization"""
        return self.backbone(x)


class EnhancedDenseNet(nn.Module):
    """Enhanced DenseNet with attention and custom head"""
    
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.4,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Load pre-trained DenseNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(self.feature_dim)
        
        # Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.use_attention:
            features = self.cbam(features)
        
        avg_pool = self.global_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        combined_features = torch.cat([avg_pool, max_pool], dim=1)
        
        output = self.classifier(combined_features)
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EnhancedResNet(nn.Module):
    """Enhanced ResNet with attention and custom head"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.4,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(self.feature_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.use_attention:
            features = self.cbam(features)
        
        avg_pool = self.global_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        combined_features = torch.cat([avg_pool, max_pool], dim=1)
        
        output = self.classifier(combined_features)
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EnsembleModel(nn.Module):
    """Ensemble model combining multiple architectures"""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        fusion_method: str = "weighted_average"  # weighted_average, max, learned
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.fusion_method = fusion_method
        
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            self.weights = torch.tensor(weights)
        
        # Learned fusion
        if fusion_method == "learned":
            num_classes = models[0].classifier[-1].out_features
            self.fusion_layer = nn.Sequential(
                nn.Linear(num_classes * self.num_models, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        self.weights = nn.Parameter(self.weights, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for model in self.models:
            with torch.set_grad_enabled(model.training):
                output = model(x)
                outputs.append(output)
        
        if self.fusion_method == "weighted_average":
            # Weighted average
            stacked_outputs = torch.stack(outputs, dim=0)
            weights = self.weights.view(-1, 1, 1).to(x.device)
            ensemble_output = torch.sum(stacked_outputs * weights, dim=0)
            
        elif self.fusion_method == "max":
            # Max fusion
            stacked_outputs = torch.stack(outputs, dim=0)
            ensemble_output, _ = torch.max(stacked_outputs, dim=0)
            
        elif self.fusion_method == "learned":
            # Learned fusion
            concatenated = torch.cat(outputs, dim=1)
            ensemble_output = self.fusion_layer(concatenated)
        else:
            # Default to weighted average
            stacked_outputs = torch.stack(outputs, dim=0)
            weights = self.weights.view(-1, 1, 1).to(x.device)
            ensemble_output = torch.sum(stacked_outputs * weights, dim=0)
        
        return ensemble_output
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models"""
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
                outputs.append(torch.softmax(output, dim=1))
        return outputs


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """Factory function to create models"""
    
    if architecture.startswith("efficientnet"):
        return EnhancedEfficientNet(
            model_name=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif architecture.startswith("densenet"):
        return EnhancedDenseNet(
            model_name=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif architecture.startswith("resnet"):
        return EnhancedResNet(
            model_name=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def create_ensemble_model(
    architectures: List[str],
    num_classes: int,
    model_paths: Optional[List[str]] = None,
    **kwargs
) -> EnsembleModel:
    """Create ensemble model from multiple architectures"""
    
    models = []
    for i, arch in enumerate(architectures):
        model = create_model(arch, num_classes, **kwargs)
        
        # Load pretrained weights if provided
        if model_paths and i < len(model_paths) and model_paths[i]:
            try:
                state_dict = torch.load(model_paths[i], map_location='cpu')
                model.load_state_dict(state_dict)
                logging.info(f"Loaded weights for {arch} from {model_paths[i]}")
            except Exception as e:
                logging.warning(f"Failed to load weights for {arch}: {e}")
        
        models.append(model)
    
    return EnsembleModel(models)


if __name__ == "__main__":
    # Test model creation
    config = get_config()
    
    # Test individual models
    print("Testing individual models...")
    efficientnet = create_model("efficientnet_b4", num_classes=2)
    densenet = create_model("densenet121", num_classes=2)
    resnet = create_model("resnet50", num_classes=2)
    
    # Test ensemble
    print("Testing ensemble model...")
    ensemble = create_ensemble_model(
        ["efficientnet_b4", "densenet121", "resnet50"],
        num_classes=2
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 384, 384)
    
    print(f"EfficientNet output shape: {efficientnet(dummy_input).shape}")
    print(f"DenseNet output shape: {densenet(dummy_input).shape}")
    print(f"ResNet output shape: {resnet(dummy_input).shape}")
    print(f"Ensemble output shape: {ensemble(dummy_input).shape}")
    
    print("✅ All models created successfully!")