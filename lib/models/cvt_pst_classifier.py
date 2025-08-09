"""
CvT with Pyramid Sparse Transformer (PST) Wrapper
==================================================

This module implements a wrapper around the original CvT backbone with an injected
Pyramid Sparse Transformer (PST) module for enhanced multi-scale feature processing.

The PST module is inserted between the CvT backbone and the classification head,
performing multi-scale feature fusion using:
1. Multi-scale downsampling with AvgPool2d
2. Depthwise + pointwise convolutions for each scale
3. Feature concatenation and fusion
4. Output to classification head for 10 paddy disease classes

Author: CvT-x-PST Project
Compatible with: Google Colab, PyTorch >= 1.8.0
Dependencies: torch, torchvision, einops (available in environment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import logging

from .cls_cvt import ConvolutionalVisionTransformer


class PyramidSparseTransformer(nn.Module):
    """
    Pyramid Sparse Transformer (PST) Module
    
    Performs multi-scale feature fusion using depthwise separable convolutions
    across multiple pyramid scales before classification.
    
    Args:
        input_dim (int): Input feature dimension from CvT backbone
        scales (list): List of pooling scales for pyramid levels
        reduction_ratio (int): Channel reduction ratio for efficiency
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, 
                 input_dim=768, 
                 scales=[1, 2, 4, 8], 
                 reduction_ratio=4,
                 dropout=0.1):
        super(PyramidSparseTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.scales = scales
        self.num_scales = len(scales)
        
        # Reduced dimension for efficiency
        self.reduced_dim = input_dim // reduction_ratio
        
        # Input projection to reduce channels
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.reduced_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale processing modules
        self.scale_processors = nn.ModuleList()
        for scale in scales:
            # Depthwise + Pointwise convolution for each scale
            scale_module = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(self.reduced_dim, self.reduced_dim, 
                         kernel_size=3, padding=1, groups=self.reduced_dim, bias=False),
                nn.BatchNorm2d(self.reduced_dim),
                nn.GELU(),
                
                # Pointwise convolution
                nn.Conv2d(self.reduced_dim, self.reduced_dim, 
                         kernel_size=1, bias=False),
                nn.BatchNorm2d(self.reduced_dim),
                nn.GELU(),
                
                # Optional dropout
                nn.Dropout2d(dropout)
            )
            self.scale_processors.append(scale_module)
        
        # Feature fusion after concatenation
        concat_dim = self.reduced_dim * self.num_scales
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of PST module
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W] from CvT backbone
            
        Returns:
            torch.Tensor: Enhanced features [B, C, H, W] with same spatial dimensions
        """
        B, C, H, W = x.shape
        
        # Convert to patch tokens for linear projection
        x_tokens = rearrange(x, 'b c h w -> b (h w) c')
        
        # Reduce channels for efficiency
        x_reduced = self.input_proj(x_tokens)  # [B, H*W, reduced_dim]
        
        # Reshape back to spatial format
        x_spatial = rearrange(x_reduced, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Multi-scale processing
        scale_features = []
        
        for i, (scale, processor) in enumerate(zip(self.scales, self.scale_processors)):
            if scale == 1:
                # Original scale - no pooling
                scaled_x = x_spatial
            else:
                # Downsample using adaptive average pooling
                pool_size = (H // scale, W // scale)
                pool_size = (max(1, pool_size[0]), max(1, pool_size[1]))  # Ensure positive
                scaled_x = F.adaptive_avg_pool2d(x_spatial, pool_size)
            
            # Process with depthwise + pointwise convolutions
            processed = processor(scaled_x)
            
            # Upsample back to original size if needed
            if processed.shape[2:] != (H, W):
                processed = F.interpolate(processed, size=(H, W), 
                                        mode='bilinear', align_corners=False)
            
            # Convert to tokens for concatenation
            processed_tokens = rearrange(processed, 'b c h w -> b (h w) c')
            scale_features.append(processed_tokens)
        
        # Concatenate multi-scale features
        concat_features = torch.cat(scale_features, dim=-1)  # [B, H*W, concat_dim]
        
        # Fuse concatenated features
        fused_features = self.fusion(concat_features)  # [B, H*W, input_dim]
        
        # Residual connection with original features
        original_tokens = rearrange(x, 'b c h w -> b (h w) c')
        enhanced_tokens = original_tokens + self.residual_weight * fused_features
        
        # Convert back to spatial format
        enhanced_spatial = rearrange(enhanced_tokens, 'b (h w) c -> b c h w', h=H, w=W)
        
        return enhanced_spatial


class CvT_PST_Classifier(nn.Module):
    """
    CvT with Pyramid Sparse Transformer Classifier
    
    This is a wrapper around the original CvT backbone that injects a PST module
    between the backbone feature extraction and the classification head.
    
    The PST module enhances multi-scale spatial reasoning without modifying
    the original CvT architecture.
    
    Args:
        cvt_config: Configuration for the CvT backbone
        num_classes (int): Number of output classes (default: 10 for paddy diseases)
        pst_scales (list): Pyramid scales for PST module
        pst_reduction_ratio (int): Channel reduction ratio for PST
        pst_dropout (float): Dropout rate for PST module
        pretrained_path (str): Path to pretrained CvT weights
    """
    
    def __init__(self, 
                 cvt_config,
                 num_classes=10,  # 10 paddy disease classes
                 pst_scales=[1, 2, 4, 8],
                 pst_reduction_ratio=4,
                 pst_dropout=0.1,
                 pretrained_path=None):
        super(CvT_PST_Classifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize CvT backbone (without classification head)
        self.backbone = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=0,  # No classification head in backbone
            spec=cvt_config.MODEL.SPEC
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained_backbone(pretrained_path)
        
        # Get backbone output dimension
        self.backbone_dim = cvt_config.MODEL.SPEC['DIM_EMBED'][-1]
        
        # Pyramid Sparse Transformer module
        self.pst_module = PyramidSparseTransformer(
            input_dim=self.backbone_dim,
            scales=pst_scales,
            reduction_ratio=pst_reduction_ratio,
            dropout=pst_dropout
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head for paddy diseases
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone_dim, num_classes)
        )
        
        # Initialize classification head
        self._init_classifier()
        
        logging.info(f"CvT-PST Classifier initialized:")
        logging.info(f"  - Backbone dim: {self.backbone_dim}")
        logging.info(f"  - PST scales: {pst_scales}")
        logging.info(f"  - Num classes: {num_classes}")
    
    def _load_pretrained_backbone(self, pretrained_path):
        """Load pretrained CvT weights into backbone"""
        try:
            # Load pretrained weights
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Get backbone state dict
            backbone_dict = self.backbone.state_dict()
            
            # Filter pretrained dict to match backbone
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # Skip classification head weights
                if k.startswith('head.'):
                    continue
                    
                if k in backbone_dict and v.shape == backbone_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    logging.warning(f"Skipping {k}: shape mismatch or not found")
            
            # Load filtered weights
            self.backbone.load_state_dict(filtered_dict, strict=False)
            logging.info(f"Loaded pretrained backbone weights from {pretrained_path}")
            
        except Exception as e:
            logging.error(f"Failed to load pretrained weights: {e}")
    
    def _init_classifier(self):
        """Initialize classification head weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """
        Extract features using CvT backbone + PST enhancement
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Enhanced features [B, C, H', W']
        """
        # Pass through CvT backbone stages
        for i in range(self.backbone.num_stages):
            x, cls_tokens = getattr(self.backbone, f'stage{i}')(x)
        
        # Apply PST module for multi-scale enhancement
        enhanced_features = self.pst_module(x)
        
        return enhanced_features
    
    def forward(self, x):
        """
        Full forward pass: backbone -> PST -> classification
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Extract enhanced features
        features = self.forward_features(x)  # [B, C, H, W]
        
        # Global average pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        
        Args:
            x (torch.Tensor): Input images [B, 3, H, W]
            
        Returns:
            dict: Dictionary containing feature maps at different stages
        """
        features = {}
        
        # Backbone features
        current_x = x
        for i in range(self.backbone.num_stages):
            current_x, cls_tokens = getattr(self.backbone, f'stage{i}')(current_x)
            features[f'stage_{i}'] = current_x
        
        # PST enhanced features
        enhanced = self.pst_module(current_x)
        features['pst_enhanced'] = enhanced
        
        return features
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning only PST and classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logging.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for full model training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logging.info("Backbone parameters unfrozen")


def create_cvt_pst_classifier(config, 
                             num_classes=10,
                             pst_scales=[1, 2, 4, 8],
                             pst_reduction_ratio=4,
                             pst_dropout=0.1,
                             pretrained_path=None):
    """
    Factory function to create CvT-PST classifier
    
    Args:
        config: Configuration object with MODEL.SPEC
        num_classes (int): Number of output classes
        pst_scales (list): Pyramid scales for PST
        pst_reduction_ratio (int): Channel reduction for PST
        pst_dropout (float): Dropout rate for PST
        pretrained_path (str): Path to pretrained weights
    
    Returns:
        CvT_PST_Classifier: Initialized model
    """
    model = CvT_PST_Classifier(
        cvt_config=config,
        num_classes=num_classes,
        pst_scales=pst_scales,
        pst_reduction_ratio=pst_reduction_ratio,
        pst_dropout=pst_dropout,
        pretrained_path=pretrained_path
    )
    
    return model


# Paddy disease class names for reference
PADDY_DISEASE_CLASSES = [
    'Bacterial_leaf_blight',
    'Brown_spot', 
    'Leaf_smut',
    'Normal',
    'Blast',
    'Dead_heart',
    'Downy_mildew',
    'Hispa', 
    'Tungro',
    'Rice_bug'
]


if __name__ == "__main__":
    # Example usage and testing
    import torch
    from types import SimpleNamespace
    
    # Mock config for testing
    mock_spec = {
        'NUM_STAGES': 3,
        'DIM_EMBED': [64, 192, 768],
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DEPTH': [1, 2, 10],
        'NUM_HEADS': [1, 3, 12],
        'MLP_RATIO': [4, 4, 4],
        'QKV_BIAS': [True, True, True],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'CLS_TOKEN': [False, False, False],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_Q': [1, 1, 1],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [2, 2, 2],
        'STRIDE_Q': [1, 1, 1],
    }
    
    mock_config = SimpleNamespace()
    mock_config.MODEL = SimpleNamespace()
    mock_config.MODEL.SPEC = mock_spec
    
    # Test model creation
    model = create_cvt_pst_classifier(
        config=mock_config,
        num_classes=10,
        pst_scales=[1, 2, 4],
        pst_reduction_ratio=4
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("CvT-PST Classifier test completed successfully!")
