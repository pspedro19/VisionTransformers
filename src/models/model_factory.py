"""
Model Factory for Video Transformer models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Try to import transformers with error handling
try:
    from transformers import (
        VideoMAEForVideoClassification,
        TimesformerForVideoClassification,
        AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AttentionVideoModel(nn.Module):
    """
    A simplified video model that can extract attention maps.
    This is a fallback when transformers are not available or for testing.
    """
    
    def __init__(self, input_size: Tuple[int, int, int] = (16, 224, 224)):
        super().__init__()
        self.input_size = input_size
        
        # Simple CNN backbone for feature extraction
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1000)  # 1000 classes for ImageNet-style classification
        )
        
    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False):
        """
        Forward pass with optional attention and hidden states output.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, frames, height, width)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with outputs, attentions, and hidden states
        """
        batch_size, channels, frames, height, width = pixel_values.shape
        
        # Reshape for 3D convolution: (B, C, T, H, W)
        x = pixel_values
        
        # Extract features
        features = self.features(x)  # (B, 256, 1, 7, 7)
        
        # Reshape for attention: (B, 256, 49) -> (B, 49, 256)
        features_flat = features.view(batch_size, 256, -1).transpose(1, 2)
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(features_flat, features_flat, features_flat)
        
        # Global average pooling
        pooled = attn_output.mean(dim=1)  # (B, 256)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'last_hidden_state': attn_output
        }
        
        if output_attentions:
            outputs['attentions'] = [attn_weights]
            
        if output_hidden_states:
            outputs['hidden_states'] = [features_flat, attn_output]
            
        return outputs


class ModelFactory:
    """
    Factory pattern for creating and managing video transformer models.
    """
    
    SUPPORTED_MODELS = {
        "videomae-base": {
            "class": VideoMAEForVideoClassification if TRANSFORMERS_AVAILABLE else AttentionVideoModel,
            "checkpoint": "MCG-NJU/videomae-base" if TRANSFORMERS_AVAILABLE else None,
            "input_size": (16, 224, 224),  # (frames, H, W)
            "description": "Video Masked Autoencoder - Good for general video understanding"
        },
        "timesformer-base": {
            "class": TimesformerForVideoClassification if TRANSFORMERS_AVAILABLE else AttentionVideoModel,
            "checkpoint": "facebook/timesformer-base-finetuned-k400" if TRANSFORMERS_AVAILABLE else None,
            "input_size": (8, 224, 224),
            "description": "TimeSformer - Efficient spatio-temporal attention"
        },
        "videomae-large": {
            "class": VideoMAEForVideoClassification if TRANSFORMERS_AVAILABLE else AttentionVideoModel,
            "checkpoint": "MCG-NJU/videomae-large" if TRANSFORMERS_AVAILABLE else None,
            "input_size": (16, 224, 224),
            "description": "Larger VideoMAE model - Better quality, slower inference"
        },
        "videomae-huge": {
            "class": VideoMAEForVideoClassification if TRANSFORMERS_AVAILABLE else AttentionVideoModel,
            "checkpoint": "MCG-NJU/videomae-huge" if TRANSFORMERS_AVAILABLE else None,
            "input_size": (16, 224, 224),
            "description": "Huge VideoMAE model - Best quality, slowest inference"
        },
        "efficientvit": {
            "class": None,  # Placeholder, implement EfficientViT
            "checkpoint": None,
            "input_size": (16, 224, 224),
            "description": "EfficientViT - Optimizado para velocidad y memoria"
        },
        "mobilevit": {
            "class": None,  # Placeholder, implement MobileViT
            "checkpoint": None,
            "input_size": (8, 224, 224),
            "description": "MobileViT - Optimizado para dispositivos móviles y edge"
        },
        "attention-video": {
            "class": AttentionVideoModel,
            "checkpoint": None,
            "input_size": (16, 224, 224),
            "description": "Custom attention model - Fallback when transformers unavailable"
        }
    }
    
    def __init__(self):
        """Initialize model factory."""
        self._model_cache = {}
        
    def create_model(self, model_name: str, device: str = "cuda", **kwargs) -> torch.nn.Module:
        """
        Create and initialize a video model.
        
        Args:
            model_name: Name of the model to create
            device: Device to load model on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Initialized model ready for inference
            
        Raises:
            ValueError: If model name is not supported
        """
        if model_name not in self.SUPPORTED_MODELS:
            available_models = list(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Model '{model_name}' not supported. Available models: {available_models}"
            )
        
        # Check cache first
        cache_key = f"{model_name}_{device}"
        if cache_key in self._model_cache:
            logger.info(f"Loading model '{model_name}' from cache")
            return self._model_cache[cache_key]
        
        config = self.SUPPORTED_MODELS[model_name]
        logger.info(f"Loading model '{model_name}' ({config['description']})")
        
        try:
            if TRANSFORMERS_AVAILABLE and config["checkpoint"]:
                # Use transformers model
                model_class = config["class"]
                model = model_class.from_pretrained(config["checkpoint"])
            else:
                # Use custom attention model
                model = AttentionVideoModel(config["input_size"])
                logger.info("Using custom attention model (transformers not available)")
            
            # Move to device
            model = model.to(device)
            
            # Set to evaluation mode
            model.eval()
            
            # Cache the model
            self._model_cache[cache_key] = model
            
            logger.info(f"Model '{model_name}' loaded successfully on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            # Fallback to custom model
            logger.info("Falling back to custom attention model")
            model = AttentionVideoModel(config["input_size"])
            model = model.to(device)
            model.eval()
            self._model_cache[cache_key] = model
            return model
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported")
        
        config = self.SUPPORTED_MODELS[model_name].copy()
        
        # Add estimated memory usage
        memory_estimates = {
            "videomae-base": {"gpu": "2-3 GB", "cpu": "1-2 GB"},
            "videomae-large": {"gpu": "4-6 GB", "cpu": "2-4 GB"},
            "videomae-huge": {"gpu": "8-12 GB", "cpu": "4-8 GB"},
            "timesformer-base": {"gpu": "2-4 GB", "cpu": "1-3 GB"}
        }
        
        config["estimated_memory"] = memory_estimates.get(
            model_name, 
            {"gpu": "Unknown", "cpu": "Unknown"}
        )
        
        return config
    
    def list_available_models(self) -> Dict[str, str]:
        """
        List all available models with descriptions.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return {
            name: config["description"] 
            for name, config in self.SUPPORTED_MODELS.items()
        }
    
    def recommend_model(self, 
                       video_duration: float, 
                       target_latency: float = 30.0,
                       available_memory_gb: float = 4.0) -> str:
        """
        Recommend optimal model based on requirements.
        
        Args:
            video_duration: Duration of input video in seconds
            target_latency: Target processing latency in seconds
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Recommended model name
        """
        # Simple heuristic-based recommendation
        if available_memory_gb < 3:
            return "videomae-base"
        elif video_duration < 10 and target_latency < 15:
            return "videomae-large" if available_memory_gb >= 6 else "videomae-base"
        elif target_latency < 10:
            return "timesformer-base"
        elif available_memory_gb >= 12:
            return "videomae-huge"
        else:
            return "videomae-base"
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        for model in self._model_cache.values():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        self._model_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache)
        } 