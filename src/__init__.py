"""
ViT-GIF Highlight: Generador Inteligente de GIFs con Atención Visual desde Videos

Sistema modular que transforma videos cortos en GIFs inteligentes destacando 
automáticamente las regiones más importantes usando Vision Transformers específicos para video.
"""

__version__ = "2.0.0"
__author__ = "ViT-GIF Team"
__description__ = "Generador Inteligente de GIFs con Atención Visual desde Videos"

# Core imports
from .core.pipeline import InMemoryPipeline
from .core.video_decoder import OptimizedVideoDecoder
from .core.attention_engine import VideoAttentionEngine
from .core.gif_composer import GifComposer

# Model imports
from .models.model_factory import ModelFactory

__all__ = [
    "InMemoryPipeline",
    "OptimizedVideoDecoder", 
    "VideoAttentionEngine",
    "GifComposer",
    "ModelFactory",
]

# Quick access API
def process_video(video_path: str, output_path: str, config_path: str = "config/mvp1.yaml", **kwargs):
    """
    Quick function to process a video to GIF.
    
    Args:
        video_path: Path to input video
        output_path: Path for output GIF
        config_path: Configuration file path
        **kwargs: Override configuration options
        
    Returns:
        Processing results dictionary
    """
    pipeline = InMemoryPipeline(config_path)
    return pipeline.process_video(video_path, output_path, kwargs if kwargs else None)


def get_available_models():
    """
    Get list of available models.
    
    Returns:
        Dictionary of model names and descriptions
    """
    factory = ModelFactory()
    return factory.list_available_models()


# Package metadata
PACKAGE_INFO = {
    "name": "vit-gif-highlight",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "license": "MIT",
    "python_requires": ">=3.9",
    "homepage": "https://github.com/your-org/vit-gif-highlight",
    "documentation": "https://vit-gif-highlight.readthedocs.io",
    "repository": "https://github.com/your-org/vit-gif-highlight.git",
    "keywords": [
        "computer-vision",
        "video-processing", 
        "gif-generation",
        "attention-mechanisms",
        "transformers",
        "pytorch",
        "marketing",
        "social-media"
    ]
} 