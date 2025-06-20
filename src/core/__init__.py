"""
Core processing modules for ViT-GIF Highlight.
"""

from .pipeline import InMemoryPipeline
from .video_decoder import OptimizedVideoDecoder
from .attention_engine import VideoAttentionEngine
from .gif_composer import GifComposer

__all__ = [
    "InMemoryPipeline",
    "OptimizedVideoDecoder",
    "VideoAttentionEngine", 
    "GifComposer",
] 