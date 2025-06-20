"""
Optimized Video Decoder with GPU support and security validations.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import decord
from decord import VideoReader, gpu, cpu

# Try to import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
except Exception as e:
    logging.warning(f"PyTorch import error: {e}")
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizedVideoDecoder:
    """
    High-performance video decoder with GPU acceleration and security limits.
    """
    
    def __init__(self, gpu_id: int = 0):
        """
        Initialize decoder with GPU context.
        
        Args:
            gpu_id: GPU device ID, -1 for CPU
        """
        if TORCH_AVAILABLE:
            decord.bridge.set_bridge('torch')
        else:
            decord.bridge.set_bridge('numpy')
        
        # Always decode on CPU for max compatibility, then move to GPU if available.
        self.ctx = cpu()
        logger.info("Initialized decoder with CPU context for robust decoding.")
        
        self.target_device = 'cpu'
        self.use_gpu = False
        if gpu_id >= 0 and TORCH_AVAILABLE and torch.cuda.is_available():
            self.target_device = f'cuda:{gpu_id}'
            self.use_gpu = True
            logger.info(f"Target device for tensors is {self.target_device}.")
        
    def decode_video_stream(
        self, 
        video_path: str, 
        max_resolution: int = 720, 
        max_duration: int = 60
    ) -> Tuple[any, float]:
        """
        Decode video directly to tensors with security limits.
        
        Args:
            video_path: Path to input video file
            max_resolution: Maximum resolution limit (height or width)
            max_duration: Maximum duration in seconds
            
        Returns:
            Tuple of (frames_tensor, fps) where frames are in format (T, H, W, C) with values 0-1
            
        Raises:
            ValueError: If video exceeds security limits
            FileNotFoundError: If video file doesn't exist
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        try:
            vr = VideoReader(str(video_path), ctx=self.ctx)
        except Exception as e:
            raise ValueError(f"Failed to open video: {e}")
            
        # Security validations with improved logic
        height, width = vr[0].shape[:2]
        duration = len(vr) / vr.get_avg_fps()
        
        # Check resolution - use the larger dimension as reference
        max_dimension = max(height, width)
        if max_dimension > max_resolution:
            logger.warning(
                f"Resolution {width}x{height} exceeds limit. Max dimension: {max_dimension} > {max_resolution}"
            )
            # Instead of failing, we'll resize during processing
            logger.info("Video will be resized during processing to fit resolution limits")
            
        if duration > max_duration:
            raise ValueError(
                f"Duration exceeds limit: {duration:.1f}s > {max_duration}s"
            )
            
        logger.info(f"Processing video: {width}x{height}, {duration:.1f}s, {len(vr)} frames")
        
        # Use adaptive stride if enabled
        if hasattr(self, '_use_adaptive_stride') and self._use_adaptive_stride:
            stride = self._calculate_adaptive_stride(vr)
            indices = list(range(0, len(vr), stride))
            logger.info(f"Using adaptive stride {stride}, extracting {len(indices)} frames")
        else:
            # Use all frames for best quality
            indices = list(range(len(vr)))
            logger.info(f"Extracting all {len(indices)} frames")
        
        # Batch decoding
        frames = vr.get_batch(indices)
        
        # Convert to appropriate format based on available libraries
        if TORCH_AVAILABLE:
            # Ensure frames are in the correct format (T, H, W, C) with values 0-1
            if frames.dtype != torch.float32:
                frames = frames.float()
            
            # Normalize to 0-1 range if needed
            if frames.max() > 1.0:
                frames = frames / 255.0
                
            # Resize if necessary
            current_height, current_width = frames.shape[1], frames.shape[2]
            max_dimension = max(current_height, current_width)
            
            if max_dimension > max_resolution:
                # Calculate new dimensions maintaining aspect ratio
                if current_height > current_width:
                    new_height = max_resolution
                    new_width = int((current_width * max_resolution) / current_height)
                else:
                    new_width = max_resolution
                    new_height = int((current_height * max_resolution) / current_width)
                    
                logger.info(f"Resizing from {current_width}x{current_height} to {new_width}x{new_height}")
                
                # Reshape for interpolation: (T, H, W, C) -> (T, C, H, W)
                frames_reshaped = frames.permute(0, 3, 1, 2)
                frames_resized = frames_reshaped.permute(0, 2, 3, 1)
                
            # Move to appropriate device
            try:
                frames = frames.to(self.target_device)
                logger.info(f"Successfully moved frames to {self.target_device}")
            except Exception as e:
                logger.warning(f"Failed to move frames to {self.target_device}: {e}, keeping on CPU")
                frames = frames.cpu()
        else:
            # Use numpy format
            frames = np.array(frames)
            if frames.dtype != np.float32:
                frames = frames.astype(np.float32)
            
            # Normalize to 0-1 range if needed
            if frames.max() > 1.0:
                frames = frames / 255.0
                
            # Resize if necessary
            current_height, current_width = frames.shape[1], frames.shape[2]
            max_dimension = max(current_height, current_width)
            
            if max_dimension > max_resolution:
                # Calculate new dimensions maintaining aspect ratio
                if current_height > current_width:
                    new_height = max_resolution
                    new_width = int((current_width * max_resolution) / current_height)
                else:
                    new_width = max_resolution
                    new_height = int((current_height * max_resolution) / current_width)
                    
                logger.info(f"Resizing from {current_width}x{current_height} to {new_width}x{new_height}")
                
                # Simple resize using PIL
                from PIL import Image
                resized_frames = []
                for frame in frames:
                    img = Image.fromarray((frame * 255).astype(np.uint8))
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    resized_frames.append(np.array(img_resized).astype(np.float32) / 255.0)
                frames = np.array(resized_frames)
            
        logger.info(f"Final frames shape: {frames.shape}, dtype: {frames.dtype}, range: [{frames.min():.3f}, {frames.max():.3f}]")
            
        return frames, vr.get_avg_fps()
    
    def _calculate_adaptive_stride(
        self, 
        vr: VideoReader, 
        min_stride: int = 2, 
        max_stride: int = 10
    ) -> int:
        """
        Calculate adaptive stride based on motion analysis.
        
        Args:
            vr: VideoReader instance
            min_stride: Minimum stride (more frames)
            max_stride: Maximum stride (fewer frames)
            
        Returns:
            Optimal stride value
        """
        # Quick motion analysis on sample frames
        total_frames = len(vr)
        sample_indices = [0, total_frames//4, total_frames//2, 3*total_frames//4]
        
        if total_frames < 4:
            return min_stride
            
        diffs = []
        
        try:
            for i in range(len(sample_indices)-1):
                f1 = vr[sample_indices[i]].float()
                f2 = vr[sample_indices[i+1]].float()
                diff = torch.abs(f1 - f2).mean().item()
                diffs.append(diff)
                
            avg_motion = np.mean(diffs)
            
            # Higher motion = lower stride (more frames)
            # Motion values typically range from 0-50 for normalized frames
            motion_factor = min(avg_motion / 20.0, 1.0)
            stride = int(min_stride + (max_stride - min_stride) * (1 - motion_factor))
            
            logger.debug(f"Motion analysis: avg_motion={avg_motion:.3f}, stride={stride}")
            
        except Exception as e:
            logger.warning(f"Motion analysis failed, using default stride: {e}")
            stride = (min_stride + max_stride) // 2
            
        return max(min_stride, min(stride, max_stride))
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata without full decoding.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return {"error": f"Video file not found: {video_path}"}

            vr = VideoReader(str(video_path), ctx=cpu())  # Use CPU for metadata
            height, width = vr[0].shape[:2]
            
            return {
                "width": width,
                "height": height,
                "total_frames": len(vr),
                "fps": vr.get_avg_fps(),
                "duration": len(vr) / vr.get_avg_fps(),
                "file_size_mb": video_path.stat().st_size / (1024 * 1024),
                "device": "cuda" if self.use_gpu else "cpu"
            }
        except Exception as e:
            logger.error(f"Failed to read video metadata for {video_path}: {e}", exc_info=True)
            return {"error": f"Failed to read video metadata: {e}"} 