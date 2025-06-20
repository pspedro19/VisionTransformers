"""
GIF Composer with attention overlays and optimization.
"""

import logging
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Optional, Tuple
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class GifComposer:
    """
    Compose GIFs with attention overlays and optimizations.
    """
    
    def __init__(self):
        """Initialize GIF composer."""
        self.overlay_styles = {
            "heatmap": self._apply_heatmap_overlay,
            "highlight": self._apply_highlight_overlay,
            "glow": self._apply_glow_overlay,
            "pulse": self._apply_pulse_overlay,
            "transparent": self._apply_transparent_overlay
        }
        
    def create_gif(
        self,
        frames: torch.Tensor,
        attention_maps: torch.Tensor,
        output_path: str,
        fps: int = 10,
        duration_per_frame: Optional[int] = None,
        overlay_style: str = "transparent",
        overlay_intensity: float = 0.6,
        overlay_color: str = "blue",
        optimization_level: int = 1,
        maintain_duration: bool = True,
        quality: int = 95,
        loop: bool = True
    ) -> dict:
        """
        Create optimized GIF with attention overlays.
        
        Args:
            frames: Video frames tensor (T, H, W, C)
            attention_maps: Attention maps tensor (T, H, W)
            output_path: Output path for GIF
            fps: Frames per second for GIF
            duration_per_frame: Duration per frame in milliseconds (overrides fps)
            overlay_style: Style of attention overlay
            overlay_intensity: Intensity of overlay (0.0 to 1.0)
            overlay_color: Color for transparent overlay ('blue' or 'yellow')
            optimization_level: Optimization level (0-3)
            maintain_duration: Whether to maintain original video duration
            quality: GIF quality (1-100)
            loop: Whether GIF should loop
            
        Returns:
            Dictionary with generation statistics
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Convert tensors to numpy if needed
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(attention_maps, torch.Tensor):
            attention_maps = attention_maps.cpu().numpy()
            
        # Ensure frames are in correct format (T, H, W, C) with values 0-255
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
            
        # Normalize attention maps
        attention_maps = self._normalize_attention_maps(attention_maps)
        
        # Apply attention overlays
        overlaid_frames = self._apply_overlays(
            frames, 
            attention_maps, 
            overlay_style, 
            overlay_intensity,
            overlay_color
        )
        
        # Convert to PIL Images with high quality
        pil_frames = []
        for frame in overlaid_frames:
            img = Image.fromarray(frame)
            # Apply high-quality processing
            img = img.convert('RGB')  # Ensure RGB mode
            if quality < 100:  # Only if we want compression
                img = img.quantize(colors=256, method=2)  # High quality quantization
            pil_frames.append(img)
            
        # Calculate frame duration
        if duration_per_frame is None:
            if maintain_duration and len(frames) > 0:
                # Calculate duration to match original video length
                total_duration = (len(frames) * 1000) / fps  # Total duration in ms
                duration_per_frame = int(total_duration / len(frames))
            else:
                duration_per_frame = int(1000 / fps)  # Convert fps to milliseconds
            
        # Apply optimizations
        optimized_frames = self._optimize_frames(pil_frames, optimization_level)
        
        # Save GIF with high quality settings
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            optimized_frames[0].save(
                output_path,
                format='GIF',
                save_all=True,
                append_images=optimized_frames[1:],
                duration=duration_per_frame,
                loop=0 if loop else 1,
                optimize=True if optimization_level > 0 else False,
                quality=quality,  # Higher quality
                disposal=2  # Restore to background
            )
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                processing_time = 0.0
                
            # Calculate statistics
            file_size = output_path.stat().st_size
            compression_ratio = (len(frames) * frames[0].size * 3) / file_size
            
            stats = {
                "output_path": str(output_path),
                "total_frames": len(frames),
                "file_size_mb": file_size / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "processing_time": processing_time,
                "fps": fps,
                "frame_duration_ms": duration_per_frame,
                "overlay_style": overlay_style,
                "quality": quality
            }
            
            logger.info(f"GIF created: {output_path} ({file_size/1024/1024:.1f}MB, {len(frames)} frames)")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to create GIF: {e}")
            raise
    
    def _normalize_attention_maps(self, attention_maps: np.ndarray) -> np.ndarray:
        """
        Normalize attention maps to 0-1 range with better contrast.
        
        Args:
            attention_maps: Raw attention maps (T, H, W)
            
        Returns:
            Normalized attention maps
        """
        normalized = []
        
        for att_map in attention_maps:
            # Apply percentile-based normalization for better contrast
            p2, p98 = np.percentile(att_map, [2, 98])
            normalized_map = np.clip((att_map - p2) / (p98 - p2), 0, 1)
            
            # Apply slight gamma correction to enhance visibility
            normalized_map = np.power(normalized_map, 0.8)
            
            normalized.append(normalized_map)
            
        return np.array(normalized)
    
    def _apply_overlays(
        self,
        frames: np.ndarray,
        attention_maps: np.ndarray,
        style: str,
        intensity: float,
        overlay_color: str = "blue"
    ) -> np.ndarray:
        """
        Apply attention overlays to frames.
        
        Args:
            frames: Video frames (T, H, W, C)
            attention_maps: Attention maps (T, H, W)
            style: Overlay style
            intensity: Overlay intensity
            overlay_color: Color for transparent overlay ('blue' or 'yellow')
            
        Returns:
            Frames with overlays applied
        """
        if style not in self.overlay_styles:
            logger.warning(f"Unknown overlay style '{style}', using 'transparent'")
            style = "transparent"
            
        overlay_func = self.overlay_styles[style]
        overlaid_frames = []
        
        # Ensure attention maps match frame dimensions
        if attention_maps.shape[1:] != frames.shape[1:3]:
            resized_attention = []
            for att_map in attention_maps:
                resized = cv2.resize(att_map, (frames.shape[2], frames.shape[1]))
                resized_attention.append(resized)
            attention_maps = np.stack(resized_attention)
        
        for frame, att_map in zip(frames, attention_maps):
            if style == "transparent":
                overlaid_frame = overlay_func(frame, att_map, intensity, overlay_color)
            else:
                overlaid_frame = overlay_func(frame, att_map, intensity)
            overlaid_frames.append(overlaid_frame)
            
        return np.array(overlaid_frames)
    
    def _apply_heatmap_overlay(
        self, 
        frame: np.ndarray, 
        attention_map: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Apply heatmap-style overlay."""
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with original frame
        blended = cv2.addWeighted(frame, 1 - intensity, heatmap, intensity, 0)
        
        return blended
    
    def _apply_highlight_overlay(
        self, 
        frame: np.ndarray, 
        attention_map: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Apply highlight-style overlay (brighten important regions)."""
        # Create highlight mask
        highlight_mask = attention_map > np.percentile(attention_map, 80)
        
        # Brighten highlighted regions
        highlighted_frame = frame.copy().astype(np.float32)
        highlighted_frame[highlight_mask] *= (1 + intensity)
        highlighted_frame = np.clip(highlighted_frame, 0, 255).astype(np.uint8)
        
        return highlighted_frame
    
    def _apply_glow_overlay(
        self, 
        frame: np.ndarray, 
        attention_map: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Apply glow-style overlay."""
        # Create glow effect
        glow_mask = (attention_map * 255).astype(np.uint8)
        
        # Apply Gaussian blur for glow effect
        glow = cv2.GaussianBlur(glow_mask, (15, 15), 0)
        
        # Create colored glow (golden color)
        glow_colored = np.zeros_like(frame)
        glow_colored[:, :, 0] = glow * 0.3  # Blue
        glow_colored[:, :, 1] = glow * 0.8  # Green  
        glow_colored[:, :, 2] = glow       # Red (golden effect)
        
        # Blend with original frame
        blended = cv2.addWeighted(frame, 1.0, glow_colored, intensity, 0)
        
        return blended
    
    def _apply_pulse_overlay(
        self, 
        frame: np.ndarray, 
        attention_map: np.ndarray, 
        intensity: float
    ) -> np.ndarray:
        """Apply pulse-style overlay (for animated effects)."""
        # This is a simplified version - full pulse would require frame index
        threshold = np.percentile(attention_map, 90)
        pulse_mask = attention_map > threshold
        
        # Create pulse effect
        pulse_frame = frame.copy()
        pulse_frame[pulse_mask] = np.clip(
            pulse_frame[pulse_mask] * (1 + intensity * 0.5), 0, 255
        )
        
        return pulse_frame
    
    def _apply_transparent_overlay(
        self, 
        frame: np.ndarray, 
        attention_map: np.ndarray, 
        intensity: float,
        color: str = "blue"
    ) -> np.ndarray:
        """Apply transparent colored overlay for attention visualization."""
        # Ensure frame is in the correct format and range
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
            
        # Ensure attention map is in 0-1 range
        if attention_map.max() > 1.0:
            attention_map = attention_map / attention_map.max()
        attention_map = np.clip(attention_map, 0, 1)
        
        # Create the result frame (start with original)
        result = frame.copy().astype(np.float32)
        
        # Create color overlay based on attention
        if color.lower() == "blue":
            # Blue overlay: increase blue channel where attention is high
            result[:, :, 2] = result[:, :, 2] + (255 - result[:, :, 2]) * attention_map * intensity
        elif color.lower() == "yellow":
            # Yellow overlay: increase red and green channels
            result[:, :, 0] = result[:, :, 0] + (255 - result[:, :, 0]) * attention_map * intensity
            result[:, :, 1] = result[:, :, 1] + (255 - result[:, :, 1]) * attention_map * intensity
        else:
            # Default to blue
            result[:, :, 2] = result[:, :, 2] + (255 - result[:, :, 2]) * attention_map * intensity
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _optimize_frames(
        self, 
        frames: List[Image.Image], 
        optimization_level: int
    ) -> List[Image.Image]:
        """
        Optimize frames for better compression.
        
        Args:
            frames: List of PIL Images
            optimization_level: Optimization level (0-3)
            
        Returns:
            Optimized frames
        """
        if optimization_level == 0:
            return frames
            
        optimized = []
        
        for frame in frames:
            if optimization_level >= 1:
                # Reduce colors using palette
                frame = frame.quantize(colors=256, method=Image.ADAPTIVE)
                
            if optimization_level >= 2:
                # Apply slight blur to reduce noise
                frame = frame.filter(ImageFilter.GaussianBlur(radius=0.5))
                
            if optimization_level >= 3:
                # More aggressive color reduction
                frame = frame.quantize(colors=128, method=Image.ADAPTIVE)
                
            optimized.append(frame)
            
        return optimized
    
    def create_preview_grid(
        self,
        frames: torch.Tensor,
        attention_maps: torch.Tensor,
        output_path: str,
        grid_size: Tuple[int, int] = (2, 4)
    ) -> str:
        """
        Create a preview grid showing original and attention frames.
        
        Args:
            frames: Video frames
            attention_maps: Attention maps
            output_path: Output path for preview image
            grid_size: Grid size (rows, cols)
            
        Returns:
            Path to created preview image
        """
        rows, cols = grid_size
        total_slots = rows * cols
        
        # Select evenly spaced frames
        frame_indices = np.linspace(0, len(frames)-1, total_slots//2, dtype=int)
        
        # Convert tensors to numpy
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(attention_maps, torch.Tensor):
            attention_maps = attention_maps.cpu().numpy()
            
        # Prepare frames
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
            
        # Normalize attention maps
        attention_maps = self._normalize_attention_maps(attention_maps)
        
        # Create grid
        frame_h, frame_w = frames.shape[1:3]
        grid_img = Image.new('RGB', (cols * frame_w, rows * frame_h), 'white')
        
        for i, frame_idx in enumerate(frame_indices):
            row = (i * 2) // cols
            col = (i * 2) % cols
            
            # Original frame
            orig_frame = Image.fromarray(frames[frame_idx])
            grid_img.paste(orig_frame, (col * frame_w, row * frame_h))
            
            # Attention overlay
            if col + 1 < cols:
                att_frame = self._apply_heatmap_overlay(
                    frames[frame_idx], attention_maps[frame_idx], 0.7
                )
                att_image = Image.fromarray(att_frame)
                grid_img.paste(att_image, ((col + 1) * frame_w, row * frame_h))
                
        # Save preview
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_img.save(output_path, 'JPEG', quality=90)
        
        logger.info(f"Preview grid saved: {output_path}")
        return str(output_path) 