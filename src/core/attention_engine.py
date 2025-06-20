"""
Video Attention Engine using Video-specific Transformers.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

# Try to import torch with error handling
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
except Exception as e:
    logging.warning(f"PyTorch import error: {e}")
    TORCH_AVAILABLE = False

# Try to import transformers with error handling
try:
    from transformers import (
        VideoMAEForVideoClassification,
        TimesformerForVideoClassification,
        AutoProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
except Exception as e:
    logging.warning(f"Transformers import error: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VideoAttentionEngine:
    """
    Extract attention maps from video-specific transformer models.
    """
    
    def __init__(self, model_name: str = "videomae-base", device: str = "auto"):
        """
        Initialize video attention engine.
        
        Args:
            model_name: Name of the video model to use
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Auto-detect device if needed
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")
            
        self.device = device
        self.model_name = model_name
        
        # Import model factory
        from ..models.model_factory import ModelFactory
        
        try:
            self.model_factory = ModelFactory()
            logger.info(f"Loading model {model_name} on {device}")
            self.model = self.model_factory.create_model(model_name, device)
            self.processor = self._get_processor(model_name)
            
            logger.info(f"Initialized attention engine with {model_name} on {device}")
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}, using fallback attention")
            self.model = None
            self.processor = None
        
    def _get_processor(self, model_name: str):
        """Get appropriate processor for the model."""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            if "videomae" in model_name.lower():
                return AutoProcessor.from_pretrained("MCG-NJU/videomae-base")
            elif "timesformer" in model_name.lower():
                return AutoProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
            else:
                return AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load processor: {e}, using default preprocessing")
            return None
    
    def extract_attention_and_track(self, frames_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extracts attention, finds the main object, and tracks it across frames.

        Args:
            frames_tensor: Video frames tensor of shape (T, H, W, C).

        Returns:
            A tuple containing:
            - Attention maps tensor of shape (T, H, W).
            - Trajectory list of (x, y) coordinates for each frame.
        """
        if self.model is None or self.processor is None:
            logger.warning("Model or processor not available, using fallback saliency.")
            saliency_maps = self._fallback_saliency(frames_tensor)
            return saliency_maps, []

        logger.info("Starting attention extraction and object tracking.")
        
        # 1. Preprocess frames and get model outputs (in batches to save memory)
        attentions, features = self._get_model_outputs_batched(frames_tensor, batch_size=16)
        if attentions is None or features is None:
            # Fallback to simple saliency if model processing fails
            return self._fallback_saliency(frames_tensor), []

        # attentions shape: (T, num_heads, patch_h*patch_w+1, patch_h*patch_w+1)
        # features shape: (T, patch_h*patch_w, feature_dim)
        num_frames = features.shape[0]
        num_patches = features.shape[1]
        patch_h = patch_w = int(np.sqrt(num_patches))

        # 2. Find the initial target in the first frame
        # We use the attention from the [CLS] token to all other patches
        initial_cls_attention = attentions[0].mean(dim=0)[0, 1:]  # Avg heads, get CLS row, skip CLS-to-CLS
        
        target_patch_idx, initial_patch_feature = self._find_initial_target(
            initial_cls_attention, features[0]
        )
        
        if target_patch_idx is None:
            logger.warning("Could not identify an initial target. No tracking will be performed.")
            # Still return a valid (but empty) attention map for consistent output
            attention_maps = self._generate_visual_attention_maps(attentions, frames_tensor.shape[1:3])
            return attention_maps, []

        # 3. Track the target through subsequent frames
        trajectory = []
        tracked_patch_feature = initial_patch_feature
        
        # Move features to the correct device for the tracking loop
        features = features.to(self.device)

        for i in range(num_frames):
            current_features = features[i]
            target_patch_idx, tracked_patch_feature = self._track_target(
                tracked_patch_feature, 
                current_features
            )
            
            # Convert patch index to frame coordinates
            patch_y, patch_x = np.unravel_index(target_patch_idx, (patch_h, patch_w))
            frame_h, frame_w = frames_tensor.shape[1:3]
            center_x = int((patch_x + 0.5) * (frame_w / patch_w))
            center_y = int((patch_y + 0.5) * (frame_h / patch_h))
            trajectory.append((center_x, center_y))

        logger.info(f"Tracking complete. Trajectory has {len(trajectory)} points.")

        # 4. Generate final attention maps for visualization
        attention_maps = self._generate_visual_attention_maps(attentions, frames_tensor.shape[1:3])

        return attention_maps, trajectory

    def _get_model_outputs_batched(self, frames_tensor: torch.Tensor, batch_size: int = 16) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Preprocesses frames in batches and returns concatenated attention and hidden states."""
        all_attentions = []
        all_features = []
        num_frames = frames_tensor.shape[0]

        logger.info(f"Processing {num_frames} frames in batches of {batch_size}...")

        try:
            for i in range(0, num_frames, batch_size):
                batch_frames = frames_tensor[i:i+batch_size]
                logger.info(f"  - Processing batch starting at frame {i}...")
                
                # Handle different model interfaces
                if self.processor is not None:
                    # Use processor for transformers models
                    inputs = list(batch_frames.permute(0, 3, 1, 2).cpu())
                    
                    processed_inputs = self.processor(
                        images=inputs, 
                        return_tensors="pt",
                        do_rescale=False
                    ).to(self.device)
                else:
                    # Use direct tensor input for custom models
                    batch_size_actual = batch_frames.shape[0]
                    processed_inputs = {
                        'pixel_values': batch_frames.permute(0, 3, 1, 2).unsqueeze(2).to(self.device)
                    }

                with torch.no_grad():
                    outputs = self.model(**processed_inputs, output_attentions=True, output_hidden_states=True)
                
                # Handle different output formats
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Transformers format
                    batch_attentions = torch.stack(outputs.attentions, dim=0).mean(dim=0).cpu()
                    batch_features = torch.stack(outputs.hidden_states, dim=0).mean(dim=0).cpu()
                    
                    # Remove CLS token feature, keeping only patch features
                    if batch_features.shape[1] > 1:  # Has CLS token
                        all_features.append(batch_features[:, 1:, :])
                    else:
                        all_features.append(batch_features)
                else:
                    # Custom model format
                    batch_attentions = outputs['attentions'][0].cpu()
                    batch_features = outputs['hidden_states'][0].cpu()
                    all_features.append(batch_features)
                
                all_attentions.append(batch_attentions)

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            logger.info("All batches processed. Concatenating results.")
            # Concatenate all results from the batches
            final_attentions = torch.cat(all_attentions, dim=0)
            final_features = torch.cat(all_features, dim=0)

            return final_attentions, final_features

        except Exception as e:
            logger.error(f"Failed to get model outputs in batched mode: {e}", exc_info=True)
            return None, None

    def _generate_visual_attention_maps(self, attentions: torch.Tensor, frame_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Generates upsampled attention maps for visualization from the CLS token.
        """
        num_frames = attentions.shape[0]
        num_patches = attentions.shape[2] - 1  # Exclude CLS token
        patch_h = patch_w = int(np.sqrt(num_patches))

        # Get attention from CLS token to all patches, averaging over heads
        cls_attention = attentions.mean(dim=1)[:, 0, 1:] # (T, num_patches)
        
        # Reshape to a 2D patch map
        attention_maps_2d = cls_attention.view(num_frames, patch_h, patch_w)

        # Upsample to original frame size for visualization
        final_attention = F.interpolate(
            attention_maps_2d.unsqueeze(1),  # Add channel dimension
            size=frame_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(1) # Remove channel dimension

        return final_attention.to(self.device)
            
    def _find_initial_target(self, cls_attention_to_patches: torch.Tensor, features: torch.Tensor) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Finds the most salient patch in the initial frame's attention map."""
        try:
            if cls_attention_to_patches.numel() == 0:
                logger.warning("Cannot find target: CLS attention tensor is empty.")
                return None, None
            
            # The patch with the highest attention from the CLS token is our target
            target_patch_idx = torch.argmax(cls_attention_to_patches).item()
            target_feature = features[target_patch_idx]
            
            logger.info(f"Initial target found at patch index: {target_patch_idx}")
            return target_patch_idx, target_feature
        except Exception as e:
            logger.error(f"Error finding initial target: {e}", exc_info=True)
            return None, None

    def _track_target(self, target_feature: torch.Tensor, all_patch_features: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Finds the most similar patch in the current frame to the target feature."""
        # Cosine similarity
        similarities = F.cosine_similarity(target_feature.unsqueeze(0), all_patch_features)
        
        # Find the best match
        best_match_idx = torch.argmax(similarities).item()
        new_target_feature = all_patch_features[best_match_idx]

        # EMA update for the target feature to allow for appearance changes
        updated_feature = 0.7 * target_feature + 0.3 * new_target_feature
        
        return best_match_idx, updated_feature

    def _fallback_saliency(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """Generates simple saliency maps as a fallback."""
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        saliency_maps = [self._compute_saliency_attention(frame) for frame in frames_np]
        return torch.tensor(np.stack(saliency_maps), dtype=torch.float32).to(self.device)
    
    def extract_attention(self, frames_tensor) -> any:
        """
        Main function to get attention or tracking results.
        For now, it wraps the new tracking logic.
        """
        attention_maps, trajectory = self.extract_attention_and_track(frames_tensor)
        # The pipeline is now updated to handle the trajectory.
        return attention_maps, trajectory
    
    def _compute_saliency_attention(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute saliency-based attention map for a single frame.
        
        Args:
            frame: Single frame (H, W, C) in 0-255 range
            
        Returns:
            Attention map (H, W) normalized to 0-1
        """
        import cv2
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Blur edges to create attention regions
            attention = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
            
            # Normalize to 0-1
            if attention.max() > 0:
                attention = attention / attention.max()
            else:
                attention = np.zeros_like(attention)
                
            # Add some central bias (objects tend to be in center)
            h, w = attention.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (min(h, w) * 0.3)**2)
            
            # Combine edge attention with center bias
            final_attention = 0.7 * attention + 0.3 * center_bias
            
            # Final normalization
            if final_attention.max() > 0:
                final_attention = final_attention / final_attention.max()
                
            return final_attention.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error computing attention: {e}")
            # Return center-focused attention as fallback
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            fallback = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (min(h, w) * 0.4)**2)
            return fallback.astype(np.float32)

    def _reshape_attention_to_spatial(
        self, 
        attention_flat, 
        target_shape: Tuple[int, int]
    ) -> any:
        """
        Reshape flattened attention to spatial dimensions.
        
        Args:
            attention_flat: Flattened attention (B, N)
            target_shape: Target spatial shape (H, W)
            
        Returns:
            Spatial attention maps (B, T, H, W)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for attention reshaping")
            return attention_flat
            
        B, N = attention_flat.shape
        
        # Assume square spatial patches for vision transformers
        spatial_tokens = int(np.sqrt(N))
        
        if spatial_tokens * spatial_tokens != N:
            # If not perfect square, try to handle temporal dimension
            # Assume format: spatial_tokens^2 * temporal_frames = N
            possible_temporal = N // (14 * 14)  # Common patch size 16x16 -> 14x14
            if possible_temporal > 0 and (14 * 14 * possible_temporal) == N:
                spatial_tokens = 14
                temporal_frames = possible_temporal
            else:
                logger.warning(f"Cannot reshape attention {N} to spatial grid, using approximation")
                spatial_tokens = int(np.sqrt(N))
                temporal_frames = 1
        else:
            temporal_frames = 1
            
        # Reshape and upsample to target size
        attention_map = attention_flat.reshape(B, temporal_frames, spatial_tokens, spatial_tokens)
        
        # Upsample to target spatial resolution
        target_h, target_w = target_shape
        attention_map = F.interpolate(
            attention_map.view(-1, 1, spatial_tokens, spatial_tokens),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        attention_map = attention_map.view(B, temporal_frames, target_h, target_w)
        
        return attention_map
    
    def _gradient_based_attention(
        self, 
        pixel_values, 
        original_shape: Tuple
    ) -> any:
        """
        Fallback gradient-based attention when model attention is not available.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for gradient-based attention")
            return pixel_values
            
        pixel_values.requires_grad_(True)
        
        try:
            outputs = self.model(pixel_values)
            if hasattr(outputs, 'logits'):
                # Use max logit as target
                target = outputs.logits.max(dim=-1)[0].sum()
            else:
                target = outputs.last_hidden_state.mean()
                
            target.backward()
            
            # Use gradient magnitude as attention
            gradients = pixel_values.grad.abs().mean(dim=1)  # Average over channels
            
            # Resize to original spatial dimensions
            if len(original_shape) >= 3:
                target_h, target_w = original_shape[-3:-1]
                gradients = F.interpolate(
                    gradients.unsqueeze(1),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
            return gradients.detach()
            
        except Exception as e:
            logger.error(f"Gradient-based attention failed: {e}")
            # Return uniform attention as last resort
            if len(original_shape) >= 3:
                target_h, target_w = original_shape[-3:-1]
                temporal_frames = original_shape[-4] if len(original_shape) > 3 else 1
                return torch.ones(1, temporal_frames, target_h, target_w, device=self.device)
            else:
                return torch.ones(1, 1, 224, 224, device=self.device)
    
    def _fallback_attention(self, frames_tensor) -> any:
        """Fallback method when model fails."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for fallback attention")
            return frames_tensor
            
        logger.warning("Using edge-based fallback attention")
        
        # Simple edge-based attention
        if frames_tensor.dim() == 5:  # (B, T, H, W, C)
            frames = frames_tensor[0]  # Take first batch
        else:  # (T, H, W, C)
            frames = frames_tensor
            
        # Convert to grayscale and compute edges
        if frames.size(-1) == 3:  # RGB
            gray = frames.mean(dim=-1)  # (T, H, W)
        else:
            gray = frames.squeeze(-1)
            
        # Simple edge detection using gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        
        edges = []
        for t in range(gray.size(0)):
            frame = gray[t:t+1].unsqueeze(0)  # (1, 1, H, W)
            grad_x = F.conv2d(frame, sobel_x, padding=1)
            grad_y = F.conv2d(frame, sobel_y, padding=1)
            edge = torch.sqrt(grad_x**2 + grad_y**2)
            edges.append(edge.squeeze())
            
        return torch.stack(edges)  # (T, H, W)
    
    def compute_frame_importance(self, attention_maps) -> List[float]:
        """
        Calculate importance score for each frame based on attention.
        
        Args:
            attention_maps: Attention maps tensor (T, H, W)
            
        Returns:
            List of importance scores for each frame
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for frame importance calculation")
            return [1.0] * len(attention_maps)
            
        scores = []
        
        for i, att_map in enumerate(attention_maps):
            # Compute score as mean of top 10% attention values
            flat = att_map.flatten()
            if len(flat) > 0:
                threshold = torch.quantile(flat, 0.9)
                score = (flat > threshold).float().mean().item()
            else:
                score = 0.0
            scores.append(score)
            
        logger.debug(f"Frame importance scores: min={min(scores):.3f}, max={max(scores):.3f}")
        
        return scores

class AttentionEngine:
    """
    Simple attention engine for individual frame processing.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize attention engine."""
        self.device = device
        
    def process_frame_attention(
        self, 
        frame, 
        model
    ) -> any:
        """
        Process a single frame to extract attention.
        
        Args:
            frame: Single frame tensor (C, H, W)
            model: Vision transformer model
            
        Returns:
            Attention map tensor (H, W)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for frame attention processing")
            return frame
            
        with torch.no_grad():
            # Add batch dimension
            frame_batch = frame.unsqueeze(0)
            
            try:
                # Forward pass with attention
                outputs = model(frame_batch, output_attentions=True)
                
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Get last layer attention
                    attention = outputs.attentions[-1]  # (B, num_heads, seq_len, seq_len)
                    
                    # Average across heads
                    attention_map = attention.mean(dim=1)  # (B, seq_len, seq_len)
                    
                    # Get attention from CLS token to patches
                    if attention_map.size(-1) > 1:
                        attention_map = attention_map[0, 0, 1:]  # Remove CLS to CLS
                    else:
                        attention_map = attention_map[0, 0, :]
                    
                    # Reshape to spatial grid
                    patch_size = int(np.sqrt(attention_map.size(0)))
                    if patch_size * patch_size == attention_map.size(0):
                        attention_map = attention_map.reshape(patch_size, patch_size)
                    else:
                        # Fallback to uniform attention
                        attention_map = torch.ones(14, 14, device=self.device)
                    
                    return attention_map
                else:
                    # Fallback: use gradient-based attention
                    return self._gradient_attention(frame_batch, model)
                    
            except Exception as e:
                logger.warning(f"Attention extraction failed: {e}, using edge detection")
                return self._edge_based_attention(frame)
    
    def _gradient_attention(self, frame, model) -> any:
        """Gradient-based attention fallback."""
        if not TORCH_AVAILABLE:
            return frame
            
        frame.requires_grad_(True)
        
        try:
            outputs = model(frame)
            if hasattr(outputs, 'logits'):
                target = outputs.logits.max()
            else:
                target = outputs.last_hidden_state.mean()
            
            target.backward()
            
            # Use gradient magnitude
            grad_map = frame.grad.abs().mean(dim=1).squeeze(0)  # (H, W)
            return grad_map.detach()
            
        except Exception:
            # Return uniform attention
            return torch.ones(224, 224, device=self.device)
    
    def _edge_based_attention(self, frame) -> any:
        """Edge-based attention fallback."""
        if not TORCH_AVAILABLE:
            return frame
            
        # Convert to grayscale
        if frame.size(0) == 3:  # RGB
            gray = frame.mean(dim=0)  # (H, W)
        else:
            gray = frame.squeeze(0)
        
        # Simple edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        
        gray_batch = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grad_x = F.conv2d(gray_batch, sobel_x, padding=1)
        grad_y = F.conv2d(gray_batch, sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
        
        return edges 