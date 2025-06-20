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
        
        # 1. Preprocess frames and get model outputs (VideoMAE processes entire video clip)
        attentions, features = self._get_model_outputs_video_clip(frames_tensor)
        if attentions is None or features is None:
            # Fallback to simple saliency if model processing fails
            return self._fallback_saliency(frames_tensor), []

        # attentions shape: (T, patch_h, patch_w) - already processed spatial attention
        # features shape: (T, num_patches, feature_dim)
        num_frames = features.shape[0]
        num_patches = features.shape[1]
        patch_h = patch_w = int(np.sqrt(num_patches))

        # 2. Find the initial target in the first frame
        # Use the spatial attention map from the first frame
        initial_spatial_attention = attentions[0].flatten()  # Flatten spatial attention to (patch_h*patch_w)
        
        # Ensure tensors are on the same device
        attentions = attentions.to(self.device)
        features = features.to(self.device)
        
        target_patch_idx, initial_patch_feature = self._find_initial_target(
            initial_spatial_attention, features[0]
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

    def _get_model_outputs_video_clip(self, frames_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Process entire video clip with VideoMAE model."""
        try:
            num_frames = frames_tensor.shape[0]
            logger.info(f"Processing {num_frames} frames as video clip...")
            
            # VideoMAE expects exactly 16 frames
            if num_frames > 16:
                # Sample 16 frames evenly
                indices = torch.linspace(0, num_frames - 1, 16).long()
                selected_frames = frames_tensor[indices]
            elif num_frames < 16:
                # Repeat frames to get exactly 16
                repeat_factor = 16 // num_frames
                remainder = 16 % num_frames
                
                repeated_frames = []
                for _ in range(repeat_factor):
                    repeated_frames.append(frames_tensor)
                if remainder > 0:
                    repeated_frames.append(frames_tensor[:remainder])
                
                selected_frames = torch.cat(repeated_frames, dim=0)
            else:
                selected_frames = frames_tensor
            
            # Resize to exactly 224x224
            frames_resized = selected_frames.permute(0, 3, 1, 2)  # (T, C, H, W)
            current_size = frames_resized.shape[2:]
            
            if current_size != (224, 224):
                logger.info(f"Resizing frames from {current_size} to 224x224 for VideoMAE")
                frames_resized = F.interpolate(
                    frames_resized,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
            
            if self.processor is not None:
                # Use VideoMAE processor
                frames_numpy = frames_resized.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
                
                # Ensure values are in 0-255 range
                if frames_numpy.max() <= 1.0:
                    frames_numpy = (frames_numpy * 255).astype('uint8')
                else:
                    frames_numpy = frames_numpy.astype('uint8')
                
                # Process with VideoMAE processor - convert to PIL Images
                from PIL import Image
                frame_list = []
                for i in range(len(frames_numpy)):
                    pil_frame = Image.fromarray(frames_numpy[i])
                    frame_list.append(pil_frame)
                
                processed_inputs = self.processor(
                    images=frame_list,
                    return_tensors="pt"
                )
                processed_inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}
            else:
                # Direct tensor input for custom models
                processed_inputs = {
                    'pixel_values': frames_resized.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, C, T, H, W)
                }
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**processed_inputs, output_attentions=True, output_hidden_states=True)
            
                         # Extract attention and features
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # VideoMAE format: list of attention tensors for each layer
                # Take the last layer attention: (batch_size, num_heads, seq_len, seq_len)
                attention_tensor = outputs.attentions[-1]  # Last layer
                logger.info(f"VideoMAE attention tensor shape: {attention_tensor.shape}")
                
                # Average across heads: (batch_size, seq_len, seq_len)
                avg_attention = attention_tensor.mean(dim=1)
                
                # Get CLS token attention to patches
                # VideoMAE has CLS token at position 0, so we get attention from CLS to all patches
                cls_attention = avg_attention[0, 0, 1:]  # Skip CLS-to-CLS attention
                
                # VideoMAE uses 14x14 patches (196 patches) for 224x224 images
                # Reshape to spatial dimensions
                spatial_patches = int(np.sqrt(cls_attention.shape[0]))
                if spatial_patches * spatial_patches == cls_attention.shape[0]:
                    spatial_attention = cls_attention.reshape(spatial_patches, spatial_patches)
                else:
                    # Fallback: use first 196 elements and reshape
                    spatial_attention = cls_attention[:196].reshape(14, 14)
                
                # Replicate this attention pattern for all frames in the original video
                final_attentions = []
                for _ in range(num_frames):
                    final_attentions.append(spatial_attention.unsqueeze(0).unsqueeze(0))  # Add batch and head dims
                
                final_attention_tensor = torch.stack(final_attentions, dim=0).squeeze(1).squeeze(1)  # (num_frames, 14, 14)
                
                # Extract features from hidden states
                hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                if hidden_states is not None:
                    # hidden_states shape: (batch_size, seq_len, hidden_size)
                    # Remove CLS token and get patch features
                    patch_features = hidden_states[0, 1:, :]  # Remove CLS token
                    
                    # Replicate features for all frames
                    final_features = []
                    for _ in range(num_frames):
                        final_features.append(patch_features.unsqueeze(0))
                    
                    final_features_tensor = torch.cat(final_features, dim=0)  # (num_frames, num_patches, hidden_size)
                else:
                    # Fallback features
                    final_features_tensor = torch.randn(num_frames, 196, 768).to(self.device)
                
                logger.info(f"Final attention shape: {final_attention_tensor.shape}")
                logger.info(f"Final features shape: {final_features_tensor.shape}")
                
                # Keep tensors on device for consistent processing
                return final_attention_tensor.to(self.device), final_features_tensor.to(self.device)
            else:
                logger.warning("No attention weights found in model output")
                return None, None
                
        except Exception as e:
            logger.error(f"Failed to get model outputs for video clip: {e}", exc_info=True)
            return None, None

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
                    try:
                        # VideoMAE expects exactly 16 frames and 224x224 resolution
                        batch_size_actual = batch_frames.shape[0]
                        
                        # Sample exactly 16 frames for VideoMAE
                        if batch_size_actual > 16:
                            indices = torch.linspace(0, batch_size_actual - 1, 16).long()
                            selected_frames = batch_frames[indices]
                        elif batch_size_actual < 16:
                            # Repeat frames to get exactly 16
                            repeat_factor = 16 // batch_size_actual
                            remainder = 16 % batch_size_actual
                            
                            repeated_frames = []
                            for _ in range(repeat_factor):
                                repeated_frames.append(batch_frames)
                            if remainder > 0:
                                repeated_frames.append(batch_frames[:remainder])
                            
                            selected_frames = torch.cat(repeated_frames, dim=0)
                        else:
                            selected_frames = batch_frames
                        
                        # Ensure frames are exactly 224x224 and convert to numpy
                        frames_resized = selected_frames.permute(0, 3, 1, 2)  # (T, C, H, W)
                        current_size = frames_resized.shape[2:]
                        
                        if current_size != (224, 224):
                            logger.info(f"Resizing frames from {current_size} to 224x224 for VideoMAE")
                            frames_resized = F.interpolate(
                                frames_resized,
                                size=(224, 224),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # Convert to numpy and prepare for processor
                        # VideoMAE processor expects video as numpy array (T, H, W, C)
                        frames_numpy = frames_resized.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
                        
                        # Ensure values are in 0-255 range for processor
                        if frames_numpy.max() <= 1.0:
                            frames_numpy = (frames_numpy * 255).astype('uint8')
                        else:
                            frames_numpy = frames_numpy.astype('uint8')
                        
                        # Process the video clip - VideoMAE processor expects specific format
                        # Convert to list of PIL Images which the processor expects
                        from PIL import Image
                        frame_list = []
                        for i in range(len(frames_numpy)):
                            # Convert numpy array to PIL Image
                            pil_frame = Image.fromarray(frames_numpy[i])
                            frame_list.append(pil_frame)
                        
                        processed_inputs = self.processor(
                            images=frame_list,  # List of PIL Images
                            return_tensors="pt"
                        )
                        
                        # Move to device
                        processed_inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}
                        
                    except Exception as e:
                        logger.warning(f"VideoMAE processor failed: {e}, using fallback")
                        # Fallback: create input manually
                        processed_inputs = {
                            'pixel_values': batch_frames.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
                        }
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
        Generates upsampled attention maps for visualization.
        """
        # Ensure input is on correct device
        attentions = attentions.to(self.device)
        
        # attentions should now be in format (num_frames, patch_h, patch_w)
        if attentions.dim() == 3:
            # New format: (num_frames, patch_h, patch_w)
            attention_maps_2d = attentions
        else:
            # Old format: handle complex tensor structure
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

        return final_attention
            
    def _find_initial_target(self, cls_attention_to_patches: torch.Tensor, features: torch.Tensor) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Finds the most salient patch in the initial frame's attention map."""
        try:
            if cls_attention_to_patches.numel() == 0:
                logger.warning("Cannot find target: CLS attention tensor is empty.")
                return None, None
            
            # Ensure tensors are on the correct device
            cls_attention_to_patches = cls_attention_to_patches.to(self.device)
            features = features.to(self.device)
            
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
        # Ensure both tensors are on the same device
        target_feature = target_feature.to(self.device)
        all_patch_features = all_patch_features.to(self.device)
        
        # Cosine similarity
        similarities = F.cosine_similarity(target_feature.unsqueeze(0), all_patch_features)
        
        # Find the best match
        best_match_idx = torch.argmax(similarities).item()
        new_target_feature = all_patch_features[best_match_idx]

        # EMA update for the target feature to allow for appearance changes
        updated_feature = 0.7 * target_feature + 0.3 * new_target_feature
        
        return best_match_idx, updated_feature

    def _fallback_saliency(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """Generates enhanced saliency maps with object tracking as a fallback."""
        logger.info("Using enhanced fallback saliency with object tracking")
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Compute attention maps and track objects
        saliency_maps = []
        centroids = []
        
        for i, frame in enumerate(frames_np):
            attention_map = self._compute_enhanced_saliency_attention(frame)
            saliency_maps.append(attention_map)
            
            # Find centroid of attention
            centroid = self._find_attention_centroid(attention_map)
            centroids.append(centroid)
            
            if i > 0 and centroids[i] is not None and centroids[i-1] is not None:
                # Check if object moved significantly
                prev_x, prev_y = centroids[i-1]
                curr_x, curr_y = centroids[i]
                distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                
                # If movement is too big, it might be a detection error
                max_movement = min(frame.shape[0], frame.shape[1]) * 0.3  # 30% of image size
                if distance > max_movement:
                    logger.info(f"Large movement detected ({distance:.1f}px), filtering centroid")
                    # Use previous centroid with some smoothing
                    centroids[i] = (
                        int(prev_x * 0.7 + curr_x * 0.3),
                        int(prev_y * 0.7 + curr_y * 0.3)
                    )
        
        # Store centroids for visualization
        self.object_centroids = centroids
        
        return torch.tensor(np.stack(saliency_maps), dtype=torch.float32).to(self.device)
    
    def _find_attention_centroid(self, attention_map: np.ndarray) -> tuple:
        """Find the centroid of the strongest attention region."""
        import cv2
        
        try:
            # Threshold to find strongest regions
            threshold = np.percentile(attention_map[attention_map > 0], 80) if attention_map.max() > 0 else 0
            if threshold <= 0:
                # No significant attention found
                return None
            
            # Create binary mask of strong attention regions
            strong_regions = (attention_map >= threshold).astype(np.uint8)
            
            # Find largest connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strong_regions, connectivity=8)
            
            if num_labels > 1:
                # Find largest component (excluding background)
                largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                centroid = centroids[largest_component]
                return (int(centroid[0]), int(centroid[1]))  # (x, y)
            else:
                # Fallback: compute weighted centroid
                if attention_map.sum() > 0:
                    y_indices, x_indices = np.indices(attention_map.shape)
                    total_attention = attention_map.sum()
                    centroid_x = (x_indices * attention_map).sum() / total_attention
                    centroid_y = (y_indices * attention_map).sum() / total_attention
                    return (int(centroid_x), int(centroid_y))
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding attention centroid: {e}")
            return None
    
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

    def _compute_enhanced_saliency_attention(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute object-focused attention map for detecting moving subjects like animals.
        
        Args:
            frame: Single frame (H, W, C) in 0-255 range
            
        Returns:
            Object-focused attention map (H, W) normalized to 0-1
        """
        import cv2
        
        try:
            # Convert to different color spaces for better object detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            else:
                gray = frame
                hsv = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2HSV)
            
            # 1. Enhanced edge detection for animal shapes
            # Use multiple edge detectors and combine
            edges1 = cv2.Canny(gray, 30, 90)  # Low threshold for softer edges (fur)
            edges2 = cv2.Canny(gray, 100, 200)  # High threshold for strong edges (outline)
            
            # Combine edge maps
            edges_combined = cv2.bitwise_or(edges1, edges2)
            
            # 2. Advanced contour-based object detection
            contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create object mask focusing on animal-like objects
            object_mask = np.zeros_like(gray, dtype=np.float32)
            h, w = gray.shape
            min_area = (h * w) * 0.01   # At least 1% of image (larger min for animals)
            max_area = (h * w) * 0.25   # At most 25% of image
            
            best_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Animal-specific shape analysis
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
                    
                    # Animals typically have aspect ratios between 1:1 and 3:1
                    if 1.0 <= aspect_ratio <= 3.5:
                        # Check contour complexity (animals have more complex shapes)
                        perimeter = cv2.arcLength(contour, True)
                        hull = cv2.convexHull(contour)
                        hull_perimeter = cv2.arcLength(hull, True)
                        
                        if perimeter > 0:
                            complexity = hull_perimeter / perimeter
                            # Animals have moderate complexity (not too simple, not too complex)
                            if 0.6 <= complexity <= 0.95:
                                best_contours.append((contour, area, complexity))
            
            # Sort by area and complexity, prefer larger objects with moderate complexity
            best_contours.sort(key=lambda x: x[1] * (1 - abs(x[2] - 0.8)), reverse=True)
            
            # Create focused mask for the best detected objects (top 2 max)
            for contour, _, _ in best_contours[:2]:
                # Create smooth object mask with distance transform
                temp_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.fillPoly(temp_mask, [contour], 255)
                
                # Use distance transform for smooth boundaries
                dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
                dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
                
                # Also create contour-only mask for edge emphasis
                contour_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, thickness=3)
                contour_mask = cv2.GaussianBlur(contour_mask.astype(np.float32), (5, 5), 0)
                contour_mask = cv2.normalize(contour_mask, None, 0, 1, cv2.NORM_MINMAX)
                
                # Combine filled object with emphasized edges
                combined_mask = 0.6 * dist_transform + 0.4 * contour_mask
                object_mask = np.maximum(object_mask, combined_mask)
            
            # 3. Color-based animal detection 
            color_attention = np.zeros_like(gray, dtype=np.float32)
            if len(frame.shape) == 3:
                # Look for animal colors (browns, blacks, whites, etc.)
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l_channel = lab[:, :, 0].astype(np.float32) / 255.0
                
                # Color distinctiveness - animals often have different colors than background
                mean_color = np.mean(l_channel)
                color_diff = np.abs(l_channel - mean_color)
                color_attention = cv2.normalize(color_diff, None, 0, 1, cv2.NORM_MINMAX)
                
                # Use saturation for colorful elements
                saturation = hsv[:, :, 1].astype(np.float32) / 255.0
                color_attention = 0.7 * color_attention + 0.3 * saturation
            
            # 4. Texture analysis for fur/hair detection
            texture_attention = np.zeros_like(gray, dtype=np.float32)
            if object_mask.max() > 0:
                # Local binary pattern-like analysis
                kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
                
                grad_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
                grad_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
                
                texture_response = np.sqrt(grad_x**2 + grad_y**2)
                texture_attention = cv2.normalize(texture_response, None, 0, 1, cv2.NORM_MINMAX)
            
            # 5. Reduced center bias (animals can be anywhere)
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 2)**2))
            center_bias = center_bias * 0.3  # Much weaker center bias
            
            # Combine all cues with strong emphasis on detected objects
            if object_mask.max() > 0.1:
                # Strong object detected - focus on it
                final_attention = (
                    0.6 * object_mask +           # Detected objects (highest priority)
                    0.2 * color_attention +       # Color distinctiveness
                    0.15 * texture_attention +    # Texture analysis
                    0.05 * center_bias           # Very weak center bias
                )
                logger.info(f"Strong object detected, max attention: {object_mask.max():.3f}")
            else:
                # No strong objects - use general saliency
                gradient_magnitude = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2
                gradient_magnitude = np.sqrt(gradient_magnitude)
                gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
                
                final_attention = (
                    0.4 * gradient_magnitude +
                    0.3 * color_attention +
                    0.2 * texture_attention +
                    0.1 * center_bias
                )
                logger.info("No strong objects detected, using general saliency")
            
            # Clean up with morphological operations
            if final_attention.max() > 0.1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                final_attention = cv2.morphologyEx(final_attention, cv2.MORPH_CLOSE, kernel)
                final_attention = cv2.morphologyEx(final_attention, cv2.MORPH_OPEN, kernel)
            
            # Light smoothing to maintain object boundaries
            final_attention = cv2.GaussianBlur(final_attention, (11, 11), 0)
            
            # Final processing
            final_attention = cv2.normalize(final_attention, None, 0, 1, cv2.NORM_MINMAX)
            final_attention = np.nan_to_num(final_attention, nan=0.0, posinf=1.0, neginf=0.0)
            final_attention = np.clip(final_attention, 0.0, 1.0)
            
            # Strong contrast enhancement for better visualization
            if final_attention.max() > 0.05:
                final_attention = np.power(final_attention, 0.4)  # Stronger gamma for contrast
                
                # Create binary-like mask for the strongest regions
                threshold = np.percentile(final_attention[final_attention > 0], 75)
                strong_regions = final_attention > threshold
                
                # Boost the strongest regions
                final_attention[strong_regions] = np.minimum(final_attention[strong_regions] * 1.5, 1.0)
            
            return final_attention.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Enhanced object detection failed: {e}")
            # Simple edge-based fallback
            try:
                h, w = gray.shape if 'gray' in locals() else frame.shape[:2]
                edges = cv2.Canny(gray if 'gray' in locals() else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 50, 150)
                fallback = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
                fallback = cv2.normalize(fallback, None, 0, 1, cv2.NORM_MINMAX)
                return fallback.astype(np.float32)
            except:
                h, w = frame.shape[:2]
                return np.ones((h, w), dtype=np.float32) * 0.5

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