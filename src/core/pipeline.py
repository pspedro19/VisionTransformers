"""
In-Memory Pipeline for Video to GIF processing with attention.
"""

import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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

from .video_decoder import OptimizedVideoDecoder
from .attention_engine import VideoAttentionEngine, AttentionEngine
from .gif_composer import GifComposer
from PIL import Image
from decord import VideoReader
import imageio
from ..models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class InMemoryPipeline:
    """
    Complete pipeline for video to GIF processing without intermediate disk writes.
    """
    
    def __init__(self, config_path: str = "config/mvp1.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components with auto device detection
        device = self.config.get('model', {}).get('device', 'auto')
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
            logger.info(f"Auto-detected device: {device}")
        
        gpu_id = 0 if device == 'cuda' else -1
        
        logger.info(f"Initializing decoder with GPU ID: {gpu_id}")
        self.decoder = OptimizedVideoDecoder(gpu_id=gpu_id)
        
        logger.info(f"Initializing attention engine with device: {device}")
        self.attention_engine = VideoAttentionEngine(
            model_name=self.config.get('model', {}).get('name', 'videomae-base'),
            device=device
        )
        
        self.gif_composer = GifComposer()
        
        self.model_factory = ModelFactory()
        
        logger.info(f"Pipeline initialized with config: {config_path}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "videomae-base",
                "device": "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
                "precision": "fp16"
            },
            "limits": {
                "max_resolution": 720,
                "max_duration": 60,
                "max_file_size": 100,
                "allowed_formats": ["mp4", "avi", "mov", "webm"]
            },
            "processing": {
                "adaptive_stride": True,
                "min_stride": 2,
                "max_stride": 10
            },
            "gif": {
                "fps": 5,
                "max_frames": 20,
                "optimization_level": 2,
                "overlay_style": "heatmap",
                "overlay_intensity": 0.7
            },
            "metrics": {
                "track_performance": True,
                "mlflow_uri": None
            }
        }
    
    def process_video(
        self, 
        video_path: str, 
        output_path: str,
        override_config: Optional[Dict[str, Any]] = None,
        profile: Optional[str] = None,
        profile_config: Optional[Dict[str, Any]] = None,
        advanced_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline processing from video to GIF.
        
        Args:
            video_path: Path to input video
            output_path: Path for output GIF
            override_config: Optional config overrides
            profile: Optional profile name
            profile_config: Optional profile configuration
            advanced_config: Optional advanced configuration
            
        Returns:
            Processing results and statistics
        """
        # Apply config overrides
        config = self.config.copy()
        if profile_config:
            config = self._merge_configs(config, profile_config)
        if advanced_config:
            config = self._merge_configs(config, advanced_config)
        if override_config:
            config = self._merge_configs(config, override_config)
            
        logger.info(f"Processing video: {video_path} -> {output_path}")
        
        try:
            # Track performance with automatic device detection
            import time
            
            if TORCH_AVAILABLE and config['model']['device'] == 'cuda':
                try:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    use_cuda_timing = True
                    logger.info("Using CUDA timing events")
                except Exception as e:
                    logger.warning(f"CUDA timing failed: {e}, using CPU timing")
                    start_time = time.time()
                    use_cuda_timing = False
            else:
                start_time = time.time()
                use_cuda_timing = False
                logger.info("Using CPU timing")
            
            # Step 1: Decode video to tensors
            logger.info("Step 1: Decoding video...")
            frames, fps = self.decoder.decode_video_stream(
                video_path,
                max_resolution=config['limits']['max_resolution'],
                max_duration=config['limits']['max_duration']
            )
            
            logger.info(f"Decoded {len(frames)} frames at {fps:.1f} FPS")
            logger.info(f"Frame tensor shape: {frames.shape}, dtype: {frames.dtype}")
            logger.info(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
            
            # Step 2: Convert frames to numpy for processing (ensure correct format)
            frames_np = frames.cpu().numpy()
            
            # Ensure frames are in 0-255 range for processing
            if frames_np.max() <= 1.0:
                frames_display = (frames_np * 255).astype(np.uint8)
            else:
                frames_display = frames_np.astype(np.uint8)
                
            logger.info(f"Display frames shape: {frames_display.shape}")
            logger.info(f"Display frames range: [{frames_display.min()}, {frames_display.max()}]")
            
            # Step 3: Extract attention maps and track object
            logger.info("Step 2: Extracting attention and tracking object...")
            attention_maps, trajectory = self.attention_engine.extract_attention_and_track(frames)
            
            logger.info(f"Attention maps shape: {attention_maps.shape}")
            logger.info(f"Attention range: [{attention_maps.min():.3f}, {attention_maps.max():.3f}]")
            logger.info(f"Trajectory length: {len(trajectory)}")
            
            # Step 4: Limit frames if needed
            max_frames = config['gif']['max_frames']
            if len(frames_display) > max_frames:
                # Take evenly spaced frames
                indices = np.linspace(0, len(frames_display)-1, max_frames, dtype=int)
                frames_display = frames_display[indices]
                attention_maps = attention_maps[indices]
                logger.info(f"Limited to {len(frames_display)} frames")
            
            # Step 5: Create GIF with attention overlays
            logger.info("Step 3: Creating GIF...")
            
            # Convert attention maps to numpy if needed
            if isinstance(attention_maps, torch.Tensor):
                attention_np = attention_maps.cpu().numpy()
            else:
                attention_np = attention_maps
                
            # Create GIF with simple overlay
            gif_style = config.get('gif', {}).get('overlay_style', 'highlight')

            if gif_style == 'tracking_dot':
                 gif_stats = self.gif_composer.create_gif(
                    frames=torch.from_numpy(frames_display),
                    attention_maps=torch.from_numpy(attention_np),
                    output_path=output_path,
                    fps=config.get('gif', {}).get('fps', 10),
                    overlay_style='transparent',
                    calculate_tracking_dot=True,
                    overlay_intensity=0
                )
            else:
                gif_stats = self._create_simple_gif(
                    frames_display,
                    attention_np,
                    output_path,
                    config,
                    trajectory=trajectory
                )
            
            # Calculate timing
            if use_cuda_timing:
                try:
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time = start_event.elapsed_time(end_event) / 1000.0
                except Exception as e:
                    logger.warning(f"CUDA timing calculation failed: {e}")
                    total_time = time.time() - start_time
            else:
                total_time = time.time() - start_time
            
            # Compile results
            results = {
                "success": True,
                "input_video": video_path,
                "output_gif": output_path,
                "total_frames": len(frames),
                "selected_frames": len(frames_display),
                "processing_time": total_time,
                "fps_original": fps,
                "fps_gif": config['gif']['fps'],
                "gif_stats": gif_stats
            }
            
            logger.info(f"Pipeline completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "input_video": video_path,
                "output_gif": output_path
            }
    
    def _create_simple_gif(
        self,
        frames: np.ndarray,
        attention_maps: np.ndarray,
        output_path: str,
        config: Dict[str, Any],
        trajectory: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Create GIF with a simple, direct overlay of attention maps.
        This version is less complex than the fallback and primary for direct visualization.
        """
        gif_config = config.get('gif', {})
        
        try:
            stats = self.gif_composer.create_gif(
                frames=frames,
                attention_maps=attention_maps,
                output_path=output_path,
                fps=gif_config.get('fps', 10),
                overlay_style=gif_config.get('overlay_style', 'highlight'),
                overlay_intensity=gif_config.get('overlay_intensity', 0.6),
                optimization_level=gif_config.get('optimization_level', 2),
                quality=gif_config.get('quality', 95),
                calculate_tracking_dot=False
            )
            return stats
            
        except Exception as e:
            logger.error(f"Failed to create simple GIF: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _select_key_frames(
        self, 
        importance_scores: List[float], 
        fps: float,
        max_frames: int,
        min_distance_frames: int = 5
    ) -> List[int]:
        """
        Select key frames using non-maximum suppression based on importance.
        
        Args:
            importance_scores: Importance score for each frame
            fps: Original video FPS
            max_frames: Maximum number of frames to select
            min_distance_frames: Minimum distance between selected frames
            
        Returns:
            List of selected frame indices
        """
        if len(importance_scores) <= max_frames:
            return list(range(len(importance_scores)))
        
        # Sort frames by importance (descending)
        frame_importance = [(i, score) for i, score in enumerate(importance_scores)]
        frame_importance.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = []
        
        for frame_idx, score in frame_importance:
            # Check if this frame is too close to already selected frames
            too_close = False
            for selected_idx in selected_indices:
                if abs(frame_idx - selected_idx) < min_distance_frames:
                    too_close = True
                    break
            
            if not too_close:
                selected_indices.append(frame_idx)
                
            # Stop if we have enough frames
            if len(selected_indices) >= max_frames:
                break
        
        # Sort selected indices chronologically
        selected_indices.sort()
        
        logger.debug(f"Selected frames: {selected_indices}")
        return selected_indices
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _track_metrics(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Track metrics using MLflow if available."""
        try:
            import mlflow
            
            mlflow_uri = config.get('metrics', {}).get('mlflow_uri')
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("model_name", config.get('model', {}).get('name'))
                mlflow.log_param("overlay_style", config.get('gif', {}).get('overlay_style'))
                mlflow.log_param("max_frames", config.get('gif', {}).get('max_frames'))
                
                # Log metrics
                mlflow.log_metric("processing_time", results['processing_time'])
                mlflow.log_metric("total_frames", results['total_frames'])
                mlflow.log_metric("selected_frames", results['selected_frames'])
                mlflow.log_metric("selection_ratio", results['selection_ratio'])
                mlflow.log_metric("file_size_mb", results['gif_stats']['file_size_mb'])
                mlflow.log_metric("compression_ratio", results['gif_stats']['compression_ratio'])
                
                # Log importance statistics
                importance = results['importance_scores']
                mlflow.log_metric("importance_mean", importance['mean'])
                mlflow.log_metric("importance_std", importance['std'])
                
                logger.info("Metrics logged to MLflow")
                
        except ImportError:
            logger.debug("MLflow not available for metrics tracking")
        except Exception as e:
            logger.warning(f"Failed to track metrics: {e}")
    
    def process_batch(
        self, 
        video_paths: List[str], 
        output_dir: str,
        batch_config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple videos in batch.
        
        Args:
            video_paths: List of video file paths
            output_dir: Output directory for GIFs
            batch_config: Optional batch processing configuration
            
        Returns:
            List of processing results for each video
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, video_path in enumerate(video_paths):
            try:
                video_name = Path(video_path).stem
                output_path = output_dir / f"{video_name}_highlight.gif"
                
                logger.info(f"Processing batch {i+1}/{len(video_paths)}: {video_path}")
                
                result = self.process_video(
                    str(video_path), 
                    str(output_path), 
                    batch_config
                )
                
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "input_video": video_path,
                    "batch_index": i
                })
        
        # Summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        logger.info(f"Batch processing completed: {successful}/{len(video_paths)} successful, total time: {total_time:.2f}s")
        
        return results
    
    def get_video_preview(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information and preview without full processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video metadata and preview information
        """
        try:
            # Get basic video info
            video_info = self.decoder.get_video_info(video_path)
            
            # If decoder returned an error, propagate it
            if "error" in video_info:
                return {
                    "error": f"Failed to decode video info: {video_info['error']}",
                    "can_process": False
                }

            # Check against limits
            limits = self.config['limits']
            within_limits = {
                "resolution": max(video_info['width'], video_info['height']) <= limits['max_resolution'],
                "duration": video_info['duration'] <= limits['max_duration'],
                "file_size": video_info['file_size_mb'] * 1024 * 1024 <= limits['max_file_size'] * 1024 * 1024
            }
            can_process = all(within_limits.values())

            # Estimate processing time
            estimated_time = self._estimate_processing_time(video_info) if can_process else -1.0
            
            # Recommend settings
            recommended_settings = self._recommend_settings(video_info) if can_process else {}

            return {
                "video_info": video_info,
                "within_limits": within_limits,
                "can_process": can_process,
                "estimated_processing_time": estimated_time,
                "recommended_settings": recommended_settings
            }
            
        except Exception as e:
            logger.error(f"Error getting video preview for {video_path}: {e}", exc_info=True)
            return {
                "error": str(e),
                "can_process": False
            }
    
    def _estimate_processing_time(self, video_info: Dict[str, Any]) -> float:
        """Estimate processing time based on video characteristics."""
        # Simple heuristic based on resolution and duration
        pixels_per_second = video_info['width'] * video_info['height'] * video_info.get('fps', 30.0)
        total_pixels = pixels_per_second * video_info['duration']
        
        # Rough estimates (seconds per megapixel)
        device = video_info.get('device', 'cpu')
        if TORCH_AVAILABLE and device == 'cuda':
            time_per_mpixel = 0.1  # GPU
        else:
            time_per_mpixel = 0.5  # CPU
            
        estimated_time = (total_pixels / 1_000_000) * time_per_mpixel
        
        return max(estimated_time, 1.0)  # Minimum 1 second
    
    def _recommend_settings(self, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal settings based on video characteristics."""
        duration = video_info['duration']
        resolution = max(video_info['width'], video_info['height'])
        fps = video_info.get('fps', 30.0)  # Default to 30 FPS if not provided
        
        # Recommend GIF FPS based on original FPS and duration
        if duration < 5:
            recommended_fps = min(8, fps / 2)
        elif duration < 15:
            recommended_fps = 5
        else:
            recommended_fps = 3
            
        # Recommend max frames based on duration
        if duration < 10:
            max_frames = min(20, int(duration * recommended_fps))
        else:
            max_frames = 15
            
        # Recommend overlay intensity based on resolution
        if resolution > 1080:
            overlay_intensity = 0.6  # Subtle for high-res
        else:
            overlay_intensity = 0.7  # Standard
            
        return {
            "gif": {
                "fps": recommended_fps,
                "max_frames": max_frames,
                "overlay_intensity": overlay_intensity,
                "overlay_style": "heatmap"
            },
            "processing": {
                "model": "videomae-base" if resolution <= 720 else "timesformer-base"
            }
        }

class VideoPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_factory = ModelFactory()
        self.attention_engine = AttentionEngine()
        
    def process_video(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        duration: float,
        fps: int = 5,
        overlay_style: str = "heatmap",
        model_name: Optional[str] = None,
        profile_config: Optional[Dict[str, Any]] = None,
        advanced_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesa un video para generar un GIF con rastreo de objetos.
        """
        try:
            # 1. Cargar modelo según configuración
            model = self.model_factory.get_model(
                model_name or self.config.get("default_model", "vit-base"),
                profile_config or {},
                advanced_config or {}
            )
            
            # 2. Cargar video y extraer frames
            vr = VideoReader(video_path)
            total_frames = len(vr)
            start_frame = int(start_time * vr.get_avg_fps())
            end_frame = min(
                total_frames,
                start_frame + int(duration * vr.get_avg_fps())
            )
            
            # Calcular frames a extraer
            frame_indices = np.linspace(
                start_frame,
                end_frame - 1,
                num=int(duration * fps),
                dtype=np.int32
            )
            
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # 3. Procesar frames con el modelo
            processed_frames = []
            attention_maps = []
            
            with torch.no_grad():
                for frame in frames:
                    # Preprocesar frame
                    frame_tensor = self._preprocess_frame(frame)
                    
                    # Obtener atención y features
                    attention, features = model(
                        frame_tensor,
                        output_attentions=True,
                        return_dict=True
                    )
                    
                    # Procesar mapa de atención
                    attention_map = self._process_attention_map(
                        attention,
                        frame.shape[:2],
                        overlay_style
                    )
                    
                    # Combinar frame original con mapa de atención
                    processed_frame = self._apply_attention_overlay(
                        frame,
                        attention_map,
                        overlay_style
                    )
                    
                    processed_frames.append(processed_frame)
                    attention_maps.append(attention_map)
            
            # 4. Generar GIF con rastreo
            self._save_gif(
                processed_frames,
                output_path,
                fps=fps,
                optimize=True
            )
            
            # 5. Generar metadatos
            metadata = {
                "frames": len(processed_frames),
                "fps": fps,
                "duration": duration,
                "start_time": start_time,
                "model": model_name,
                "overlay_style": overlay_style,
                "attention_stats": self._calculate_attention_stats(attention_maps)
            }
            
            return {
                "status": "success",
                "output_path": output_path,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error en pipeline: {str(e)}")
            raise
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocesa un frame para el modelo."""
        # Convertir a tensor y normalizar
        frame_tensor = torch.from_numpy(frame).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        frame_tensor = frame_tensor / 255.0
        
        # Normalizar con media y std de ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        frame_tensor = (frame_tensor - mean) / std
        
        # Redimensionar si es necesario
        if frame_tensor.shape[1:] != (224, 224):
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return frame_tensor.unsqueeze(0)  # Agregar dimensión de batch
    
    def _process_attention_map(
        self,
        attention: torch.Tensor,
        target_size: Tuple[int, int],
        style: str
    ) -> np.ndarray:
        """Procesa el mapa de atención según el estilo seleccionado."""
        # Obtener mapa de atención promedio de todas las cabezas
        attention_map = attention.mean(dim=(0, 1))  # [H, W]
        
        # Redimensionar al tamaño objetivo
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Normalizar
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min()
        )
        
        # Convertir a numpy
        attention_map = attention_map.cpu().numpy()
        
        # Aplicar estilo
        if style == "heatmap":
            attention_map = cv2.applyColorMap(
                (attention_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
        elif style == "highlight":
            attention_map = np.stack([attention_map] * 3, axis=-1)
        elif style == "glow":
            attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
            attention_map = np.stack([attention_map] * 3, axis=-1)
        elif style == "pulse":
            attention_map = np.stack([attention_map] * 3, axis=-1)
            attention_map = cv2.GaussianBlur(attention_map, (25, 25), 0)
        
        return attention_map
    
    def _apply_attention_overlay(
        self,
        frame: np.ndarray,
        attention_map: np.ndarray,
        style: str
    ) -> np.ndarray:
        """Aplica el overlay de atención al frame."""
        if style == "heatmap":
            return cv2.addWeighted(frame, 0.7, attention_map, 0.3, 0)
        elif style in ["highlight", "glow"]:
            mask = attention_map > 0.5
            frame[mask] = frame[mask] * 0.7 + np.array([255, 255, 200]) * 0.3
            return frame
        elif style == "pulse":
            alpha = attention_map * 0.4
            overlay = np.ones_like(frame) * np.array([100, 200, 255])
            return frame * (1 - alpha) + overlay * alpha
        return frame
    
    def _save_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int,
        optimize: bool = True
    ):
        """Guarda los frames como GIF."""
        # Convertir frames a PIL
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Guardar como GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # Duración en ms
            loop=0,
            optimize=optimize
        )
    
    def _calculate_attention_stats(
        self,
        attention_maps: List[np.ndarray]
    ) -> Dict[str, float]:
        """Calcula estadísticas de los mapas de atención."""
        attention_values = np.concatenate([
            map.flatten() for map in attention_maps
        ])
        
        return {
            "mean": float(np.mean(attention_values)),
            "std": float(np.std(attention_values)),
            "max": float(np.max(attention_values)),
            "min": float(np.min(attention_values))
        } 