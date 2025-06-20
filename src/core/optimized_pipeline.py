"""
Pipeline Optimizado de Doble Canal
==================================

Pipeline principal que integra:
- Detección automática de hardware
- Sistema de doble canal de atención
- Monitoreo en tiempo real
- Optimizaciones dinámicas
- Post-procesamiento inteligente
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .hardware_detector import auto_detect_and_configure, HardwareInfo, PerformanceProfile
from .dual_channel_attention import DualChannelAttentionSystem, ChannelAOutput, ChannelBOutput
from .performance_monitor import PerformanceMonitor
from .video_decoder import VideoDecoder
from .gif_composer import GIFComposer

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Resultado del procesamiento."""
    output_path: str
    processing_time: float
    frames_processed: int
    avg_fps: float
    quality_metrics: Dict[str, float]
    hardware_used: HardwareInfo
    performance_summary: Dict[str, Any]

class OptimizedDualChannelPipeline:
    """
    Pipeline optimizado de doble canal con detección automática de hardware.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.hardware_info = None
        self.performance_profile = None
        self.config = None
        self.attention_system = None
        self.performance_monitor = None
        self.video_decoder = None
        self.gif_composer = None
        
        # Estado del procesamiento
        self.is_processing = False
        self.current_progress = 0.0
        self.progress_callback = None
        
        # Configuración de post-procesamiento
        self.post_processing_config = {
            'upscale_enabled': True,
            'temporal_smoothing': True,
            'quality_enhancement': True,
            'compression_optimization': True
        }
        
        logger.info("🚀 Pipeline optimizado de doble canal inicializado")
    
    def initialize_system(self) -> Tuple[HardwareInfo, PerformanceProfile, Dict]:
        """
        Inicializa el sistema con detección automática de hardware.
        
        Returns:
            Tuple con información de hardware, perfil de rendimiento y configuración
        """
        logger.info("🔍 Iniciando detección automática de hardware...")
        
        # Detección automática y configuración
        self.hardware_info, self.performance_profile, self.config = auto_detect_and_configure()
        
        # Inicializar componentes del sistema
        self._initialize_components()
        
        logger.info("✅ Sistema inicializado correctamente")
        
        return self.hardware_info, self.performance_profile, self.config
    
    def _initialize_components(self):
        """Inicializa todos los componentes del sistema."""
        # Sistema de atención de doble canal
        self.attention_system = DualChannelAttentionSystem(self.config)
        
        # Monitor de rendimiento
        self.performance_monitor = PerformanceMonitor(
            config=self.config,
            update_interval=1.0,
            enable_alerts=True
        )
        
        # Decodificador de video
        self.video_decoder = VideoDecoder(
            max_resolution=tuple(self.config['video']['max_resolution']),
            target_fps=self.config['video']['target_fps'],
            temporal_sampling=self.config['video']['temporal_sampling']
        )
        
        # Compositor de GIF
        self.gif_composer = GIFComposer(
            output_format=self.config['visualization']['output_format'],
            quality=self.config['visualization']['output_quality'],
            frame_rate=self.config['visualization']['frame_rate']
        )
        
        logger.info("✅ Componentes del sistema inicializados")
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResult:
        """
        Procesa un video con el sistema de doble canal optimizado.
        
        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta de salida (opcional)
            progress_callback: Callback para progreso
            
        Returns:
            Resultado del procesamiento
        """
        if self.is_processing:
            raise RuntimeError("Ya hay un procesamiento en curso")
        
        if not self.attention_system:
            self.initialize_system()
        
        self.is_processing = True
        self.progress_callback = progress_callback
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Iniciando procesamiento de: {video_path}")
            
            # Iniciar monitoreo
            self.performance_monitor.start_monitoring()
            
            # Decodificar video
            frames = self._decode_video(video_path)
            total_frames = frames.shape[2]
            
            # Actualizar frames restantes en el monitor
            self.performance_monitor.update_frames_remaining(total_frames)
            
            # Procesar con doble canal
            channel_a_output, channel_b_output = self._process_dual_channel(frames)
            
            # Componer visualización final
            output_path = self._compose_visualization(
                frames, channel_a_output, channel_b_output, output_path
            )
            
            # Post-procesamiento
            if self.post_processing_config['upscale_enabled']:
                output_path = self._apply_post_processing(output_path)
            
            # Calcular métricas finales
            processing_time = time.time() - start_time
            performance_summary = self.performance_monitor.get_performance_summary()
            
            result = ProcessingResult(
                output_path=output_path,
                processing_time=processing_time,
                frames_processed=total_frames,
                avg_fps=performance_summary.get('avg_fps', 0),
                quality_metrics=self._calculate_quality_metrics(output_path),
                hardware_used=self.hardware_info,
                performance_summary=performance_summary
            )
            
            logger.info(f"✅ Procesamiento completado en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}")
            raise
        finally:
            self.is_processing = False
            self.performance_monitor.stop_monitoring()
    
    def _decode_video(self, video_path: str) -> torch.Tensor:
        """Decodifica el video con optimizaciones."""
        logger.info("📹 Decodificando video...")
        
        # Verificar límites de duración
        video_info = self.video_decoder.get_video_info(video_path)
        max_duration = self.config['video']['max_duration']
        
        if video_info['duration'] > max_duration:
            logger.warning(f"⚠️ Video muy largo ({video_info['duration']:.1f}s), limitando a {max_duration}s")
        
        # Decodificar con optimizaciones
        frames = self.video_decoder.decode_video(
            video_path,
            max_duration=max_duration,
            target_resolution=tuple(self.config['video']['max_resolution'])
        )
        
        logger.info(f"✅ Video decodificado: {frames.shape}")
        return frames
    
    def _process_dual_channel(self, frames: torch.Tensor) -> Tuple[ChannelAOutput, ChannelBOutput]:
        """Procesa el video con ambos canales de atención."""
        logger.info("🔄 Procesando doble canal de atención...")
        
        # Procesar en chunks si es necesario
        if self.config['processing'].get('chunk_size'):
            return self._process_in_chunks(frames)
        else:
            return self.attention_system.process_video(frames)
    
    def _process_in_chunks(self, frames: torch.Tensor) -> Tuple[ChannelAOutput, ChannelBOutput]:
        """Procesa el video en chunks para optimizar memoria."""
        B, C, T, H, W = frames.shape
        chunk_size = self.config['processing']['chunk_size']
        chunk_overlap = self.config['processing'].get('chunk_overlap', 0)
        
        logger.info(f"📦 Procesando en chunks de {chunk_size}s con overlap de {chunk_overlap}s")
        
        all_channel_a_outputs = []
        all_channel_b_outputs = []
        
        for chunk_start in range(0, T, chunk_size - chunk_overlap):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_frames = frames[:, :, chunk_start:chunk_end]
            
            # Procesar chunk
            chunk_a_output, chunk_b_output = self.attention_system.process_video(chunk_frames)
            
            all_channel_a_outputs.append(chunk_a_output)
            all_channel_b_outputs.append(chunk_b_output)
            
            # Actualizar progreso
            progress = (chunk_end / T) * 100
            self.current_progress = progress
            if self.progress_callback:
                self.progress_callback(progress)
            
            # Mostrar estado del monitor
            self.performance_monitor.print_status_display()
            
            # Limpiar caché GPU si es necesario
            if self.config['processing'].get('cache_cleanup_interval'):
                if (len(all_channel_a_outputs) % self.config['processing']['cache_cleanup_interval']) == 0:
                    torch.cuda.empty_cache()
        
        # Combinar resultados de chunks
        return self._combine_chunk_results(all_channel_a_outputs, all_channel_b_outputs)
    
    def _combine_chunk_results(
        self, 
        channel_a_outputs: List[ChannelAOutput], 
        channel_b_outputs: List[ChannelBOutput]
    ) -> Tuple[ChannelAOutput, ChannelBOutput]:
        """Combina los resultados de múltiples chunks."""
        # Combinar atención global (Canal A)
        combined_attention_weights = []
        for output in channel_a_outputs:
            combined_attention_weights.append(output.global_attention.attention_weights)
        
        combined_attention = torch.cat(combined_attention_weights, dim=0)
        
        # Combinar detecciones de objetos (Canal B)
        combined_detections = []
        combined_object_maps = {}
        combined_trajectories = {}
        
        for output in channel_b_outputs:
            combined_detections.extend(output.detections)
            combined_object_maps.update(output.object_attention_maps)
            combined_trajectories.update(output.object_trajectories)
        
        # Crear outputs combinados
        combined_channel_a = ChannelAOutput(
            global_attention=channel_a_outputs[0].global_attention,
            confidence_map=combined_attention
        )
        
        combined_channel_b = ChannelBOutput(
            detections=combined_detections,
            object_attention_maps=combined_object_maps,
            tracking_info={'total_tracks': len(combined_trajectories)},
            object_trajectories=combined_trajectories
        )
        
        return combined_channel_a, combined_channel_b
    
    def _compose_visualization(
        self,
        frames: torch.Tensor,
        channel_a_output: ChannelAOutput,
        channel_b_output: ChannelBOutput,
        output_path: Optional[str]
    ) -> str:
        """Compone la visualización final con ambos canales."""
        logger.info("🎨 Componiendo visualización final...")
        
        if not output_path:
            output_dir = Path(self.config['output']['directory'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_name = Path(frames).stem if hasattr(frames, 'stem') else 'video'
            timestamp = int(time.time())
            output_path = output_dir / f"{video_name}_dual_channel_{timestamp}.gif"
        
        # Configuración de visualización
        viz_config = self.config['visualization']
        
        # Componer GIF con ambos canales
        self.gif_composer.compose_dual_channel_gif(
            frames=frames,
            channel_a_attention=channel_a_output.global_attention.attention_weights,
            channel_b_detections=channel_b_output.detections,
            channel_a_style=viz_config['channel_a_style'],
            channel_b_style=viz_config['channel_b_style'],
            output_path=str(output_path)
        )
        
        logger.info(f"✅ Visualización guardada en: {output_path}")
        return str(output_path)
    
    def _apply_post_processing(self, output_path: str) -> str:
        """Aplica post-procesamiento para mejorar la calidad."""
        logger.info("🔧 Aplicando post-procesamiento...")
        
        if self.post_processing_config['upscale_enabled']:
            output_path = self._upscale_output(output_path)
        
        if self.post_processing_config['temporal_smoothing']:
            output_path = self._apply_temporal_smoothing(output_path)
        
        if self.post_processing_config['quality_enhancement']:
            output_path = self._enhance_quality(output_path)
        
        if self.post_processing_config['compression_optimization']:
            output_path = self._optimize_compression(output_path)
        
        return output_path
    
    def _upscale_output(self, input_path: str) -> str:
        """Upscaling inteligente del output."""
        try:
            import cv2
            from PIL import Image
            
            # Cargar GIF
            gif = Image.open(input_path)
            frames = []
            
            # Upscale cada frame
            for frame in gif.frames:
                frame_array = np.array(frame)
                
                # Upscale usando interpolación bicúbica
                upscaled = cv2.resize(
                    frame_array, 
                    None, 
                    fx=1.5, 
                    fy=1.5, 
                    interpolation=cv2.INTER_CUBIC
                )
                
                frames.append(Image.fromarray(upscaled))
            
            # Guardar GIF upscaled
            output_path = input_path.replace('.gif', '_upscaled.gif')
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=gif.info.get('duration', 100)
            )
            
            logger.info(f"✅ Upscaling aplicado: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"⚠️ Error en upscaling: {e}")
            return input_path
    
    def _apply_temporal_smoothing(self, input_path: str) -> str:
        """Aplica suavizado temporal al GIF."""
        try:
            import cv2
            from PIL import Image
            
            # Cargar GIF
            gif = Image.open(input_path)
            frames = []
            
            # Aplicar suavizado temporal
            for i, frame in enumerate(gif.frames):
                frame_array = np.array(frame)
                
                if i > 0:
                    # Suavizado con frame anterior
                    prev_frame = np.array(frames[-1])
                    smoothed = cv2.addWeighted(frame_array, 0.7, prev_frame, 0.3, 0)
                    frames.append(Image.fromarray(smoothed))
                else:
                    frames.append(Image.fromarray(frame_array))
            
            # Guardar GIF suavizado
            output_path = input_path.replace('.gif', '_smoothed.gif')
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=gif.info.get('duration', 100)
            )
            
            logger.info(f"✅ Suavizado temporal aplicado: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"⚠️ Error en suavizado temporal: {e}")
            return input_path
    
    def _enhance_quality(self, input_path: str) -> str:
        """Mejora la calidad del GIF."""
        try:
            from PIL import Image, ImageEnhance
            
            # Cargar GIF
            gif = Image.open(input_path)
            frames = []
            
            # Mejorar cada frame
            for frame in gif.frames:
                # Aumentar contraste
                enhancer = ImageEnhance.Contrast(frame)
                enhanced = enhancer.enhance(1.2)
                
                # Aumentar saturación
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.1)
                
                frames.append(enhanced)
            
            # Guardar GIF mejorado
            output_path = input_path.replace('.gif', '_enhanced.gif')
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=gif.info.get('duration', 100)
            )
            
            logger.info(f"✅ Mejora de calidad aplicada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"⚠️ Error en mejora de calidad: {e}")
            return input_path
    
    def _optimize_compression(self, input_path: str) -> str:
        """Optimiza la compresión del GIF."""
        try:
            from PIL import Image
            
            # Cargar GIF
            gif = Image.open(input_path)
            
            # Optimizar compresión
            output_path = input_path.replace('.gif', '_optimized.gif')
            gif.save(
                output_path,
                optimize=True,
                quality=85,
                method=6  # Método de compresión más eficiente
            )
            
            logger.info(f"✅ Compresión optimizada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"⚠️ Error en optimización de compresión: {e}")
            return input_path
    
    def _calculate_quality_metrics(self, output_path: str) -> Dict[str, float]:
        """Calcula métricas de calidad del output."""
        try:
            from PIL import Image
            
            gif = Image.open(output_path)
            
            # Métricas básicas
            metrics = {
                'file_size_mb': Path(output_path).stat().st_size / (1024**2),
                'frame_count': gif.n_frames,
                'resolution': gif.size[0] * gif.size[1],
                'duration': gif.info.get('duration', 0) / 1000,  # segundos
                'fps': gif.n_frames / (gif.info.get('duration', 1000) / 1000)
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ Error al calcular métricas de calidad: {e}")
            return {}
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del procesamiento."""
        return {
            'is_processing': self.is_processing,
            'progress': self.current_progress,
            'hardware_info': self.hardware_info,
            'performance_profile': self.performance_profile,
            'current_metrics': self.performance_monitor.get_current_metrics() if self.performance_monitor else None
        }
    
    def stop_processing(self):
        """Detiene el procesamiento actual."""
        if self.is_processing:
            self.is_processing = False
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            logger.info("⏹️ Procesamiento detenido por el usuario")
    
    def get_recommendations(self) -> List[str]:
        """Obtiene recomendaciones basadas en el hardware y configuración."""
        recommendations = []
        
        if self.hardware_info:
            if self.hardware_info.gpu_memory_mb <= 4000:
                recommendations.extend([
                    "🔧 Para mejor rendimiento, cierra otras aplicaciones que usen GPU",
                    "🔧 Procesa videos en segmentos de 15-30 segundos",
                    "🔧 Considera reducir la resolución del video de entrada"
                ])
            
            if self.hardware_info.ram_available_gb < 4:
                recommendations.extend([
                    "🔧 Cierra aplicaciones innecesarias para liberar RAM",
                    "🔧 Considera aumentar la memoria virtual del sistema"
                ])
        
        return recommendations 