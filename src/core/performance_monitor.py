"""
Sistema de Monitoreo de Rendimiento en Tiempo Real
==================================================

Este módulo proporciona monitoreo en tiempo real de:
- Uso de GPU y temperatura
- Uso de memoria RAM y VRAM
- FPS y tiempo de procesamiento
- Optimizaciones dinámicas
"""

import time
import psutil
import torch
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Métricas del sistema en tiempo real."""
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_temperature: float = 0.0
    gpu_clock_speed: float = 0.0
    fps: float = 0.0
    processing_time_ms: float = 0.0

@dataclass
class PerformanceAlert:
    """Alerta de rendimiento."""
    level: str  # "info", "warning", "critical"
    message: str
    timestamp: float
    metric: str
    value: float
    threshold: float

class PerformanceMonitor:
    """
    Monitor de rendimiento en tiempo real con optimizaciones dinámicas.
    """
    
    def __init__(
        self,
        config: Dict,
        update_interval: float = 1.0,
        history_size: int = 100,
        enable_alerts: bool = True
    ):
        self.config = config
        self.update_interval = update_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        
        # Métricas históricas
        self.metrics_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=50)
        
        # Estado del sistema
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Callbacks para optimizaciones dinámicas
        self.optimization_callbacks = []
        
        # Umbrales de alerta
        self.thresholds = {
            'gpu_memory_percent': 90.0,
            'gpu_temperature': 80.0,
            'ram_percent': 85.0,
            'fps_drop': 0.5,  # 50% drop en FPS
            'processing_time_ms': 1000.0  # 1 segundo por frame
        }
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            'total_frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_fps': 0.0,
            'peak_gpu_memory': 0.0,
            'peak_temperature': 0.0,
            'optimizations_applied': 0
        }
        
        # Configuración de optimizaciones dinámicas
        self.dynamic_optimizations = {
            'batch_size_reduction': False,
            'resolution_reduction': False,
            'mixed_precision': True,
            'gradient_checkpointing': False
        }
        
        logger.info("✅ Monitor de rendimiento inicializado")
    
    def start_monitoring(self):
        """Inicia el monitoreo en tiempo real."""
        if self.is_monitoring:
            logger.warning("⚠️ El monitoreo ya está activo")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 Monitoreo de rendimiento iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("⏹️ Monitoreo de rendimiento detenido")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Verificar alertas
                if self.enable_alerts:
                    alerts = self._check_alerts(metrics)
                    for alert in alerts:
                        self.alerts_history.append(alert)
                        self._handle_alert(alert)
                
                # Aplicar optimizaciones dinámicas
                self._apply_dynamic_optimizations(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Error en monitoreo: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Recopila métricas del sistema."""
        timestamp = time.time()
        
        # Métricas de CPU y RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_available_gb = ram.available / (1024**3)
        
        # Métricas de GPU
        gpu_percent = 0.0
        gpu_memory_percent = 0.0
        gpu_memory_used_gb = 0.0
        gpu_temperature = 0.0
        gpu_clock_speed = 0.0
        
        if torch.cuda.is_available():
            try:
                # Uso de GPU
                gpu_percent = torch.cuda.utilization()
                
                # Memoria GPU
                gpu_memory = torch.cuda.memory_stats()
                gpu_memory_used_gb = gpu_memory['allocated_bytes.all.current'] / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_percent = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
                
                # Temperatura GPU (requiere nvidia-smi)
                gpu_temperature = self._get_gpu_temperature()
                
                # Velocidad de reloj GPU
                gpu_clock_speed = self._get_gpu_clock_speed()
                
            except Exception as e:
                logger.warning(f"⚠️ Error al obtener métricas GPU: {e}")
        
        # Calcular FPS y tiempo de procesamiento
        fps = self._calculate_fps()
        processing_time_ms = self._calculate_processing_time()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            ram_available_gb=ram_available_gb,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_temperature=gpu_temperature,
            gpu_clock_speed=gpu_clock_speed,
            fps=fps,
            processing_time_ms=processing_time_ms
        )
        
        return metrics
    
    def _get_gpu_temperature(self) -> float:
        """Obtiene la temperatura de la GPU usando nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def _get_gpu_clock_speed(self) -> float:
        """Obtiene la velocidad de reloj de la GPU."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=clocks.current.graphics', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0
    
    def _calculate_fps(self) -> float:
        """Calcula FPS basado en el tiempo de procesamiento."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Calcular FPS basado en el tiempo entre frames
        recent_metrics = list(self.metrics_history)[-10:]  # Últimos 10 frames
        if len(recent_metrics) >= 2:
            time_diff = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            if time_diff > 0:
                return len(recent_metrics) / time_diff
        
        return 0.0
    
    def _calculate_processing_time(self) -> float:
        """Calcula el tiempo de procesamiento por frame."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Promedio de tiempo de procesamiento de los últimos frames
        recent_metrics = list(self.metrics_history)[-5:]
        if len(recent_metrics) >= 2:
            total_time = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            return (total_time / len(recent_metrics)) * 1000  # Convertir a ms
        
        return 0.0
    
    def _check_alerts(self, metrics: SystemMetrics) -> List[PerformanceAlert]:
        """Verifica si hay alertas basadas en las métricas."""
        alerts = []
        
        # Alerta de memoria GPU
        if metrics.gpu_memory_percent > self.thresholds['gpu_memory_percent']:
            alerts.append(PerformanceAlert(
                level="warning",
                message=f"Uso alto de VRAM: {metrics.gpu_memory_percent:.1f}%",
                timestamp=metrics.timestamp,
                metric="gpu_memory_percent",
                value=metrics.gpu_memory_percent,
                threshold=self.thresholds['gpu_memory_percent']
            ))
        
        # Alerta de temperatura GPU
        if metrics.gpu_temperature > self.thresholds['gpu_temperature']:
            alerts.append(PerformanceAlert(
                level="critical",
                message=f"Temperatura GPU alta: {metrics.gpu_temperature:.1f}°C",
                timestamp=metrics.timestamp,
                metric="gpu_temperature",
                value=metrics.gpu_temperature,
                threshold=self.thresholds['gpu_temperature']
            ))
        
        # Alerta de RAM
        if metrics.ram_percent > self.thresholds['ram_percent']:
            alerts.append(PerformanceAlert(
                level="warning",
                message=f"Uso alto de RAM: {metrics.ram_percent:.1f}%",
                timestamp=metrics.timestamp,
                metric="ram_percent",
                value=metrics.ram_percent,
                threshold=self.thresholds['ram_percent']
            ))
        
        # Alerta de FPS bajo
        if metrics.fps > 0 and len(self.metrics_history) >= 10:
            avg_fps = sum(m.fps for m in list(self.metrics_history)[-10:]) / 10
            if metrics.fps < avg_fps * self.thresholds['fps_drop']:
                alerts.append(PerformanceAlert(
                    level="warning",
                    message=f"FPS bajo detectado: {metrics.fps:.1f} (promedio: {avg_fps:.1f})",
                    timestamp=metrics.timestamp,
                    metric="fps",
                    value=metrics.fps,
                    threshold=avg_fps * self.thresholds['fps_drop']
                ))
        
        # Alerta de tiempo de procesamiento alto
        if metrics.processing_time_ms > self.thresholds['processing_time_ms']:
            alerts.append(PerformanceAlert(
                level="warning",
                message=f"Tiempo de procesamiento alto: {metrics.processing_time_ms:.1f}ms",
                timestamp=metrics.timestamp,
                metric="processing_time_ms",
                value=metrics.processing_time_ms,
                threshold=self.thresholds['processing_time_ms']
            ))
        
        return alerts
    
    def _handle_alert(self, alert: PerformanceAlert):
        """Maneja una alerta de rendimiento."""
        if alert.level == "critical":
            logger.critical(f"🚨 {alert.message}")
        elif alert.level == "warning":
            logger.warning(f"⚠️ {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.message}")
    
    def _apply_dynamic_optimizations(self, metrics: SystemMetrics):
        """Aplica optimizaciones dinámicas basadas en las métricas."""
        optimizations_applied = []
        
        # Reducir batch size si la memoria GPU está alta
        if (metrics.gpu_memory_percent > 85 and 
            not self.dynamic_optimizations['batch_size_reduction']):
            
            self._reduce_batch_size()
            optimizations_applied.append("batch_size_reduction")
        
        # Reducir resolución si la temperatura está alta
        if (metrics.gpu_temperature > 75 and 
            not self.dynamic_optimizations['resolution_reduction']):
            
            self._reduce_resolution()
            optimizations_applied.append("resolution_reduction")
        
        # Activar gradient checkpointing si la memoria está muy alta
        if (metrics.gpu_memory_percent > 90 and 
            not self.dynamic_optimizations['gradient_checkpointing']):
            
            self._enable_gradient_checkpointing()
            optimizations_applied.append("gradient_checkpointing")
        
        # Limpiar caché GPU si es necesario
        if metrics.gpu_memory_percent > 95:
            self._clear_gpu_cache()
            optimizations_applied.append("gpu_cache_clear")
        
        if optimizations_applied:
            self.performance_stats['optimizations_applied'] += len(optimizations_applied)
            logger.info(f"🔧 Optimizaciones aplicadas: {', '.join(optimizations_applied)}")
    
    def _reduce_batch_size(self):
        """Reduce el tamaño del batch dinámicamente."""
        current_batch_size = self.config['hardware']['batch_size']
        if current_batch_size > 1:
            new_batch_size = max(1, current_batch_size // 2)
            self.config['hardware']['batch_size'] = new_batch_size
            self.dynamic_optimizations['batch_size_reduction'] = True
            logger.info(f"🔧 Batch size reducido a {new_batch_size}")
    
    def _reduce_resolution(self):
        """Reduce la resolución dinámicamente."""
        current_res = self.config['video']['max_resolution']
        if current_res[0] > 480:
            new_res = (max(480, current_res[0] // 2), max(480, current_res[1] // 2))
            self.config['video']['max_resolution'] = new_res
            self.dynamic_optimizations['resolution_reduction'] = True
            logger.info(f"🔧 Resolución reducida a {new_res[0]}x{new_res[1]}")
    
    def _enable_gradient_checkpointing(self):
        """Activa gradient checkpointing."""
        self.config['attention']['channel_a']['gradient_checkpointing'] = True
        self.dynamic_optimizations['gradient_checkpointing'] = True
        logger.info("🔧 Gradient checkpointing activado")
    
    def _clear_gpu_cache(self):
        """Limpia la caché de GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Caché GPU limpiada")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Obtiene las métricas actuales."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_performance_summary(self) -> Dict:
        """Obtiene un resumen del rendimiento."""
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        
        # Calcular estadísticas
        gpu_memory_usage = [m.gpu_memory_percent for m in metrics_list if m.gpu_memory_percent > 0]
        temperatures = [m.gpu_temperature for m in metrics_list if m.gpu_temperature > 0]
        fps_values = [m.fps for m in metrics_list if m.fps > 0]
        
        summary = {
            'monitoring_duration': metrics_list[-1].timestamp - metrics_list[0].timestamp,
            'total_samples': len(metrics_list),
            'avg_gpu_memory_percent': sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0,
            'peak_gpu_memory_percent': max(gpu_memory_usage) if gpu_memory_usage else 0,
            'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else 0,
            'peak_temperature': max(temperatures) if temperatures else 0,
            'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            'min_fps': min(fps_values) if fps_values else 0,
            'max_fps': max(fps_values) if fps_values else 0,
            'total_alerts': len(self.alerts_history),
            'optimizations_applied': self.performance_stats['optimizations_applied']
        }
        
        return summary
    
    def print_status_display(self):
        """Imprime el display de estado en tiempo real."""
        metrics = self.get_current_metrics()
        if not metrics:
            return
        
        # Calcular tiempo restante estimado
        fps = metrics.fps if metrics.fps > 0 else 1
        frames_remaining = self.performance_stats.get('frames_remaining', 0)
        time_remaining = frames_remaining / fps if fps > 0 else 0
        
        status_display = f"""
┌─────────────────────────────────────────────────────────┐
│                    MONITOR DE RENDIMIENTO               │
├─────────────────────────────────────────────────────────┤
│ GPU: {metrics.gpu_percent:>3.0f}% | VRAM: {metrics.gpu_memory_percent:>5.1f}% | T°: {metrics.gpu_temperature:>4.0f}°C │
│ RAM: {metrics.ram_percent:>3.0f}% ({metrics.ram_available_gb:>4.1f}GB disponible)                    │
│                                                         │
│ FPS actual: {metrics.fps:>6.1f} | Tiempo/frame: {metrics.processing_time_ms:>6.1f}ms        │
│ Tiempo restante: {time_remaining:>6.0f}s                              │
│                                                         │
│ Optimizaciones aplicadas: {self.performance_stats['optimizations_applied']:>2d}                    │
│ Alertas totales: {len(self.alerts_history):>2d}                                    │
└─────────────────────────────────────────────────────────┘
"""
        print(status_display)
    
    def add_optimization_callback(self, callback: Callable[[Dict], None]):
        """Añade un callback para optimizaciones dinámicas."""
        self.optimization_callbacks.append(callback)
    
    def update_frames_remaining(self, frames: int):
        """Actualiza el número de frames restantes."""
        self.performance_stats['frames_remaining'] = frames 