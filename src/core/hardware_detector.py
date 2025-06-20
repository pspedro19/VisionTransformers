"""
Sistema de Detección Automática de Hardware y Optimización
==========================================================

Este módulo detecta automáticamente las capacidades del hardware y selecciona
la configuración óptima para el procesamiento de videos con atención visual.
"""

import os
import sys
import platform
import subprocess
import psutil
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """Información detallada del hardware del sistema."""
    gpu_name: str
    gpu_memory_mb: int
    cuda_version: Optional[str]
    cuda_available: bool
    cpu_cores: int
    cpu_name: str
    ram_gb: int
    ram_available_gb: int
    disk_space_gb: int
    os_name: str
    python_version: str
    torch_version: str

@dataclass
class PerformanceProfile:
    """Perfil de rendimiento recomendado para el hardware detectado."""
    config_name: str
    batch_size: int
    max_resolution: Tuple[int, int]
    target_fps: int
    estimated_processing_time: float  # segundos por minuto de video
    memory_usage_mb: int
    quality_level: str

class HardwareDetector:
    """Detector automático de hardware y optimizador de configuración."""
    
    def __init__(self):
        self.hardware_info = None
        self.performance_profile = None
        self.config_path = Path(__file__).parent.parent.parent / "config"
        
    def detect_hardware(self) -> HardwareInfo:
        """Detecta automáticamente las capacidades del hardware."""
        logger.info("🔍 Iniciando detección automática de hardware...")
        
        # Información del sistema
        os_name = platform.system()
        python_version = sys.version.split()[0]
        torch_version = torch.__version__
        
        # Información de CPU
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_name = self._get_cpu_name()
        
        # Información de RAM
        ram_gb = psutil.virtual_memory().total // (1024**3)
        ram_available_gb = psutil.virtual_memory().available // (1024**3)
        
        # Información de disco
        disk_space_gb = psutil.disk_usage('/').free // (1024**3)
        
        # Información de GPU
        gpu_info = self._detect_gpu()
        
        hardware_info = HardwareInfo(
            gpu_name=gpu_info['name'],
            gpu_memory_mb=gpu_info['memory_mb'],
            cuda_version=gpu_info['cuda_version'],
            cuda_available=torch.cuda.is_available(),
            cpu_cores=cpu_cores,
            cpu_name=cpu_name,
            ram_gb=ram_gb,
            ram_available_gb=ram_available_gb,
            disk_space_gb=disk_space_gb,
            os_name=os_name,
            python_version=python_version,
            torch_version=torch_version
        )
        
        self.hardware_info = hardware_info
        return hardware_info
    
    def _get_cpu_name(self) -> str:
        """Obtiene el nombre del procesador."""
        try:
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            elif platform.system() == "Darwin":
                return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        except:
            pass
        return "CPU Desconocido"
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detecta información de la GPU."""
        gpu_info = {
            'name': 'No GPU detectada',
            'memory_mb': 0,
            'cuda_version': None
        }
        
        # Detectar CUDA
        if torch.cuda.is_available():
            try:
                gpu_info['name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                
                # Obtener versión de CUDA
                cuda_version = torch.version.cuda
                if cuda_version:
                    gpu_info['cuda_version'] = cuda_version
                    
                logger.info(f"✅ GPU detectada: {gpu_info['name']} ({gpu_info['memory_mb']}MB VRAM)")
                
            except Exception as e:
                logger.warning(f"⚠️ Error al detectar GPU: {e}")
        else:
            logger.warning("⚠️ CUDA no disponible - Procesamiento en CPU")
            
        return gpu_info
    
    def select_optimal_config(self) -> PerformanceProfile:
        """Selecciona la configuración óptima basada en el hardware detectado."""
        if not self.hardware_info:
            self.detect_hardware()
        
        gpu_memory = self.hardware_info.gpu_memory_mb
        gpu_name = self.hardware_info.gpu_name.lower()
        
        # Mapeo de configuraciones por GPU
        if gpu_memory >= 16000 or "rtx 4090" in gpu_name or "rtx 4080" in gpu_name:
            config_name = "ultra_quality"
            batch_size = 16
            max_resolution = (3840, 2160)
            target_fps = 25
            estimated_time = 2.0
            memory_usage = 22000
            quality_level = "Ultra"
            
        elif gpu_memory >= 10000 or "rtx 4070" in gpu_name or "rtx 3080" in gpu_name:
            config_name = "high_quality"
            batch_size = 8
            max_resolution = (2560, 1440)
            target_fps = 20
            estimated_time = 3.0
            memory_usage = 10000
            quality_level = "Alta"
            
        elif gpu_memory >= 6000 or "rtx 3070" in gpu_name or "rtx 3060" in gpu_name:
            config_name = "balanced"
            batch_size = 4
            max_resolution = (1920, 1080)
            target_fps = 15
            estimated_time = 4.0
            memory_usage = 6000
            quality_level = "Balanceada"
            
        elif gpu_memory >= 4000 or "rtx 3050" in gpu_name or "gtx 1660" in gpu_name:
            config_name = "memory_saver"
            batch_size = 2
            max_resolution = (1280, 720)
            target_fps = 6
            estimated_time = 6.0
            memory_usage = 3500
            quality_level = "Optimizada"
            
        else:
            # Configuración para CPU o GPU muy limitada
            config_name = "cpu_only"
            batch_size = 1
            max_resolution = (854, 480)
            target_fps = 4
            estimated_time = 15.0
            memory_usage = 1000
            quality_level = "Básica"
        
        profile = PerformanceProfile(
            config_name=config_name,
            batch_size=batch_size,
            max_resolution=max_resolution,
            target_fps=target_fps,
            estimated_processing_time=estimated_time,
            memory_usage_mb=memory_usage,
            quality_level=quality_level
        )
        
        self.performance_profile = profile
        return profile
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Carga la configuración YAML especificada."""
        config_file = self.config_path / f"{config_name}.yaml"
        
        if not config_file.exists():
            logger.error(f"❌ Archivo de configuración no encontrado: {config_file}")
            raise FileNotFoundError(f"Configuración {config_name} no encontrada")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Configuración cargada: {config_name}")
        return config
    
    def generate_hardware_report(self) -> str:
        """Genera un reporte detallado del hardware y configuración."""
        if not self.hardware_info:
            self.detect_hardware()
        
        if not self.performance_profile:
            self.select_optimal_config()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ANÁLISIS DE HARDWARE COMPLETADO          ║
╠══════════════════════════════════════════════════════════════╣
║  GPU: {self.hardware_info.gpu_name:<45} ║
║  VRAM: {self.hardware_info.gpu_memory_mb}MB{' ' * (44-len(str(self.hardware_info.gpu_memory_mb)))} ║
║  CUDA: {self.hardware_info.cuda_version or 'No disponible':<45} ║
║  CPU: {self.hardware_info.cpu_name:<45} ║
║  Cores: {self.hardware_info.cpu_cores}{' ' * (44-len(str(self.hardware_info.cpu_cores)))} ║
║  RAM: {self.hardware_info.ram_gb}GB (Disponible: {self.hardware_info.ram_available_gb}GB){' ' * (20-len(str(self.hardware_info.ram_gb))-len(str(self.hardware_info.ram_available_gb)))} ║
║  OS: {self.hardware_info.os_name}{' ' * (44-len(self.hardware_info.os_name))} ║
║                                                              ║
║  CONFIGURACIÓN SELECCIONADA:                                 ║
║  > {self.performance_profile.config_name}.yaml{' ' * (35-len(self.performance_profile.config_name))} ║
║  > Batch size: {self.performance_profile.batch_size}{' ' * (40-len(str(self.performance_profile.batch_size)))} ║
║  > Resolución: {self.performance_profile.max_resolution[0]}x{self.performance_profile.max_resolution[1]}{' ' * (35-len(str(self.performance_profile.max_resolution[0]))-len(str(self.performance_profile.max_resolution[1])))} ║
║  > FPS objetivo: {self.performance_profile.target_fps}{' ' * (35-len(str(self.performance_profile.target_fps)))} ║
║  > Calidad: {self.performance_profile.quality_level}{' ' * (40-len(self.performance_profile.quality_level))} ║
║  > Tiempo estimado: {self.performance_profile.estimated_processing_time:.1f}s/min{' ' * (25-len(f'{self.performance_profile.estimated_processing_time:.1f}'))} ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report
    
    def validate_system_requirements(self) -> Tuple[bool, List[str]]:
        """Valida que el sistema cumple con los requisitos mínimos."""
        issues = []
        
        # Verificar RAM mínima
        if self.hardware_info.ram_gb < 8:
            issues.append(f"RAM insuficiente: {self.hardware_info.ram_gb}GB (mínimo 8GB)")
        
        # Verificar espacio en disco
        if self.hardware_info.disk_space_gb < 5:
            issues.append(f"Espacio en disco insuficiente: {self.hardware_info.disk_space_gb}GB (mínimo 5GB)")
        
        # Verificar CUDA para GPU
        if self.hardware_info.gpu_memory_mb > 0 and not self.hardware_info.cuda_available:
            issues.append("GPU detectada pero CUDA no disponible")
        
        # Verificar versión de Python
        python_version = tuple(map(int, self.hardware_info.python_version.split('.')[:2]))
        if python_version < (3, 9):
            issues.append(f"Versión de Python muy antigua: {self.hardware_info.python_version} (mínimo 3.9)")
        
        return len(issues) == 0, issues
    
    def get_optimization_recommendations(self) -> List[str]:
        """Genera recomendaciones de optimización específicas para el hardware."""
        recommendations = []
        
        if self.hardware_info.gpu_memory_mb <= 4000:
            recommendations.extend([
                "🔧 Cerrar otras aplicaciones que usen GPU (juegos, navegadores)",
                "🔧 Procesar videos en segmentos de 15-30 segundos",
                "🔧 Usar el modo 'memory_saver' para máxima compatibilidad",
                "🔧 Considerar reducir la resolución del video de entrada"
            ])
        
        if self.hardware_info.ram_available_gb < 4:
            recommendations.extend([
                "🔧 Cerrar aplicaciones innecesarias para liberar RAM",
                "🔧 Considerar aumentar la memoria virtual del sistema"
            ])
        
        if self.hardware_info.disk_space_gb < 10:
            recommendations.extend([
                "🔧 Limpiar archivos temporales del sistema",
                "🔧 Procesar videos uno por uno para ahorrar espacio"
            ])
        
        if not self.hardware_info.cuda_available and self.hardware_info.gpu_memory_mb > 0:
            recommendations.extend([
                "🔧 Instalar drivers NVIDIA actualizados",
                "🔧 Verificar instalación de CUDA Toolkit",
                "🔧 Reinstalar PyTorch con soporte CUDA"
            ])
        
        return recommendations

def auto_detect_and_configure() -> Tuple[HardwareInfo, PerformanceProfile, Dict[str, Any]]:
    """
    Función principal para detección automática y configuración.
    
    Returns:
        Tuple con información de hardware, perfil de rendimiento y configuración
    """
    detector = HardwareDetector()
    
    # Detectar hardware
    hardware_info = detector.detect_hardware()
    
    # Seleccionar configuración óptima
    performance_profile = detector.select_optimal_config()
    
    # Validar requisitos
    is_valid, issues = detector.validate_system_requirements()
    if not is_valid:
        logger.warning("⚠️ Problemas detectados en el sistema:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Cargar configuración
    config = detector.load_config(performance_profile.config_name)
    
    # Mostrar reporte
    print(detector.generate_hardware_report())
    
    # Mostrar recomendaciones si hay problemas
    if not is_valid or hardware_info.gpu_memory_mb <= 4000:
        recommendations = detector.get_optimization_recommendations()
        if recommendations:
            print("\n🔧 RECOMENDACIONES DE OPTIMIZACIÓN:")
            for rec in recommendations:
                print(f"  {rec}")
    
    return hardware_info, performance_profile, config 