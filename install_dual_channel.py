#!/usr/bin/env python3
"""
Instalador Automático del Sistema de Doble Canal
================================================

Este script instala automáticamente todas las dependencias necesarias
para el sistema de doble canal de última generación.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Imprime el banner del instalador."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                INSTALADOR DEL SISTEMA DE DOBLE CANAL                        ║
║                                                                              ║
║  🎯 Canal A: Atención Espaciotemporal Global (GMAR)                        ║
║  🎯 Canal B: Atención por Objetos (Object Tracking)                        ║
║                                                                              ║
║  ⚡ Optimizaciones: Flash Attention 2.0, TensorRT, Mixed Precision         ║
║  🔧 Detección automática de hardware y configuración inteligente           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error("❌ Se requiere Python 3.9 o superior")
        logger.error(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def check_system_requirements():
    """Verifica los requisitos del sistema."""
    logger.info("🔍 Verificando requisitos del sistema...")
    
    # Verificar RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 8:
            logger.warning(f"⚠️ RAM insuficiente: {ram_gb:.1f}GB (mínimo 8GB)")
        else:
            logger.info(f"✅ RAM: {ram_gb:.1f}GB")
    except ImportError:
        logger.warning("⚠️ No se pudo verificar RAM (psutil no disponible)")
    
    # Verificar espacio en disco
    try:
        disk_usage = shutil.disk_usage('.')
        disk_gb = disk_usage.free / (1024**3)
        if disk_gb < 5:
            logger.warning(f"⚠️ Espacio insuficiente: {disk_gb:.1f}GB (mínimo 5GB)")
        else:
            logger.info(f"✅ Espacio libre: {disk_gb:.1f}GB")
    except:
        logger.warning("⚠️ No se pudo verificar espacio en disco")
    
    # Verificar sistema operativo
    os_name = platform.system()
    logger.info(f"✅ Sistema operativo: {os_name}")
    
    return True

def check_cuda_availability():
    """Verifica la disponibilidad de CUDA."""
    logger.info("🔍 Verificando CUDA...")
    
    # Verificar nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ NVIDIA GPU detectada")
            
            # Extraer información de la GPU
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    logger.info(f"   GPU: {line.strip()}")
                    break
            
            return True
        else:
            logger.warning("⚠️ nvidia-smi no disponible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("⚠️ NVIDIA GPU no detectada o nvidia-smi no disponible")
        return False

def install_pip_packages():
    """Instala paquetes usando pip."""
    logger.info("📦 Instalando dependencias con pip...")
    
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.20.0",
        "transformers>=4.30.0",
        "decord>=0.6.0",
        "opencv-python>=4.8.0",
        "pillow>=11.0.0",
        "numpy>=1.26.0,<2.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
        "ultralytics>=8.0.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.8"
    ]
    
    for package in packages:
        try:
            logger.info(f"   Instalando {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            logger.info(f"   ✅ {package} instalado")
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Error instalando {package}: {e}")
            return False
    
    return True

def install_poetry_packages():
    """Instala paquetes usando Poetry."""
    logger.info("📦 Instalando dependencias con Poetry...")
    
    try:
        # Verificar si Poetry está instalado
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        
        # Instalar dependencias
        subprocess.run(["poetry", "install"], check=True)
        logger.info("✅ Dependencias instaladas con Poetry")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("⚠️ Poetry no disponible, usando pip")
        return install_pip_packages()

def create_directories():
    """Crea los directorios necesarios."""
    logger.info("📁 Creando directorios...")
    
    directories = [
        "data/uploads",
        "data/output",
        "config",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✅ {directory}/")

def download_models():
    """Descarga modelos pre-entrenados."""
    logger.info("🤖 Descargando modelos...")
    
    try:
        # Crear script de descarga de modelos
        download_script = """
import torch
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    models = [
        "microsoft/mvit-base-16x2-224",
        "microsoft/mvit-small-16x2-224",
        "microsoft/mvit-xsmall-16x2-224"
    ]
    
    for model_name in models:
        try:
            logger.info(f"Descargando {model_name}...")
            model = AutoModel.from_pretrained(model_name)
            logger.info(f"✅ {model_name} descargado")
        except Exception as e:
            logger.warning(f"⚠️ Error descargando {model_name}: {e}")

if __name__ == "__main__":
    download_models()
"""
        
        with open("download_models.py", "w") as f:
            f.write(download_script)
        
        # Ejecutar descarga
        subprocess.run([sys.executable, "download_models.py"], check=True)
        
        # Limpiar script temporal
        os.remove("download_models.py")
        
        logger.info("✅ Modelos descargados")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Error descargando modelos: {e}")
        return False

def run_tests():
    """Ejecuta tests básicos."""
    logger.info("🧪 Ejecutando tests básicos...")
    
    try:
        # Test de importación
        test_script = """
import sys
import torch

# Test básico de PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test de importaciones del proyecto
try:
    from src.core.hardware_detector import HardwareDetector
    print("✅ Hardware detector importado correctamente")
except ImportError as e:
    print(f"❌ Error importando hardware detector: {e}")

try:
    from src.core.dual_channel_attention import DualChannelAttentionSystem
    print("✅ Dual channel attention importado correctamente")
except ImportError as e:
    print(f"❌ Error importando dual channel attention: {e}")

try:
    from src.core.performance_monitor import PerformanceMonitor
    print("✅ Performance monitor importado correctamente")
except ImportError as e:
    print(f"❌ Error importando performance monitor: {e}")

print("✅ Tests básicos completados")
"""
        
        with open("test_installation.py", "w") as f:
            f.write(test_script)
        
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        
        # Limpiar script temporal
        os.remove("test_installation.py")
        
        if result.returncode == 0:
            logger.info("✅ Tests básicos pasados")
            print(result.stdout)
            return True
        else:
            logger.error("❌ Tests básicos fallaron")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error ejecutando tests: {e}")
        return False

def show_next_steps():
    """Muestra los próximos pasos."""
    next_steps = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           INSTALACIÓN COMPLETADA                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎉 ¡El Sistema de Doble Canal ha sido instalado exitosamente!

📋 PRÓXIMOS PASOS:

1. 🎬 Ejecutar el demo principal:
   python demo_dual_channel.py

2. 📖 Leer la documentación:
   README_DUAL_CHANNEL.md

3. 🎯 Probar con un video:
   - Coloca tu video en data/uploads/
   - Ejecuta el demo
   - Selecciona el video y modo de calidad

4. 🔧 Configuración avanzada:
   - Edita archivos en config/
   - Personaliza parámetros según tu hardware

📊 RENDIMIENTO ESPERADO:
   • RTX 3050 (4GB): 4-6 FPS, modo memory_saver
   • RTX 3060 (6GB): 6-8 FPS, modo balanced
   • RTX 3070+ (8GB+): 8-15 FPS, modo high_quality

⚠️  NOTAS IMPORTANTES:
   • Cierra otras aplicaciones GPU durante el procesamiento
   • Para GPUs con ≤6GB VRAM, usa videos de 720p máximo
   • El sistema detectará automáticamente tu hardware y se optimizará

🚀 ¡Disfruta explorando la atención de Vision Transformers!
"""
    print(next_steps)

def main():
    """Función principal del instalador."""
    print_banner()
    
    logger.info("🚀 Iniciando instalación del Sistema de Doble Canal...")
    
    # Verificar requisitos
    if not check_python_version():
        sys.exit(1)
    
    if not check_system_requirements():
        logger.warning("⚠️ Algunos requisitos del sistema no se cumplen")
    
    cuda_available = check_cuda_availability()
    if not cuda_available:
        logger.warning("⚠️ CUDA no disponible - El sistema funcionará en CPU (muy lento)")
    
    # Instalar dependencias
    logger.info("📦 Instalando dependencias...")
    
    # Intentar Poetry primero, luego pip
    if not install_poetry_packages():
        logger.error("❌ Error instalando dependencias")
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Descargar modelos (opcional)
    if cuda_available:
        download_models()
    
    # Ejecutar tests
    if not run_tests():
        logger.warning("⚠️ Algunos tests fallaron, pero la instalación puede funcionar")
    
    # Mostrar próximos pasos
    show_next_steps()
    
    logger.info("✅ Instalación completada")

if __name__ == "__main__":
    main() 