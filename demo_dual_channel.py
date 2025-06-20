#!/usr/bin/env python3
"""
Demo del Sistema de Doble Canal de Última Generación
====================================================

Este script demuestra el sistema completo de doble canal con:
- Detección automática de hardware
- Procesamiento optimizado
- Monitoreo en tiempo real
- Visualización de resultados
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dual_channel_demo.log')
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Imprime el banner del sistema."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SISTEMA DE DOBLE CANAL DE ÚLTIMA GENERACIÓN              ║
║                                                                              ║
║  🎯 Canal A: Atención Espaciotemporal Global (GMAR)                        ║
║  🎯 Canal B: Atención por Objetos (Object Tracking)                        ║
║                                                                              ║
║  ⚡ Optimizaciones: Flash Attention 2.0, TensorRT, Mixed Precision         ║
║  🔧 Detección automática de hardware y configuración inteligente           ║
║  📊 Monitoreo en tiempo real con optimizaciones dinámicas                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_hardware_requirements():
    """Imprime los requisitos de hardware."""
    requirements = """
📋 REQUISITOS DE HARDWARE:

┌─────────────────────────────────────────────────────────────────────────────┐
│ GPU Recomendadas:                                                          │
│ • RTX 4090 (24GB) - Ultra Quality, 4K, 20-25 FPS                          │
│ • RTX 4080 (16GB) - High Quality, 1440p, 15-20 FPS                        │
│ • RTX 4070 Ti (12GB) - High Quality, 1080p, 12-15 FPS                     │
│ • RTX 3080 (10GB) - Balanced, 1080p, 10-12 FPS                            │
│ • RTX 3070 (8GB) - Balanced, 720p, 8-10 FPS                               │
│ • RTX 3060 (6GB) - Optimized, 720p, 6-8 FPS                               │
│ • RTX 3050 (4GB) - Memory Saver, 480p→720p, 4-6 FPS                       │
│ • GTX 1660 (6GB) - Legacy, 480p, 3-5 FPS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ RAM Mínima: 8GB (Recomendado: 16GB+)                                      │
│ CPU: 4+ cores (Recomendado: 8+ cores)                                     │
│ Espacio: 5GB+ libre para archivos temporales                              │
│ CUDA: 11.8+ (para GPUs NVIDIA)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(requirements)

def get_video_input() -> Optional[str]:
    """Obtiene la ruta del video de entrada del usuario."""
    print("\n🎬 SELECCIÓN DE VIDEO:")
    print("=" * 50)
    
    # Verificar si hay videos en el directorio data/uploads
    uploads_dir = Path("data/uploads")
    if uploads_dir.exists():
        video_files = list(uploads_dir.glob("*.mp4")) + list(uploads_dir.glob("*.avi")) + list(uploads_dir.glob("*.mov"))
        
        if video_files:
            print("📁 Videos encontrados en data/uploads:")
            for i, video_file in enumerate(video_files, 1):
                size_mb = video_file.stat().st_size / (1024 * 1024)
                print(f"  {i}. {video_file.name} ({size_mb:.1f} MB)")
            
            print(f"  {len(video_files) + 1}. Especificar ruta manual")
            
            while True:
                try:
                    choice = input(f"\nSelecciona un video (1-{len(video_files) + 1}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(video_files):
                        return str(video_files[choice_num - 1])
                    elif choice_num == len(video_files) + 1:
                        break
                    else:
                        print("❌ Opción inválida")
                except ValueError:
                    print("❌ Por favor ingresa un número válido")
    
    # Solicitar ruta manual
    while True:
        video_path = input("\n📂 Ingresa la ruta completa del video: ").strip()
        
        if not video_path:
            print("❌ La ruta no puede estar vacía")
            continue
        
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"❌ El archivo no existe: {video_path}")
            continue
        
        if video_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            print("❌ Formato de video no soportado. Usa: .mp4, .avi, .mov, .mkv")
            continue
        
        return str(video_file)

def select_quality_mode() -> str:
    """Permite al usuario seleccionar el modo de calidad."""
    print("\n🎛️  MODO DE CALIDAD:")
    print("=" * 50)
    
    modes = {
        "1": {
            "name": "RÁPIDO",
            "description": "Procesamiento rápido con calidad media",
            "time": "2-3 minutos",
            "quality": "Media",
            "resolution": "480p"
        },
        "2": {
            "name": "BALANCEADO",
            "description": "Balance entre velocidad y calidad (Recomendado)",
            "time": "4-5 minutos",
            "quality": "Buena",
            "resolution": "720p upscaled"
        },
        "3": {
            "name": "CALIDAD",
            "description": "Máxima calidad posible",
            "time": "8-10 minutos",
            "quality": "Máxima",
            "resolution": "720p nativa"
        }
    }
    
    for key, mode in modes.items():
        print(f"  {key}. {mode['name']}")
        print(f"     ⏱️  Tiempo: {mode['time']}")
        print(f"     🎨 Calidad: {mode['quality']}")
        print(f"     📐 Resolución: {mode['resolution']}")
        print(f"     📝 {mode['description']}")
        print()
    
    while True:
        choice = input("Selecciona el modo (1-3): ").strip()
        if choice in modes:
            selected_mode = modes[choice]
            print(f"\n✅ Modo seleccionado: {selected_mode['name']}")
            print(f"   Tiempo estimado: {selected_mode['time']}")
            return choice
        else:
            print("❌ Opción inválida. Selecciona 1, 2 o 3.")

def progress_callback(progress: float):
    """Callback para mostrar el progreso."""
    bar_length = 40
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r🔄 Progreso: {bar} {progress:.1f}%", end='', flush=True)
    
    if progress >= 100:
        print()  # Nueva línea al completar

def main():
    """Función principal del demo."""
    print_banner()
    print_hardware_requirements()
    
    # Verificar si el usuario quiere continuar
    print("\n⚠️  ADVERTENCIA:")
    print("Este sistema utiliza recursos intensivos de GPU.")
    print("Asegúrate de tener una GPU compatible y drivers actualizados.")
    
    continue_choice = input("\n¿Continuar con la demostración? (s/N): ").strip().lower()
    if continue_choice not in ['s', 'si', 'sí', 'y', 'yes']:
        print("👋 Demo cancelado. ¡Hasta luego!")
        return
    
    try:
        # Importar el pipeline optimizado
        from src.core.optimized_pipeline import OptimizedDualChannelPipeline
        
        # Obtener entrada del usuario
        video_path = get_video_input()
        if not video_path:
            print("❌ No se seleccionó ningún video")
            return
        
        quality_mode = select_quality_mode()
        
        print(f"\n🚀 INICIANDO PROCESAMIENTO:")
        print("=" * 50)
        print(f"📹 Video: {Path(video_path).name}")
        print(f"🎛️  Modo: {quality_mode}")
        print(f"⏰ Iniciando a las: {time.strftime('%H:%M:%S')}")
        
        # Inicializar pipeline
        pipeline = OptimizedDualChannelPipeline()
        
        # Inicializar sistema con detección automática
        print("\n🔍 DETECTANDO HARDWARE...")
        hardware_info, performance_profile, config = pipeline.initialize_system()
        
        # Mostrar recomendaciones
        recommendations = pipeline.get_recommendations()
        if recommendations:
            print("\n🔧 RECOMENDACIONES:")
            for rec in recommendations:
                print(f"  {rec}")
        
        # Procesar video
        print(f"\n🎬 PROCESANDO VIDEO...")
        print("Presiona Ctrl+C para detener el procesamiento")
        
        result = pipeline.process_video(
            video_path=video_path,
            progress_callback=progress_callback
        )
        
        # Mostrar resultados
        print(f"\n✅ PROCESAMIENTO COMPLETADO!")
        print("=" * 50)
        print(f"📁 Archivo generado: {result.output_path}")
        print(f"⏱️  Tiempo total: {result.processing_time:.2f} segundos")
        print(f"🎬 Frames procesados: {result.frames_processed}")
        print(f"📊 FPS promedio: {result.avg_fps:.1f}")
        
        # Métricas de calidad
        if result.quality_metrics:
            print(f"\n📈 MÉTRICAS DE CALIDAD:")
            print(f"  📏 Tamaño del archivo: {result.quality_metrics.get('file_size_mb', 0):.1f} MB")
            print(f"  🎬 Número de frames: {result.quality_metrics.get('frame_count', 0)}")
            print(f"  📐 Resolución: {result.quality_metrics.get('resolution', 0)} píxeles")
            print(f"  ⏱️  Duración: {result.quality_metrics.get('duration', 0):.1f} segundos")
            print(f"  🎯 FPS del GIF: {result.quality_metrics.get('fps', 0):.1f}")
        
        # Resumen de rendimiento
        if result.performance_summary:
            print(f"\n📊 RESUMEN DE RENDIMIENTO:")
            print(f"  🔥 Pico de memoria GPU: {result.performance_summary.get('peak_gpu_memory_percent', 0):.1f}%")
            print(f"  🌡️  Temperatura máxima: {result.performance_summary.get('peak_temperature', 0):.1f}°C")
            print(f"  ⚡ Optimizaciones aplicadas: {result.performance_summary.get('optimizations_applied', 0)}")
            print(f"  ⚠️  Alertas totales: {result.performance_summary.get('total_alerts', 0)}")
        
        print(f"\n🎉 ¡Procesamiento completado exitosamente!")
        print(f"📂 El GIF está disponible en: {result.output_path}")
        
        # Abrir el archivo si es posible
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(result.output_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", result.output_path])
            else:  # Linux
                subprocess.run(["xdg-open", result.output_path])
                
            print("🔍 Abriendo archivo generado...")
        except:
            print("💡 Puedes abrir manualmente el archivo generado para ver el resultado.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Procesamiento detenido por el usuario")
        if 'pipeline' in locals():
            pipeline.stop_processing()
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        logger.error(f"Error en demo: {e}", exc_info=True)
    
    print("\n👋 ¡Gracias por usar el Sistema de Doble Canal!")

if __name__ == "__main__":
    main() 