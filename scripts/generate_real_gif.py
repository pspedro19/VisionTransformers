#!/usr/bin/env python3
"""
Script para generar GIFs reales con atención visual usando Vision Transformers.
Este script usa el pipeline completo para procesar videos y generar GIFs con overlays de atención.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.pipeline import InMemoryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_gif_with_attention():
    """Genera un GIF real con atención visual usando el pipeline completo."""
    logger.info("Generando GIF con atención visual usando Vision Transformers")
    logger.info("=" * 60)
    
    # Inicializar pipeline
    try:
        pipeline = InMemoryPipeline("config/mvp2.yaml")
        logger.info("Pipeline inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar pipeline: {e}")
        return False
    
    # Buscar videos disponibles
    uploads_dir = Path("data/uploads")
    sample_videos = list(uploads_dir.glob("*.mp4"))
    
    if not sample_videos:
        logger.error("No se encontraron videos en data/uploads/")
        return False
    
    # Usar el primer video disponible
    video_path = str(sample_videos[0])
    logger.info(f"Procesando video: {Path(video_path).name}")
    
    # Crear directorio de salida
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    
    # Configuración para generar GIF con atención
    config = {
        "gif": {
            "fps": 8,
            "max_frames": 20,
            "overlay_style": "heatmap",
            "overlay_intensity": 0.8
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 1,
            "max_stride": 2
        }
    }
    
    output_path = str(output_dir / f"real_attention_{Path(video_path).stem}.gif")
    
    try:
        logger.info("Iniciando procesamiento con atención visual...")
        
        # Procesar video con atención visual
        result = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            override_config=config
        )
        
        if result.get('success'):
            logger.info(f"GIF generado exitosamente: {Path(output_path).name}")
            
            # Mostrar estadísticas disponibles
            if 'gif_stats' in result:
                stats = result['gif_stats']
                logger.info(f"Frames procesados: {stats.get('frame_count', 'N/A')}")
                logger.info(f"Resolución: {stats.get('width', 'N/A')}x{stats.get('height', 'N/A')}")
                if 'file_size_mb' in stats:
                    logger.info(f"Tamaño del archivo: {stats['file_size_mb']:.2f} MB")
            
            if 'processing_time' in result:
                logger.info(f"Tiempo de procesamiento: {result['processing_time']:.1f}s")
            
            # Verificar que el archivo existe
            if Path(output_path).exists():
                file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                logger.info(f"Archivo verificado: {file_size_mb:.2f} MB")
                return True
            else:
                logger.error("El archivo GIF no fue creado")
                return False
        else:
            logger.error("El procesamiento falló")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal."""
    logger.info("Iniciando generación de GIF con atención visual")
    
    success = generate_gif_with_attention()
    
    if success:
        logger.info("=" * 60)
        logger.info("¡GIF con atención visual generado exitosamente!")
        logger.info("Revisa data/output/ para ver el resultado")
    else:
        logger.error("Falló la generación del GIF")
        sys.exit(1)

if __name__ == "__main__":
    main() 