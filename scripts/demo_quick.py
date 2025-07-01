#!/usr/bin/env python3
"""
Quick Demo Script for Vision Transformers GIF Generator
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

def demo_system():
    """Demonstrate the system with different configurations."""
    logger.info("Vision Transformers GIF Generator Demo")
    logger.info("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = InMemoryPipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Find available videos
    uploads_dir = Path("data/uploads")
    sample_videos = list(uploads_dir.glob("*.mp4"))
    
    if not sample_videos:
        logger.error("No sample videos found")
        return
    
    # Choose a video for demo
    video_path = str(sample_videos[1])
    logger.info(f"Using video: {Path(video_path).name}")
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    
    # Demo 1: Fast GIF with heatmap overlay
    logger.info("\nDemo 1: Fast GIF with Heatmap Overlay")
    config1 = {
        "gif": {
            "fps": 10,
            "max_frames": 15,
            "overlay_style": "heatmap",
            "overlay_intensity": 0.8
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        }
    }
    
    output_path1 = str(output_dir / f"demo1_{Path(video_path).stem}_heatmap.gif")
    
    try:
        results1 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path1,
            override_config=config1
        )
        
        if results1.get('success'):
            logger.info(f"Demo 1 completed: {Path(output_path1).name}")
            logger.info(f"Size: {results1['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"Time: {results1['processing_time']:.1f}s")
        else:
            logger.error("Demo 1 failed")
            
    except Exception as e:
        logger.error(f"Demo 1 error: {e}")
    
    # Demo 2: Highlight overlay with different settings
    logger.info("\nDemo 2: Highlight Overlay")
    config2 = {
        "gif": {
            "fps": 8,
            "max_frames": 20,
            "overlay_style": "highlight",
            "overlay_intensity": 0.6
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        }
    }
    
    output_path2 = str(output_dir / f"demo2_{Path(video_path).stem}_highlight.gif")
    
    try:
        results2 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path2,
            override_config=config2
        )
        
        if results2.get('success'):
            logger.info(f"Demo 2 completed: {Path(output_path2).name}")
            logger.info(f"Size: {results2['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"Time: {results2['processing_time']:.1f}s")
        else:
            logger.error("Demo 2 failed")
            
    except Exception as e:
        logger.error(f"Demo 2 error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed! Check data/output/ for results")
    
    # Show summary
    output_files = list(output_dir.glob("demo*.gif"))
    if output_files:
        logger.info(f"Generated {len(output_files)} demo GIFs:")
        for file in output_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   - {file.name} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    demo_system() 