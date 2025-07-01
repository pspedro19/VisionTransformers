#!/usr/bin/env python3
"""
Test script to generate a GIF with attention highlighting.
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

def main():
    """Test GIF generation with attention highlighting."""
    logger.info("🎬 Testing ViT-GIF Highlight Generation")
    
    # Initialize pipeline
    try:
        pipeline = InMemoryPipeline()
        logger.info("✅ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Find a sample video
    uploads_dir = Path("data/uploads")
    sample_videos = list(uploads_dir.glob("*.mp4"))
    
    if not sample_videos:
        logger.error("❌ No sample videos found in data/uploads/")
        return
    
    # Use the first sample video
    video_path = str(sample_videos[0])
    logger.info(f"📹 Using sample video: {video_path}")
    
    # Create output path
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = str(output_dir / f"test_{video_name}_highlighted.gif")
    
    # Configuration for highlighting important parts
    config = {
        "gif": {
            "fps": 8,
            "max_frames": 30,
            "optimization_level": 2,
            "overlay_style": "heatmap",  # This will highlight important areas
            "overlay_intensity": 0.8     # High intensity for clear highlighting
        },
        "model": {
            "name": "videomae-base",
            "device": "auto",
            "precision": "fp16"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 2,
            "max_stride": 8
        }
    }
    
    logger.info("🚀 Starting GIF generation with attention highlighting...")
    logger.info(f"📊 Configuration: {config}")
    
    try:
        # Process the video
        results = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            override_config=config
        )
        
        logger.info("✅ GIF generation completed successfully!")
        logger.info(f"📊 Results: {results}")
        
        # Check if output file exists
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"📁 Output GIF saved: {output_path}")
            logger.info(f"📏 File size: {file_size:.2f} MB")
            
            # Show some key metrics
            if "processing_stats" in results:
                stats = results["processing_stats"]
                logger.info(f"⏱️  Processing time: {stats.get('total_time', 'N/A')}s")
                logger.info(f"🎞️  Frames processed: {stats.get('frames_processed', 'N/A')}")
                logger.info(f"🎯 Attention maps generated: {stats.get('attention_maps', 'N/A')}")
            
            if "attention_stats" in results:
                att_stats = results["attention_stats"]
                logger.info(f"🎯 Average attention score: {att_stats.get('avg_attention', 'N/A'):.3f}")
                logger.info(f"🎯 Max attention score: {att_stats.get('max_attention', 'N/A'):.3f}")
                
        else:
            logger.error("❌ Output file was not created")
            
    except Exception as e:
        logger.error(f"❌ GIF generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 