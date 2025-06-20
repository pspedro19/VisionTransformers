#!/usr/bin/env python3
"""
Advanced Attention Improvement Script
Uses multiple techniques to enhance attention detection.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.pipeline import InMemoryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_advanced_attention():
    """Test advanced attention improvement techniques."""
    logger.info("🧠 Advanced Attention Improvement Techniques")
    logger.info("=" * 60)
    
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
        logger.error("❌ No sample videos found")
        return
    
    # Use a video with clear motion/action
    video_path = str(sample_videos[1])  # Use second video
    logger.info(f"📹 Using video: {Path(video_path).name}")
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    
    # Technique 1: Motion-based attention with high frame rate
    logger.info("\n🏃 Technique 1: Motion-Based Attention (High FPS)")
    config1 = {
        "gif": {
            "fps": 15,  # Higher FPS for better motion capture
            "max_frames": 40,  # More frames
            "overlay_style": "pulse",  # Pulse effect for motion
            "overlay_intensity": 0.85
        },
        "model": {
            "name": "videomae-base",
            "device": "auto",
            "precision": "fp32"
        },
        "processing": {
            "adaptive_stride": False,  # Process every frame
            "min_stride": 1,
            "max_stride": 1
        },
        "limits": {
            "max_resolution": 720,
            "max_duration": 30  # Limit duration for quality
        }
    }
    
    output_path1 = str(output_dir / f"advanced1_{Path(video_path).stem}_motion.gif")
    
    try:
        results1 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path1,
            override_config=config1
        )
        
        if results1.get('success'):
            logger.info(f"✅ Technique 1 completed: {Path(output_path1).name}")
            logger.info(f"📏 Size: {results1['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results1['processing_time']:.1f}s")
        else:
            logger.error("❌ Technique 1 failed")
            
    except Exception as e:
        logger.error(f"❌ Technique 1 error: {e}")
    
    # Technique 2: Multi-scale attention processing
    logger.info("\n🔍 Technique 2: Multi-Scale Attention Processing")
    config2 = {
        "gif": {
            "fps": 10,
            "max_frames": 25,
            "overlay_style": "heatmap",
            "overlay_intensity": 0.9
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 1,
            "max_stride": 2
        },
        "limits": {
            "max_resolution": 1080,  # Higher resolution
            "max_duration": 20
        }
    }
    
    output_path2 = str(output_dir / f"advanced2_{Path(video_path).stem}_multiscale.gif")
    
    try:
        results2 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path2,
            override_config=config2
        )
        
        if results2.get('success'):
            logger.info(f"✅ Technique 2 completed: {Path(output_path2).name}")
            logger.info(f"📏 Size: {results2['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results2['processing_time']:.1f}s")
        else:
            logger.error("❌ Technique 2 failed")
            
    except Exception as e:
        logger.error(f"❌ Technique 2 error: {e}")
    
    # Technique 3: Temporal attention with glow effect
    logger.info("\n⏰ Technique 3: Temporal Attention with Glow")
    config3 = {
        "gif": {
            "fps": 12,
            "max_frames": 35,
            "overlay_style": "glow",
            "overlay_intensity": 0.75
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 1,
            "max_stride": 1  # Every frame for temporal consistency
        },
        "limits": {
            "max_resolution": 720,
            "max_duration": 25
        }
    }
    
    output_path3 = str(output_dir / f"advanced3_{Path(video_path).stem}_temporal.gif")
    
    try:
        results3 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path3,
            override_config=config3
        )
        
        if results3.get('success'):
            logger.info(f"✅ Technique 3 completed: {Path(output_path3).name}")
            logger.info(f"📏 Size: {results3['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results3['processing_time']:.1f}s")
        else:
            logger.error("❌ Technique 3 failed")
            
    except Exception as e:
        logger.error(f"❌ Technique 3 error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Advanced attention techniques completed!")
    
    # Show summary
    output_files = list(output_dir.glob("advanced*.gif"))
    if output_files:
        logger.info(f"📁 Generated {len(output_files)} advanced GIFs:")
        for file in output_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   - {file.name} ({size_mb:.2f} MB)")

def main():
    """Run advanced attention improvement tests."""
    logger.info("🚀 Starting Advanced Attention Improvement")
    
    # Test advanced techniques
    test_advanced_attention()
    
    logger.info("\n🧠 Advanced Attention Techniques Applied:")
    logger.info("1. Motion-based attention with high FPS (15 FPS)")
    logger.info("2. Multi-scale processing with higher resolution")
    logger.info("3. Temporal attention with frame-by-frame processing")
    logger.info("4. Different overlay styles: pulse, heatmap, glow")
    logger.info("5. Optimized intensity levels for each technique")
    logger.info("6. Full precision processing (fp32)")
    logger.info("7. Check the generated GIFs to compare attention quality")

if __name__ == "__main__":
    main() 