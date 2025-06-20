#!/usr/bin/env python3
"""
Custom Attention Configuration Script
Allows you to fine-tune attention detection parameters.
"""

import sys
from pathlib import Path
import logging

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

def create_custom_gif(video_path, config_name, custom_config):
    """Create a GIF with custom attention configuration."""
    logger.info(f"🎨 Creating custom GIF: {config_name}")
    
    try:
        pipeline = InMemoryPipeline()
        
        # Create output directory
        output_dir = Path("data/output")
        output_dir.mkdir(exist_ok=True)
        
        # Create output path
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"custom_{config_name}_{video_name}.gif")
        
        # Process video with custom config
        results = pipeline.process_video(
            video_path=video_path,
            output_path=output_path,
            override_config=custom_config
        )
        
        if results.get('success'):
            logger.info(f"✅ Custom GIF created: {Path(output_path).name}")
            logger.info(f"📏 Size: {results['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results['processing_time']:.1f}s")
            return True
        else:
            logger.error("❌ Custom GIF creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating custom GIF: {e}")
        return False

def main():
    """Main function with predefined custom configurations."""
    logger.info("🎯 Custom Attention Configuration Tool")
    logger.info("=" * 60)
    
    # Find a sample video
    uploads_dir = Path("data/uploads")
    sample_videos = list(uploads_dir.glob("*.mp4"))
    
    if not sample_videos:
        logger.error("❌ No sample videos found")
        return
    
    # Use first video
    video_path = str(sample_videos[0])
    logger.info(f"📹 Using video: {Path(video_path).name}")
    
    # Configuration 1: Maximum Attention (for weak attention)
    logger.info("\n🔥 Configuration 1: Maximum Attention")
    config1 = {
        "gif": {
            "fps": 12,
            "max_frames": 30,
            "overlay_style": "highlight",  # Bright highlighting
            "overlay_intensity": 1.0  # Maximum intensity
        },
        "model": {
            "name": "videomae-base",
            "device": "auto",
            "precision": "fp32"  # Full precision
        },
        "processing": {
            "adaptive_stride": False,  # Process every frame
            "min_stride": 1,
            "max_stride": 1
        },
        "limits": {
            "max_resolution": 1080,  # High resolution
            "max_duration": 20
        }
    }
    
    create_custom_gif(video_path, "max_attention", config1)
    
    # Configuration 2: Motion-Focused Attention
    logger.info("\n🏃 Configuration 2: Motion-Focused Attention")
    config2 = {
        "gif": {
            "fps": 15,  # High FPS for motion
            "max_frames": 40,
            "overlay_style": "pulse",  # Pulse effect for motion
            "overlay_intensity": 0.9
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": False,  # Every frame for motion tracking
            "min_stride": 1,
            "max_stride": 1
        },
        "limits": {
            "max_resolution": 720,
            "max_duration": 25
        }
    }
    
    create_custom_gif(video_path, "motion_focused", config2)
    
    # Configuration 3: Subtle Attention (for strong attention)
    logger.info("\n🌟 Configuration 3: Subtle Attention")
    config3 = {
        "gif": {
            "fps": 8,
            "max_frames": 20,
            "overlay_style": "glow",  # Subtle glow effect
            "overlay_intensity": 0.5  # Low intensity
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 2,
            "max_stride": 4
        },
        "limits": {
            "max_resolution": 720,
            "max_duration": 15
        }
    }
    
    create_custom_gif(video_path, "subtle_attention", config3)
    
    # Configuration 4: Balanced Attention
    logger.info("\n⚖️ Configuration 4: Balanced Attention")
    config4 = {
        "gif": {
            "fps": 10,
            "max_frames": 25,
            "overlay_style": "heatmap",  # Balanced heatmap
            "overlay_intensity": 0.75  # Medium intensity
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
            "max_resolution": 720,
            "max_duration": 20
        }
    }
    
    create_custom_gif(video_path, "balanced", config4)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Custom configurations completed!")
    
    # Show summary
    output_files = list(Path("data/output").glob("custom*.gif"))
    if output_files:
        logger.info(f"📁 Generated {len(output_files)} custom GIFs:")
        for file in output_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   - {file.name} ({size_mb:.2f} MB)")
    
    logger.info("\n💡 How to use these configurations:")
    logger.info("1. Compare the generated GIFs to see which attention style works best")
    logger.info("2. Use the configuration that gives the best results for your videos")
    logger.info("3. Adjust parameters based on your specific needs")
    logger.info("4. Copy the configuration to your own scripts")

if __name__ == "__main__":
    main() 