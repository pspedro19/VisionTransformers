#!/usr/bin/env python3
"""
Script to improve attention detection in ViT-GIF Highlight system.
Implements multiple strategies for better attention capture.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pipeline import InMemoryPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_attention_improvements():
    """Test different attention improvement strategies."""
    logger.info("🎯 Testing Attention Improvement Strategies")
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
    
    # Use a video that might have clear attention patterns
    video_path = str(sample_videos[2])  # Use third video
    logger.info(f"📹 Using video: {Path(video_path).name}")
    
    # Create output directory
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True)
    
    # Strategy 1: Higher resolution processing
    logger.info("\n🔍 Strategy 1: Higher Resolution Processing")
    config1 = {
        "gif": {
            "fps": 8,
            "max_frames": 25,
            "overlay_style": "heatmap",
            "overlay_intensity": 0.9  # Higher intensity
        },
        "model": {
            "name": "videomae-base",
            "device": "auto",
            "precision": "fp32"  # Full precision for better accuracy
        },
        "processing": {
            "adaptive_stride": False,  # Process all frames
            "min_stride": 1,
            "max_stride": 1
        },
        "limits": {
            "max_resolution": 1080  # Higher resolution
        }
    }
    
    output_path1 = str(output_dir / f"improved1_{Path(video_path).stem}_highres.gif")
    
    try:
        results1 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path1,
            override_config=config1
        )
        
        if results1.get('success'):
            logger.info(f"✅ Strategy 1 completed: {Path(output_path1).name}")
            logger.info(f"📏 Size: {results1['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results1['processing_time']:.1f}s")
        else:
            logger.error("❌ Strategy 1 failed")
            
    except Exception as e:
        logger.error(f"❌ Strategy 1 error: {e}")
    
    # Strategy 2: Multiple attention styles
    logger.info("\n🎨 Strategy 2: Multiple Attention Styles")
    config2 = {
        "gif": {
            "fps": 10,
            "max_frames": 20,
            "overlay_style": "highlight",  # Different style
            "overlay_intensity": 0.8
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 1,  # More frequent sampling
            "max_stride": 3
        }
    }
    
    output_path2 = str(output_dir / f"improved2_{Path(video_path).stem}_highlight.gif")
    
    try:
        results2 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path2,
            override_config=config2
        )
        
        if results2.get('success'):
            logger.info(f"✅ Strategy 2 completed: {Path(output_path2).name}")
            logger.info(f"📏 Size: {results2['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results2['processing_time']:.1f}s")
        else:
            logger.error("❌ Strategy 2 failed")
            
    except Exception as e:
        logger.error(f"❌ Strategy 2 error: {e}")
    
    # Strategy 3: Glow overlay for subtle attention
    logger.info("\n✨ Strategy 3: Glow Overlay for Subtle Attention")
    config3 = {
        "gif": {
            "fps": 12,
            "max_frames": 30,
            "overlay_style": "glow",  # Glow effect
            "overlay_intensity": 0.7
        },
        "model": {
            "name": "videomae-base",
            "device": "auto"
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 1,
            "max_stride": 2  # Very frequent sampling
        }
    }
    
    output_path3 = str(output_dir / f"improved3_{Path(video_path).stem}_glow.gif")
    
    try:
        results3 = pipeline.process_video(
            video_path=video_path,
            output_path=output_path3,
            override_config=config3
        )
        
        if results3.get('success'):
            logger.info(f"✅ Strategy 3 completed: {Path(output_path3).name}")
            logger.info(f"📏 Size: {results3['gif_stats']['file_size_mb']:.2f} MB")
            logger.info(f"⏱️ Time: {results3['processing_time']:.1f}s")
        else:
            logger.error("❌ Strategy 3 failed")
            
    except Exception as e:
        logger.error(f"❌ Strategy 3 error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Attention improvement tests completed!")
    
    # Show summary
    output_files = list(output_dir.glob("improved*.gif"))
    if output_files:
        logger.info(f"📁 Generated {len(output_files)} improved GIFs:")
        for file in output_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   - {file.name} ({size_mb:.2f} MB)")

def create_attention_analysis_script():
    """Create a script to analyze attention patterns."""
    analysis_script = '''
#!/usr/bin/env python3
"""
Attention Analysis Script
Analyzes attention patterns in generated GIFs.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_attention_patterns(gif_path):
    """Analyze attention patterns in a GIF."""
    cap = cv2.VideoCapture(gif_path)
    
    attention_scores = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate attention metrics
        # 1. Variance (higher = more attention variation)
        variance = np.var(gray)
        
        # 2. Edge density (higher = more edges/attention)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Brightness variation
        brightness_var = np.std(gray)
        
        attention_score = (variance + edge_density * 1000 + brightness_var) / 3
        attention_scores.append(attention_score)
        
        frame_count += 1
    
    cap.release()
    
    return attention_scores, frame_count

def plot_attention_analysis(gif_paths):
    """Plot attention analysis for multiple GIFs."""
    plt.figure(figsize=(15, 10))
    
    for i, gif_path in enumerate(gif_paths):
        scores, frames = analyze_attention_patterns(gif_path)
        
        plt.subplot(2, 2, i+1)
        plt.plot(scores)
        plt.title(f'Attention Pattern: {Path(gif_path).name}')
        plt.xlabel('Frame')
        plt.ylabel('Attention Score')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Analyze generated GIFs
    output_dir = Path("data/output")
    gif_files = list(output_dir.glob("improved*.gif"))
    
    if gif_files:
        plot_attention_analysis(gif_files[:4])  # Analyze first 4
    else:
        print("No improved GIFs found for analysis")
'''
    
    with open("attention_analysis.py", "w", encoding="utf-8") as f:
        f.write(analysis_script)
    
    logger.info("📊 Created attention_analysis.py for detailed analysis")

def main():
    """Run attention improvement tests."""
    logger.info("🚀 Starting Attention Improvement Analysis")
    
    # Test different strategies
    test_attention_improvements()
    
    # Create analysis script
    create_attention_analysis_script()
    
    logger.info("\n💡 Recommendations for Better Attention:")
    logger.info("1. Use higher resolution (1080p) for better detail")
    logger.info("2. Try different overlay styles: heatmap, highlight, glow, pulse, transparent")
    logger.info("3. Increase overlay intensity (0.8-0.9)")
    logger.info("4. Use full precision (fp32) instead of fp16")
    logger.info("5. Process more frames with lower stride")
    logger.info("6. Try different models: VideoMAE, TimeSformer, Video-Swin")
    logger.info("7. Check the generated GIFs to compare attention quality")

if __name__ == "__main__":
    main() 