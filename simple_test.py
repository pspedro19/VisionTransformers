#!/usr/bin/env python3
"""
Simple test script to verify ViT-GIF Highlight system works.
"""

import sys
from pathlib import Path
import logging

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that all basic imports work."""
    logger.info("🔍 Testing basic imports...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠️ CUDA not available, using CPU")
    except ImportError as e:
        logger.error(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import decord
        logger.info(f"✅ Decord version: {decord.__version__}")
    except ImportError as e:
        logger.error(f"❌ Decord import failed: {e}")
        return False
    
    try:
        import cv2
        logger.info(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow available")
    except ImportError as e:
        logger.error(f"❌ PIL import failed: {e}")
        return False
    
    return True

def test_video_access():
    """Test that we can access the sample videos."""
    logger.info("📹 Testing video access...")
    
    uploads_dir = Path("data/uploads")
    if not uploads_dir.exists():
        logger.error(f"❌ Uploads directory not found: {uploads_dir}")
        return None
    
    sample_videos = list(uploads_dir.glob("*.mp4"))
    if not sample_videos:
        logger.error("❌ No MP4 videos found in uploads directory")
        return None
    
    # Choose the smallest video for testing
    video_path = min(sample_videos, key=lambda p: p.stat().st_size)
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"✅ Found test video: {video_path.name}")
    logger.info(f"📏 File size: {file_size_mb:.2f} MB")
    
    return str(video_path)

def test_video_decoding(video_path):
    """Test basic video decoding."""
    logger.info("🎬 Testing video decoding...")
    
    try:
        from decord import VideoReader
        import numpy as np
        
        # Open video
        vr = VideoReader(video_path)
        logger.info(f"✅ Video opened successfully")
        logger.info(f"📊 Video info: {len(vr)} frames, {vr.get_avg_fps():.2f} FPS")
        logger.info(f"📐 Resolution: {vr[0].shape}")
        
        # Read first few frames
        frame_indices = list(range(min(5, len(vr))))
        frames = vr.get_batch(frame_indices)
        
        logger.info(f"✅ Successfully read {len(frames)} frames")
        logger.info(f"📊 Frame shape: {frames.shape}")
        logger.info(f"📊 Frame dtype: {frames.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Video decoding failed: {e}")
        return False

def test_simple_gif_creation(video_path):
    """Test creating a simple GIF without attention."""
    logger.info("🎨 Testing simple GIF creation...")
    
    try:
        from decord import VideoReader
        import imageio
        import numpy as np
        from pathlib import Path
        
        # Create output directory
        output_dir = Path("data/output")
        output_dir.mkdir(exist_ok=True)
        
        # Read video
        vr = VideoReader(video_path)
        
        # Take first 10 frames or less
        max_frames = min(10, len(vr))
        frame_indices = list(range(max_frames))
        frames = vr.get_batch(frame_indices)
        
        # Convert to numpy and ensure correct format
        frames_np = frames.asnumpy()
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        
        # Create output path
        video_name = Path(video_path).stem
        output_path = str(output_dir / f"test_{video_name}_simple.gif")
        
        # Save as GIF
        imageio.mimsave(output_path, frames_np, fps=5)
        
        # Check if file was created
        if Path(output_path).exists():
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"✅ Simple GIF created: {output_path}")
            logger.info(f"📏 File size: {file_size_mb:.2f} MB")
            return True
        else:
            logger.error("❌ GIF file was not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple GIF creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting ViT-GIF Highlight System Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        logger.error("❌ Basic imports failed, stopping tests")
        return
    
    # Test 2: Video access
    video_path = test_video_access()
    if not video_path:
        logger.error("❌ Video access failed, stopping tests")
        return
    
    # Test 3: Video decoding
    if not test_video_decoding(video_path):
        logger.error("❌ Video decoding failed, stopping tests")
        return
    
    # Test 4: Simple GIF creation
    if not test_simple_gif_creation(video_path):
        logger.error("❌ Simple GIF creation failed")
        return
    
    logger.info("=" * 50)
    logger.info("🎉 All basic tests passed! System is working correctly.")
    logger.info("📁 Check data/output/ for generated test GIFs")

if __name__ == "__main__":
    main() 