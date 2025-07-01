#!/usr/bin/env python3
"""
Script simple para probar el pipeline que ya funciona.
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

def test_working_pipeline():
    """Test the pipeline that already works."""
    logger.info("Testing working pipeline")
    
    # Initialize pipeline
    pipeline = InMemoryPipeline("config/mvp1.yaml")
    
    # Find a video
    uploads_dir = Path("data/uploads")
    sample_videos = list(uploads_dir.glob("*.mp4"))
    
    if not sample_videos:
        logger.error("No videos found")
        return False
    
    video_path = str(sample_videos[0])
    output_path = "data/output/test_working_pipeline.gif"
    
    logger.info(f"Processing: {Path(video_path).name}")
    
    # Use the pipeline that already works
    result = pipeline.process_video(
        video_path=video_path,
        output_path=output_path
    )
    
    if result.get('success'):
        logger.info("Pipeline worked successfully!")
        logger.info(f"Output: {output_path}")
        return True
    else:
        logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = test_working_pipeline()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")
        sys.exit(1) 