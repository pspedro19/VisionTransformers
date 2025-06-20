#!/usr/bin/env python3
"""
Script to run the FastAPI web server for ViT-GIF Highlight.
"""

import os
import sys
from pathlib import Path
import uvicorn
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.api.app import app
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("💡 Install with: poetry install --extras api")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI server."""
    logger.info("🚀 Starting ViT-GIF Highlight API Server...")
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create necessary directories
    for dir_path in ["data/uploads", "data/output"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Server configuration
    config = {
        "app": "src.api.app:app",  # Import string for reload mode
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Enable auto-reload
        "workers": 1,    # Number of worker processes
        "log_level": "info",
        "access_log": True,
        "limit_concurrency": 10,  # Limit concurrent connections
        "timeout_keep_alive": 5,  # Seconds to keep idle connections
    }
    
    logger.info(f"📍 API will be available at: http://localhost:{config['port']}")
    logger.info(f"📚 Documentation at: http://localhost:{config['port']}/docs")
    logger.info("🛑 Press Ctrl+C to stop")
    
    # Run server
    uvicorn.run(**config)

if __name__ == "__main__":
    main() 