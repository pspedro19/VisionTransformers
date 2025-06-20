#!/usr/bin/env python3
"""
Script to run the Streamlit demo for ViT-GIF Highlight.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run the Streamlit demo."""
    print("🎬 Starting ViT-GIF Highlight Streamlit Demo...")
    print("📍 Demo will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found!")
        print("💡 Install with: poetry install --extras ui")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Path to the streamlit demo
    demo_path = project_root / "src" / "streamlit_demo.py"
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_path}")
        sys.exit(1)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit demo stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 