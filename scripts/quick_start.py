#!/usr/bin/env python3
"""
Quick Start Script for Vision Transformers GIF Generator
"""

import os
import sys
from pathlib import Path
import subprocess

def create_streamlit_demo():
    """Create a simple Streamlit demo."""
    demo_content = '''
import streamlit as st
import tempfile
import time

st.set_page_config(
    page_title="Vision Transformers GIF Generator",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1>Vision Transformers GIF Generator</h1>
    <p>Convert videos to GIFs with visual attention highlighting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Features")
    st.markdown("""
    - Upload video interactively
    - Select video segment
    - Multiple overlay styles
    - Various AI models
    - Advanced configuration
    """)

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Time", "Config", "Generate"])

with tab1:
    st.header("Upload Video")
    
    uploaded_file = st.file_uploader(
        "Select your video",
        type=['mp4', 'avi', 'mov', 'webm'],
        help="Drag and drop your file here"
    )
    
    if uploaded_file:
        st.success("Video uploaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Size", f"{len(uploaded_file.getvalue()) / (1024*1024):.1f} MB")
        with col2:
            st.metric("Format", uploaded_file.type)
        with col3:
            st.metric("Duration", "~30s")
        
        st.video(uploaded_file)

with tab2:
    st.header("Select Segment")
    
    if 'uploaded_file' not in locals() or not uploaded_file:
        st.warning("First upload a video in the Upload tab")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider("Start (seconds)", 0.0, 25.0, 0.0, 0.1)
        with col2:
            duration = st.slider("Duration (seconds)", 1.0, 30.0, 5.0, 0.1)
        
        st.markdown("**Quick duration:**")
        cols = st.columns(5)
        for i, dur in enumerate([3, 5, 10, 15, 30]):
            with cols[i]:
                if st.button(f"{dur}s"):
                    duration = dur
        
        st.info(f"Segment: {start_time:.1f}s - {start_time + duration:.1f}s ({duration:.1f}s)")

with tab3:
    st.header("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visual")
        fps = st.slider("FPS", 1, 15, 5)
        overlay_style = st.selectbox("Style", ["heatmap", "highlight", "glow", "pulse", "transparent"])
        overlay_intensity = st.slider("Intensity", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        st.subheader("AI Model")
        model = st.selectbox("Model", ["Auto", "VideoMAE Base", "VideoMAE Large", "TimeSformer"])
        optimization = st.slider("Optimization", 0, 3, 2)
    
    st.info("Configuration saved")

with tab4:
    st.header("Generate GIF")
    
    if st.button("Generate Smart GIF", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        
        steps = [
            (25, "Extracting segment..."),
            (50, "Analyzing with AI..."),
            (75, "Applying overlay..."),
            (100, "Completed!")
        ]
        
        for prog, msg in steps:
            status.text(msg)
            progress_bar.progress(prog)
            time.sleep(1)
        
        st.success("GIF generated successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", "8.5s")
        with col2:
            st.metric("Frames", "42")
        with col3:
            st.metric("Size", "2.1 MB")
        
        st.download_button("Download GIF", "demo_gif_data", "result.gif")

st.markdown("---")
st.markdown("*Demo v2.0 - In production would connect to real Vision Transformers pipeline*")
'''
    
    with open("src/streamlit_demo.py", "w", encoding="utf-8") as f:
        f.write(demo_content)

def create_simple_html():
    """Create simple HTML interface."""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Transformers GIF Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; }
        .container { max-width: 800px; margin: 0 auto; padding: 2rem; }
        .header { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-bottom: 2rem; }
        .card { background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .btn { background: #3b82f6; color: white; padding: 0.75rem 1.5rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; }
        .btn:hover { background: #2563eb; }
        .upload-area { border: 2px dashed #e5e7eb; padding: 3rem; text-align: center; border-radius: 10px; margin: 1rem 0; }
        .upload-area:hover { border-color: #3b82f6; background: rgba(59, 130, 246, 0.05); }
        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
        .demo-note { background: #fef3c7; border: 1px solid #f59e0b; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Vision Transformers GIF Generator</h1>
            <p>Convert videos to GIFs with visual attention highlighting</p>
        </header>
        
        <div class="card">
            <h2>Upload Video</h2>
            <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                <p>Click here to select your video</p>
                <small>Formats: MP4, AVI, MOV, WEBM (max 100MB)</small>
                <input type="file" id="videoInput" accept="video/*" style="display: none;">
            </div>
            <video id="videoPlayer" controls style="width: 100%; display: none; margin-top: 1rem; border-radius: 8px;"></video>
        </div>
        
        <div class="card">
            <h2>Segment Configuration</h2>
            <div class="controls">
                <div>
                    <label>Start (seconds):</label>
                    <input type="range" id="startTime" min="0" max="30" value="0" step="0.1">
                    <span id="startValue">0.0s</span>
                </div>
                <div>
                    <label>Duration (seconds):</label>
                    <input type="range" id="duration" min="1" max="30" value="5" step="0.1">
                    <span id="durationValue">5.0s</span>
                </div>
            </div>
            <div style="margin: 1rem 0;">
                <button class="btn" onclick="generateGIF()">Generate GIF</button>
            </div>
        </div>
        
        <div class="demo-note">
            <strong>Demo Mode:</strong> This is a demonstration interface. In production, it would connect to the actual Vision Transformers pipeline.
        </div>
    </div>

    <script>
        document.getElementById('videoInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const video = document.getElementById('videoPlayer');
                video.src = URL.createObjectURL(file);
                video.style.display = 'block';
            }
        });

        document.getElementById('startTime').addEventListener('input', function(e) {
            document.getElementById('startValue').textContent = e.target.value + 's';
        });

        document.getElementById('duration').addEventListener('input', function(e) {
            document.getElementById('durationValue').textContent = e.target.value + 's';
        });

        function generateGIF() {
            alert('Demo: GIF generation would start here with the actual Vision Transformers pipeline.');
        }
    </script>
</body>
</html>
'''
    
    with open("static/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    """Main function to set up the project."""
    print("Setting up Vision Transformers GIF Generator...")
    
    # Create necessary directories
    directories = ["data/uploads", "data/output", "static", "config"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create demo files
    create_streamlit_demo()
    create_simple_html()
    
    print("\nSetup completed!")
    print("\nTo start the demo:")
    print("1. python demo_quick.py")
    print("2. streamlit run src/streamlit_demo.py")
    print("3. Open static/index.html in your browser")

if __name__ == "__main__":
    main() 