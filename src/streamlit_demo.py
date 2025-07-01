
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
