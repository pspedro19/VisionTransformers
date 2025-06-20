
import streamlit as st
import tempfile
import time

st.set_page_config(
    page_title="ViT-GIF Highlight Demo",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<div style="text-align: center; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1>🎬 ViT-GIF Highlight Demo</h1>
    <p>Generador Inteligente de GIFs con Atención Visual</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Funcionalidades")
    st.markdown("""
    - ✅ Upload de video interactivo
    - ✅ Selección de segmento
    - ✅ 5 estilos de overlay
    - ✅ Múltiples modelos IA
    - ✅ Configuración avanzada
    """)

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "⏱️ Tiempo", "⚙️ Config", "🚀 Generar"])

with tab1:
    st.header("📁 Cargar Video")
    
    uploaded_file = st.file_uploader(
        "Selecciona tu video",
        type=['mp4', 'avi', 'mov', 'webm'],
        help="Arrastra y suelta tu archivo aquí"
    )
    
    if uploaded_file:
        st.success("✅ Video cargado exitosamente!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tamaño", f"{len(uploaded_file.getvalue()) / (1024*1024):.1f} MB")
        with col2:
            st.metric("Formato", uploaded_file.type)
        with col3:
            st.metric("Duración", "~30s")
        
        st.video(uploaded_file)

with tab2:
    st.header("⏱️ Seleccionar Segmento")
    
    if 'uploaded_file' not in locals() or not uploaded_file:
        st.warning("⚠️ Primero carga un video en la pestaña Upload")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider("Inicio (segundos)", 0.0, 25.0, 0.0, 0.1)
        with col2:
            duration = st.slider("Duración (segundos)", 1.0, 30.0, 5.0, 0.1)
        
        st.markdown("**Duración rápida:**")
        cols = st.columns(5)
        for i, dur in enumerate([3, 5, 10, 15, 30]):
            with cols[i]:
                if st.button(f"{dur}s"):
                    duration = dur
        
        st.info(f"📊 Segmento: {start_time:.1f}s - {start_time + duration:.1f}s ({duration:.1f}s)")

with tab3:
    st.header("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Visual")
        fps = st.slider("FPS", 1, 15, 5)
        overlay_style = st.selectbox("Estilo", ["heatmap", "highlight", "glow", "pulse"])
        overlay_intensity = st.slider("Intensidad", 0.0, 1.0, 0.7, 0.1)
    
    with col2:
        st.subheader("🧠 Modelo IA")
        model = st.selectbox("Modelo", ["Automático", "VideoMAE Base", "VideoMAE Large", "TimeSformer"])
        optimization = st.slider("Optimización", 0, 3, 2)
    
    st.info("📊 Configuración guardada")

with tab4:
    st.header("🚀 Generar GIF")
    
    if st.button("🚀 Generar GIF Inteligente", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        
        steps = [
            (25, "📹 Extrayendo segmento..."),
            (50, "🧠 Analizando con IA..."),
            (75, "🎨 Aplicando overlay..."),
            (100, "✅ ¡Completado!")
        ]
        
        for prog, msg in steps:
            status.text(msg)
            progress_bar.progress(prog)
            time.sleep(1)
        
        st.success("🎉 ¡GIF generado exitosamente!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tiempo", "8.5s")
        with col2:
            st.metric("Frames", "42")
        with col3:
            st.metric("Tamaño", "2.1 MB")
        
        st.download_button("💾 Descargar GIF", "demo_gif_data", "resultado.gif")

st.markdown("---")
st.markdown("*Demo v2.0 - En producción se conectaría al pipeline real de ViT-GIF Highlight*")
