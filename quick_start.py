#!/usr/bin/env python3
"""
Quick Start Script for ViT-GIF Highlight Web Interface
Creates all necessary files and starts the web interface.
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
    - ✅ 4 estilos de overlay
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
'''
    
    with open("src/streamlit_demo.py", "w", encoding="utf-8") as f:
        f.write(demo_content)

def create_simple_html():
    """Create simple HTML interface."""
    html_content = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViT-GIF Highlight</title>
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
            <h1>🎬 ViT-GIF Highlight</h1>
            <p>Generador Inteligente de GIFs con Atención Visual</p>
        </header>
        
        <div class="card">
            <h2>📁 Cargar Video</h2>
            <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                <p>🎬 Haz clic aquí para seleccionar tu video</p>
                <small>Formatos: MP4, AVI, MOV, WEBM (máx. 100MB)</small>
                <input type="file" id="videoInput" accept="video/*" style="display: none;">
            </div>
            <video id="videoPlayer" controls style="width: 100%; display: none; margin-top: 1rem; border-radius: 8px;"></video>
        </div>
        
        <div class="card">
            <h2>⏱️ Configuración del Segmento</h2>
            <div class="controls">
                <div>
                    <label>Inicio (segundos):</label>
                    <input type="range" id="startTime" min="0" max="30" value="0" step="0.1">
                    <span id="startValue">0.0s</span>
                </div>
                <div>
                    <label>Duración (segundos):</label>
                    <input type="range" id="duration" min="1" max="30" value="5" step="0.1">
                    <span id="durationValue">5.0s</span>
                </div>
            </div>
            <div style="margin: 1rem 0;">
                <strong>Duración rápida:</strong>
                <button class="btn" onclick="setDuration(3)" style="margin: 0.25rem;">3s</button>
                <button class="btn" onclick="setDuration(5)" style="margin: 0.25rem;">5s</button>
                <button class="btn" onclick="setDuration(10)" style="margin: 0.25rem;">10s</button>
            </div>
        </div>
        
        <div class="card">
            <h2>⚙️ Configuración del GIF</h2>
            <div class="controls">
                <div>
                    <label>FPS:</label>
                    <input type="range" id="fps" min="1" max="15" value="5">
                    <span id="fpsValue">5</span>
                </div>
                <div>
                    <label>Estilo de Overlay:</label>
                    <select id="overlayStyle">
                        <option value="heatmap">🔥 Heatmap</option>
                        <option value="highlight">✨ Highlight</option>
                        <option value="glow">🌟 Glow</option>
                        <option value="pulse">💫 Pulse</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Generar GIF</h2>
            <button class="btn" onclick="generateGIF()" style="width: 100%; padding: 1rem; font-size: 1.1rem;">
                🚀 Generar GIF Inteligente
            </button>
            <div id="progressContainer" style="display: none; margin-top: 1rem;">
                <div id="progressBar" style="background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                    <div id="progressFill" style="background: #3b82f6; height: 8px; width: 0%; transition: width 0.3s;"></div>
                </div>
                <p id="progressText" style="margin-top: 0.5rem;">Procesando...</p>
            </div>
        </div>
        
        <div class="demo-note">
            <strong>📝 Nota:</strong> Esta es una demo de la interfaz. En la implementación completa se conectaría al pipeline de ViT-GIF Highlight para generar GIFs reales con overlays de atención visual.
        </div>
    </div>
    
    <script>
        // File upload handling
        document.getElementById('videoInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const video = document.getElementById('videoPlayer');
                video.src = URL.createObjectURL(file);
                video.style.display = 'block';
                alert('✅ Video cargado: ' + file.name);
            }
        });
        
        // Range slider updates
        document.getElementById('startTime').addEventListener('input', function() {
            document.getElementById('startValue').textContent = this.value + 's';
        });
        
        document.getElementById('duration').addEventListener('input', function() {
            document.getElementById('durationValue').textContent = this.value + 's';
        });
        
        document.getElementById('fps').addEventListener('input', function() {
            document.getElementById('fpsValue').textContent = this.value;
        });
        
        function setDuration(seconds) {
            document.getElementById('duration').value = seconds;
            document.getElementById('durationValue').textContent = seconds + 's';
        }
        
        function generateGIF() {
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressContainer.style.display = 'block';
            
            const steps = [
                {progress: 25, text: '📹 Extrayendo segmento...'},
                {progress: 50, text: '🧠 Analizando con IA...'},
                {progress: 75, text: '🎨 Aplicando overlay...'},
                {progress: 100, text: '✅ ¡Completado!'}
            ];
            
            let currentStep = 0;
            const interval = setInterval(() => {
                if (currentStep < steps.length) {
                    const step = steps[currentStep];
                    progressFill.style.width = step.progress + '%';
                    progressText.textContent = step.text;
                    currentStep++;
                } else {
                    clearInterval(interval);
                    setTimeout(() => {
                        alert('🎉 ¡GIF generado exitosamente!\\n\\nEn la implementación real, aquí se descargaría el archivo.');
                        progressContainer.style.display = 'none';
                    }, 500);
                }
            }, 1500);
        }
    </script>
</body>
</html>
'''
    
    os.makedirs("static", exist_ok=True)
    with open("static/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

def main():
    """Main setup and execution."""
    print("🎬 ViT-GIF Highlight - Quick Start Setup")
    print("=" * 50)
    
    # Create necessary directories
    print("📁 Creando directorios...")
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("src/api", exist_ok=True)
    
    # Create demo files
    print("📝 Creando archivos demo...")
    create_streamlit_demo()
    create_simple_html()
    
    print("✅ Setup completado!")
    print()
    
    # Show available options
    print("🚀 Opciones disponibles:")
    print()
    print("1️⃣  Demo HTML Simple:")
    print("   - Abre: static/index.html en tu navegador")
    print("   - Interfaz básica con controles interactivos")
    print()
    
    print("2️⃣  FastAPI Backend:")
    print("   - Ejecuta: python scripts/run_api.py")
    print("   - URL: http://localhost:8000")
    print()
    
    print("3️⃣  Streamlit Demo:")
    print("   - Instala: poetry install --extras ui")
    print("   - Ejecuta: python scripts/run_streamlit.py") 
    print("   - URL: http://localhost:8501")
    print()
    
    # Ask user preference
    choice = input("¿Qué opción quieres ejecutar? (1/2/3) o 'q' para salir: ").strip()
    
    if choice == "1":
        print("🌐 Abriendo demo HTML...")
        html_path = Path("static/index.html").absolute()
        print(f"📍 Abrir en navegador: file://{html_path}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{html_path}")
        except:
            print("⚠️  No se pudo abrir automáticamente. Abre manualmente el archivo.")
    
    elif choice == "2":
        print("🚀 Iniciando FastAPI...")
        try:
            subprocess.run([sys.executable, "scripts/run_api.py"])
        except KeyboardInterrupt:
            print("\n👋 FastAPI detenido")
    
    elif choice == "3":
        print("🎨 Iniciando Streamlit...")
        try:
            subprocess.run([sys.executable, "scripts/run_streamlit.py"])
        except KeyboardInterrupt:
            print("\n👋 Streamlit detenido")
    
    elif choice.lower() == "q":
        print("👋 ¡Hasta luego!")
    
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main() 