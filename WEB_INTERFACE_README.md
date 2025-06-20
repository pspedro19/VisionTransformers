# 🎬 ViT-GIF Highlight - Interfaz Web Interactiva

## 🌟 **Experiencia de Usuario Web Completa**

Este proyecto incluye **dos interfaces web modernas** para convertir videos a GIFs inteligentes con overlays de atención visual:

### 🚀 **Opciones de Interfaz**

1. **FastAPI + Frontend Moderno** - Interfaz completa con controles interactivos
2. **Streamlit Demo** - Prototipo rápido para pruebas

---

## 📋 **Instalación y Configuración**

### **1. Instalar Dependencias**

```bash
# Instalar dependencias básicas
poetry install

# Para interfaz FastAPI (recomendado)
poetry install --extras api

# Para demo Streamlit
poetry install --extras ui

# Para ambas interfaces
poetry install --extras all
```

### **2. Configuración Inicial**

```bash
# Crear directorios necesarios
mkdir -p data/uploads data/output static

# Verificar instalación
poetry run python -c "import fastapi, uvicorn; print('✅ FastAPI listo')"
poetry run python -c "import streamlit; print('✅ Streamlit listo')"
```

---

## 🎯 **Opción 1: FastAPI + Frontend Interactivo**

### **Características Avanzadas:**
- ✨ **Timeline interactivo** con sliders arrastrablesll
- 🎮 **Controles de tiempo en tiempo real**
- 📊 **Vista previa en vivo** del segmento seleccionado
- ⚡ **WebSocket** para updates en tiempo real
- 🎨 **5 estilos de overlay** (heatmap, highlight, glow, pulse, transparent)
- 📱 **Responsivo** para móviles y tablets

### **🚀 Ejecutar FastAPI:**

```bash
# Opción 1: Script directo
python scripts/run_api.py

# Opción 2: Comando manual
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**🌐 Abrir:** http://localhost:8000

### **📱 Flujo de Usuario FastAPI:**

1. **📁 Cargar Video**
   - Drag & drop o click para seleccionar
   - Validación automática de formato y tamaño
   - Preview inmediato del video

2. **⏱️ Seleccionar Tiempo**
   - Timeline visual interactivo
   - Handles arrastrables para inicio/fin
   - Botones de duración rápida (3s, 5s, 10s, etc.)
   - Input manual de tiempo exacto
   - Preview del segmento seleccionado

3. **⚙️ Configurar GIF**
   - FPS slider (1-15)
   - Máximo de frames (5-50)
   - Estilo de overlay visual
   - Intensidad de atención (0-100%)
   - Selección de modelo IA
   - Nivel de optimización

4. **🚀 Generar**
   - Resumen de configuración
   - Estimación de tiempo
   - Progress bar en tiempo real
   - Steps de procesamiento visuales

5. **📥 Resultado**
   - Vista previa del GIF generado
   - Estadísticas de procesamiento
   - Botón de descarga
   - Opción de compartir

---

## 🎨 **Opción 2: Streamlit Demo**

### **Características:**
- 🎯 **Interfaz simplificada** para pruebas rápidas
- 📊 **Métricas en tiempo real**
- 🎛️ **Controles intuitivos**
- 📱 **Auto-responsive**

### **🚀 Ejecutar Streamlit:**

```bash
# Opción 1: Script directo
python scripts/run_streamlit.py

# Opción 2: Comando manual
poetry run streamlit run src/streamlit_demo.py --server.port 8501
```

**🌐 Abrir:** http://localhost:8501

### **📱 Flujo de Usuario Streamlit:**

1. **📁 Upload** - Arrastra video o usa file picker
2. **⏱️ Timeline** - Sliders para seleccionar segmento  
3. **⚙️ Settings** - Configuración visual y modelo IA
4. **🚀 Generate** - Procesamiento con progress simulado
5. **📊 Results** - Estadísticas y descarga

---

## 🎯 **Funcionalidades Interactivas Clave**

### **Timeline Interactivo (FastAPI)**
```javascript
// Controles arrastrables
- Inicio: Click y arrastra handle izquierdo
- Fin: Click y arrastra handle derecho  
- Click en timeline: Mueve punto de inicio
- Botones rápidos: 3s, 5s, 10s, 15s, 30s
```

### **Configuración Avanzada**
```yaml
Overlay Styles:
  - 🔥 Heatmap: Mapa de calor clásico
  - ✨ Highlight: Brillo en áreas importantes
  - 🌟 Glow: Resplandor suave
  - 💫 Pulse: Pulsación dinámica

Modelos IA:
  - VideoMAE Base: Rápido (2-3GB GPU)
  - VideoMAE Large: Calidad (4-6GB GPU)  
  - TimeSformer: Eficiente (2-4GB GPU)
  - Automático: Selección inteligente
```

### **Validaciones de Seguridad**
- ✅ Tamaño máximo: 100MB
- ✅ Formatos: MP4, AVI, MOV, WEBM
- ✅ Duración máxima: 60 segundos
- ✅ Rate limiting por IP
- ✅ Validación MIME type

---

## 📊 **API Endpoints (FastAPI)**

### **Core Endpoints:**
```http
POST /api/upload          # Subir video
POST /api/process         # Procesar segmento  
GET  /api/status/{job_id} # Estado del trabajo
GET  /api/health          # Health check
WS   /ws/{job_id}         # WebSocket updates
```

### **Ejemplo de Uso:**
```bash
# Upload video
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@video.mp4"

# Process segment  
curl -X POST "http://localhost:8000/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "uuid-here",
    "start_time": 5.0,
    "duration": 10.0,
    "fps": 8,
    "overlay_style": "heatmap"
  }'
```

---

## 🎨 **Personalización del Frontend**

### **Archivos Clave:**
```
static/
├── index.html      # Estructura HTML
├── css/style.css   # Estilos modernos
└── js/app.js       # Lógica interactiva
```

### **Modificar Estilos:**
```css
/* En static/css/style.css */
:root {
  --primary-color: #3b82f6;    /* Color principal */
  --success-color: #10b981;    /* Color éxito */
  --error-color: #ef4444;      /* Color error */
}
```

### **Agregar Funcionalidades:**
```javascript
// En static/js/app.js
class ViTGIFApp {
  // Agregar nuevos métodos aquí
  customFeature() {
    // Tu código personalizado
  }
}
```

---

## 🔧 **Solución de Problemas**

### **Error: "Module not found"**
```bash
# Reinstalar dependencias
poetry install --extras all
poetry run pip list | grep -E "(fastapi|streamlit)"
```

### **Error: "Port already in use"**
```bash
# Cambiar puerto
poetry run uvicorn src.api.app:app --port 8001
poetry run streamlit run src/streamlit_demo.py --server.port 8502
```

### **Error: "Static files not found"**
```bash
# Verificar estructura
ls -la static/
mkdir -p static/css static/js
```

### **Performance Issues**
```yaml
# Optimizar configuración
gif:
  fps: 5              # Menor FPS = archivo más pequeño
  max_frames: 15      # Menos frames = más rápido
  optimization_level: 3  # Máxima compresión
```

---

## 📈 **Métricas y Monitoreo**

### **Métricas Disponibles:**
- ⏱️ Tiempo de procesamiento
- 💾 Tamaño de archivo resultante
- 🎯 Frames seleccionados por IA
- 📊 Ratio de compresión
- 🔄 Jobs activos y completados

### **Logs del Sistema:**
```bash
# Ver logs en tiempo real
poetry run uvicorn src.api.app:app --log-level debug

# Logs de Streamlit  
poetry run streamlit run src/streamlit_demo.py --logger.level debug
```

---

## 🚀 **Desarrollo y Extensión**

### **Agregar Nuevos Modelos:**
```python
# En src/models/model_factory.py
AVAILABLE_MODELS.update({
    "mi-modelo": {
        "class": "MiModeloCustom",
        "config": {...}
    }
})
```

### **Nuevos Estilos de Overlay:**
```python
# En src/core/gif_composer.py  
def apply_custom_overlay(self, frames, attention_maps):
    # Tu lógica de overlay personalizada
    return enhanced_frames
```

### **Deployment:**
```dockerfile
# Usar Dockerfile incluido
docker build -t vitgif-web .
docker run -p 8000:8000 vitgif-web

# O con docker-compose
docker-compose up web
```

---

## 📝 **Ejemplos de Uso**

### **Caso 1: Marketing Content**
```yaml
Configuración Recomendada:
  duration: 5-10s
  fps: 8-12  
  overlay_style: "highlight"
  model: "videomae-base"
  optimization: 2
```

### **Caso 2: Social Media**
```yaml
Configuración Óptima:
  duration: 3-5s
  fps: 5-8
  overlay_style: "pulse" 
  model: "timesformer-base"
  optimization: 3
```

### **Caso 3: Análisis Técnico**
```yaml
Configuración Detallada:
  duration: 10-15s
  fps: 12-15
  overlay_style: "heatmap"
  model: "videomae-large"
  optimization: 1
```

---

## 🎉 **¡Listo para Usar!**

**FastAPI Interface:** http://localhost:8000
**Streamlit Demo:** http://localhost:8501  
**API Docs:** http://localhost:8000/docs

### **Próximos Pasos:**
1. 🎬 Sube tu primer video
2. ⏱️ Selecciona el segmento perfecto
3. 🎨 Configura el estilo visual
4. 🚀 ¡Genera tu GIF inteligente!

---

*¿Necesitas ayuda? Revisa los logs, consulta la documentación de la API, o contacta al equipo de desarrollo.* 