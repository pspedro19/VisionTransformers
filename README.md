# 🎬 ViT-GIF Highlight v2.0

> **Generador Inteligente de GIFs con Atención Visual desde Videos**

Sistema modular que transforma videos cortos en GIFs inteligentes destacando automáticamente las regiones más importantes usando Vision Transformers específicos para video, con un enfoque claro en **Marketing y Redes Sociales**.

![Demo](docs/demo.gif)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# CPU version
docker run -v $(pwd)/videos:/input -v $(pwd)/output:/output \
  vitgif/highlight:cpu process /input/video.mp4 /output/awesome.gif

# GPU version (requires NVIDIA Docker)
docker run --gpus all -v $(pwd)/videos:/input -v $(pwd)/output:/output \
  vitgif/highlight:gpu process /input/video.mp4 /output/awesome.gif
```

### Option 2: Local Installation

```bash
# Install with Poetry (recommended)
git clone https://github.com/your-org/vit-gif-highlight.git
cd vit-gif-highlight
poetry install --extras "all"

# Or with pip
pip install vit-gif-highlight[all]

# Process a video
vitgif process video.mp4 output.gif
```

### Option 3: Python API

```python
import src as vitgif

# Quick processing
result = vitgif.process_video("video.mp4", "output.gif")

# Advanced usage
from src.core.pipeline import InMemoryPipeline

pipeline = InMemoryPipeline("config/mvp2.yaml")
result = pipeline.process_video(
    "input.mp4", 
    "output.gif",
    override_config={
        "gif": {"fps": 8, "overlay_style": "glow"},
        "model": {"name": "videomae-large"}
    }
)
```

## 📊 Performance Benchmarks

| Video Resolution | Duration | GPU Time | CPU Time | Quality Score |
|------------------|----------|----------|----------|---------------|
| 720p            | 30s      | 12s      | 45s      | 0.89         |
| 480p            | 15s      | 5s       | 20s      | 0.92         |
| 1080p           | 60s      | 28s      | 120s     | 0.87         |

*Tested on RTX 4090 / Intel i9-12900K*

## 🏗️ Architecture

```mermaid
graph LR
    A[📹 Video Input] --> B[🔍 Decord GPU Decoder]
    B --> C[🧠 Video Transformer]
    C --> D[👁️ Attention Maps]
    D --> E[🎯 Key Frame Selection]
    E --> F[🎨 Attention Overlay]
    F --> G[📀 Optimized GIF]
    
    H[⚙️ Config] --> B
    H --> C
    H --> F
    
    I[📊 MLflow] --> J[📈 Metrics]
    C --> J
    E --> J
```

## ✨ Key Features

### 🎯 MVP-1 (Available Now)
- ✅ **GPU-Accelerated Processing** - 60% faster with Decord + CUDA
- ✅ **Video-Specific Models** - VideoMAE, TimeSformer, Video-Swin
- ✅ **Intelligent Frame Selection** - Non-max suppression temporal
- ✅ **Security Limits** - 720p/60s/100MB built-in protection
- ✅ **Multiple Overlay Styles** - Heatmap, highlight, glow, pulse, transparent
- ✅ **CLI + Python API** - Easy integration
- ✅ **Docker Support** - CPU/GPU variants

### 🚀 MVP-2 (Coming Soon)
- 🔄 **Multi-scale Attention** - Better detail preservation
- 🔄 **Temporal Coherence** - Smooth attention transitions
- 🔄 **REST API** - Production-ready FastAPI
- 🔄 **Streamlit UI** - Real-time preview
- 🔄 **Batch Processing** - Multiple videos at once

### 🌟 Pro Features (Roadmap)
- 📋 **Custom Fine-tuning** - Domain-specific models
- 🎨 **Artistic Styles** - Neural style transfer
- 📝 **Auto Descriptions** - AI-generated captions
- ⚡ **Token Fusion** - 50% memory reduction

## 🛠️ Installation

### System Requirements

- **Python**: 3.9+
- **GPU**: NVIDIA with 4GB+ VRAM (optional but recommended)
- **Memory**: 8GB+ RAM
- **Storage**: 5GB+ for models

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-org/vit-gif-highlight.git
cd vit-gif-highlight

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --extras "all"

# Setup pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest
```

### Docker Installation

```bash
# Build CPU version
docker build --target cpu-runtime -t vitgif:cpu .

# Build GPU version (requires NVIDIA Docker)
docker build --target gpu-runtime -t vitgif:gpu .

# Build development environment
docker build --target development -t vitgif:dev .
```

## 📖 Usage Examples

### CLI Interface

```bash
# Basic usage
vitgif process video.mp4 output.gif

# Custom settings
vitgif process video.mp4 output.gif \
  --fps 8 \
  --model videomae-large \
  --overlay-style glow \
  --overlay-intensity 0.8

# Batch processing
vitgif batch /path/to/videos/ /path/to/output/ \
  --config config/mvp2.yaml

# Preview video info
vitgif preview video.mp4

# List available models
vitgif models
```

### Python API

```python
# Simple processing
import src as vitgif

result = vitgif.process_video(
    "input.mp4", 
    "output.gif",
    fps=8,
    model="videomae-large"
)

print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Compression ratio: {result['compression_ratio']:.1f}x")
```

### Advanced Pipeline Usage

```python
from src.core.pipeline import InMemoryPipeline

# Initialize with custom config
pipeline = InMemoryPipeline("config/mvp2.yaml")

# Get video preview
preview = pipeline.get_video_preview("video.mp4")
print(f"Can process: {preview['can_process']}")
print(f"Estimated time: {preview['estimated_processing_time']:.1f}s")

# Process with overrides
result = pipeline.process_video(
    "video.mp4",
    "output.gif",
    override_config={
        "gif": {
            "fps": 10,
            "max_frames": 25,
            "overlay_style": "pulse"
        },
        "model": {
            "name": "timesformer-base"
        }
    }
)

# Batch processing
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = pipeline.process_batch(videos, "output_dir/")
```

## ⚙️ Configuration

### Model Selection

| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| `videomae-base` | ⚡⚡⚡ | ⭐⭐⭐ | 2-3GB | General use |
| `videomae-large` | ⚡⚡ | ⭐⭐⭐⭐ | 4-6GB | High quality |
| `timesformer-base` | ⚡⚡⚡ | ⭐⭐⭐ | 2-4GB | Speed focused |
| `videomae-huge` | ⚡ | ⭐⭐⭐⭐⭐ | 8-12GB | Maximum quality |

### Overlay Styles

- **`heatmap`** - Classic attention heatmap (red = high attention)
- **`highlight`** - Brighten important regions
- **`glow`** - Golden glow effect around key areas
- **`pulse`** - Animated pulse effect
- **`transparent`** - Faint overlay showing attention without obstruction

### Configuration Files

```yaml
# config/mvp1.yaml - Basic configuration
model:
  name: "videomae-base"
  device: "cuda"
  precision: "fp16"

limits:
  max_resolution: 720
  max_duration: 60
  max_file_size: 100

gif:
  fps: 5
  max_frames: 20
  overlay_style: "heatmap"
  overlay_intensity: 0.7
```

## 🐳 Docker Usage

### Quick Start with Docker Compose

```bash
# CPU processing
docker-compose --profile cpu up vitgif-cpu

# GPU processing (requires NVIDIA Docker)
docker-compose --profile gpu up vitgif-gpu

# Development environment
docker-compose --profile dev up vitgif-dev

# Full stack with monitoring
docker-compose --profile gpu --profile mlflow --profile monitoring up
```

### Environment Variables

```bash
# Core settings
VITGIF_DEVICE=cuda              # Device selection
VITGIF_CONFIG_PATH=/app/config/mvp2.yaml
PYTHONPATH=/app

# GPU settings
CUDA_VISIBLE_DEVICES=0          # GPU selection

# Performance tuning
OMP_NUM_THREADS=4               # CPU threads (CPU mode)
TORCH_COMPILE_MODE=reduce-overhead  # PyTorch 2.0 optimization
```

## 📊 Monitoring & Metrics

### MLflow Integration

The system automatically tracks metrics when configured:

```yaml
metrics:
  track_performance: true
  mlflow_uri: "http://localhost:5000"
```

Tracked metrics include:
- Processing time per video
- Frame selection efficiency
- Compression ratios
- Model performance
- Business KPIs (estimated CTR boost)

### Prometheus Metrics

When running with monitoring profile:

```bash
# View metrics
curl http://localhost:9090/metrics

# Grafana dashboard
open http://localhost:3000
# Login: admin / vitgif123
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run only fast tests
poetry run pytest -m "not slow"

# Run GPU tests (requires GPU)
poetry run pytest -m gpu

# Run specific test file
poetry run pytest tests/test_pipeline.py -v
```

## 🚀 Deployment

### Production with Docker

```bash
# Build production image
docker build --target gpu-runtime -t vitgif:prod .

# Run with docker-compose
docker-compose --profile prod --profile gpu up -d

# Check health
docker-compose exec vitgif-gpu python -c "import src; print('OK')"
```

### Kubernetes Deployment

```yaml
# k8s/vitgif-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vitgif-gpu
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vitgif
  template:
    metadata:
      labels:
        app: vitgif
    spec:
      containers:
      - name: vitgif
        image: vitgif:gpu
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
        env:
        - name: VITGIF_CONFIG_PATH
          value: "/app/config/production.yaml"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Workflow

```bash
# 1. Fork and clone
git clone https://github.com/your-username/vit-gif-highlight.git

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Install development dependencies
poetry install --extras "all"
poetry run pre-commit install

# 4. Make changes and test
poetry run pytest
poetry run black src/
poetry run ruff check src/

# 5. Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 6. Create Pull Request
```

### Code Quality

We maintain high code quality with:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** hooks
- **Pytest** for testing (90%+ coverage)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** - Video model implementations
- **Decord** - High-performance video decoding
- **OpenAI CLIP** - Vision-language understanding inspiration
- **MMAction2** - Video understanding frameworks

## 📞 Support

- 📧 **Email**: support@vitgif-highlight.com
- 💬 **Discord**: [Join our community](https://discord.gg/vitgif)
- 📖 **Documentation**: [docs.vitgif-highlight.com](https://docs.vitgif-highlight.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/vit-gif-highlight/issues)

## 🗺️ Roadmap

### Q1 2024
- [ ] MVP-2 Release (API + UI)
- [ ] Temporal coherence improvements
- [ ] Mobile optimization

### Q2 2024
- [ ] Custom fine-tuning pipeline
- [ ] Enterprise features
- [ ] Cloud deployment templates

### Q3 2024
- [ ] Real-time processing
- [ ] Video streaming support
- [ ] Advanced analytics

---

<div align="center">

**⭐ Star us on GitHub if this project helped you! ⭐**

[Website](https://vitgif-highlight.com) • [Documentation](https://docs.vitgif-highlight.com) • [Examples](examples/) • [Changelog](CHANGELOG.md)

</div> 