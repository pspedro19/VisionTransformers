# Vision Transformers GIF Generator

A production-ready system that converts videos to GIFs with AI-powered visual attention highlighting using Vision Transformers.

## Features

- **Real AI Processing**: Uses VideoMAE and TimeSformer models for actual attention detection
- **Multiple Overlay Styles**: Heatmap, highlight, glow, pulse, and transparent overlays
- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Adaptive Processing**: Smart frame sampling based on video content
- **Production Ready**: Fully functional pipeline with error handling and monitoring

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- 8GB+ RAM recommended
- 2GB+ free disk space

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VisionTransformers

# Install dependencies
pip install -r requirements.txt

# Or install via pip
pip install -e .
```

## Project Structure

```
VisionTransformers/
├── src/                    # Core source code
│   ├── api/               # FastAPI web interface
│   ├── core/              # Main pipeline components
│   │   ├── pipeline.py           # Main processing pipeline
│   │   ├── attention_engine.py   # AI attention detection
│   │   ├── gif_composer.py       # GIF generation and overlays
│   │   ├── video_decoder.py      # Video processing
│   │   ├── custom_attention.py   # Custom attention configurations
│   │   └── improve_attention.py  # Attention improvement strategies
│   └── models/            # Model factory and configurations
├── tests/                 # Test files
│   ├── test_pipeline.py          # Main pipeline tests
│   ├── test_working_pipeline.py  # Working pipeline verification
│   ├── test_gif_generation.py    # GIF generation tests
│   └── test_simple.py            # Basic functionality tests
├── scripts/               # Utility scripts
│   ├── generate_real_gif.py      # Real GIF generation script
│   ├── demo_quick.py             # Quick demo script
│   ├── quick_start.py            # Project setup script
│   └── run_api.py               # API server launcher
├── config/                # Configuration files
│   ├── mvp1.yaml         # Minimal viable configuration
│   ├── mvp2.yaml         # Balanced configuration
│   ├── high_quality.yaml # High quality settings
│   └── ultra_quality.yaml # Maximum quality settings
├── data/                  # Data directories
│   ├── uploads/          # Input videos
│   └── output/           # Generated GIFs
└── static/               # Web interface assets
```

## Quick Start

### 1. Generate a Real GIF

```bash
# Generate a GIF with attention highlighting
python scripts/generate_real_gif.py
```

### 2. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_working_pipeline.py
```

### 3. Start API Server

```bash
# Start FastAPI server on port 8000
python scripts/run_api.py

# Access the API at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### 4. Quick Demo

```bash
# Run a quick demonstration
python scripts/demo_quick.py
```

## Usage Examples

### Basic GIF Generation

```python
from src.core.pipeline import InMemoryPipeline

# Initialize pipeline
pipeline = InMemoryPipeline("config/mvp2.yaml")

# Process video
result = pipeline.process_video(
    video_path="data/uploads/sample.mp4",
    output_path="data/output/result.gif"
)

if result['success']:
    print(f"GIF created: {result['gif_stats']['file_size_mb']:.2f} MB")
```

### Custom Configuration

```python
# Custom attention configuration
config = {
    "gif": {
        "fps": 10,
        "max_frames": 25,
        "overlay_style": "heatmap",
        "overlay_intensity": 0.8
    },
    "model": {
        "name": "videomae-base",
        "device": "auto"
    }
}

result = pipeline.process_video(
    video_path="input.mp4",
    output_path="output.gif",
    override_config=config
)
```

## Configuration Options

### GIF Settings
- `fps`: Frames per second (1-15)
- `max_frames`: Maximum frames to process (10-50)
- `overlay_style`: "heatmap", "highlight", "glow", "pulse", "transparent"
- `overlay_intensity`: Overlay strength (0.0-1.0)

### Model Settings
- `name`: "videomae-base", "videomae-large", "timesformer"
- `device`: "auto", "cpu", "cuda"
- `precision`: "fp16", "fp32"

### Processing Settings
- `adaptive_stride`: Smart frame sampling
- `min_stride`: Minimum frame interval
- `max_stride`: Maximum frame interval

## API Endpoints

- `POST /process-video`: Process video and generate GIF
- `GET /health`: Health check
- `GET /models`: Available models
- `GET /configs`: Available configurations

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run individual test
python tests/test_working_pipeline.py
```

## Performance

- **Processing Speed**: 2-10 seconds for 30-frame GIFs
- **Memory Usage**: 2-4GB RAM during processing
- **GPU Acceleration**: 2-5x faster with CUDA
- **Output Quality**: 720p-1080p resolution support

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_resolution` in config
2. **Slow Processing**: Enable GPU acceleration or reduce `max_frames`
3. **Large File Sizes**: Increase `optimization_level` or reduce `fps`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
pipeline = InMemoryPipeline("config/mvp1.yaml")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review test files for usage examples
- Run `python tests/test_simple.py` to verify system setup 