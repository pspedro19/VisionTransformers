#!/bin/bash
set -e

# ViT-GIF Highlight Docker Entrypoint
# Automatically detects hardware and sets optimal configuration

echo "🎬 ViT-GIF Highlight v2.0 - Docker Entrypoint"
echo "=============================================="

# Function to detect available hardware
detect_hardware() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "🚀 NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
            export VITGIF_DEVICE=${VITGIF_DEVICE:-cuda}
            export VITGIF_CONFIG_PATH=${VITGIF_CONFIG_PATH:-/app/config/mvp2.yaml}
            return 0
        fi
    fi
    
    echo "💻 CPU-only mode detected"
    export VITGIF_DEVICE=cpu
    export VITGIF_CONFIG_PATH=${VITGIF_CONFIG_PATH:-/app/config/mvp1.yaml}
    return 1
}

# Function to validate environment
validate_environment() {
    echo "🔍 Validating environment..."
    
    # Check Python imports
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError as e:
    print(f'PyTorch import error: {e}')
    sys.exit(1)

try:
    import src
    print('✅ ViT-GIF Highlight package imported successfully')
except ImportError as e:
    print(f'❌ ViT-GIF package import error: {e}')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Environment validation failed"
        exit 1
    fi
    
    echo "✅ Environment validation passed"
}

# Function to setup directories
setup_directories() {
    echo "📁 Setting up directories..."
    
    # Create required directories if they don't exist
    mkdir -p /app/logs
    mkdir -p /app/data/input
    mkdir -p /app/data/output
    mkdir -p /app/models
    
    # Set permissions
    chmod 755 /app/logs /app/data/input /app/data/output /app/models
    
    echo "✅ Directories setup complete"
}

# Function to check configuration
check_configuration() {
    echo "⚙️ Checking configuration..."
    
    if [ -f "$VITGIF_CONFIG_PATH" ]; then
        echo "✅ Configuration file found: $VITGIF_CONFIG_PATH"
        
        # Validate YAML syntax
        python -c "
import yaml
try:
    with open('$VITGIF_CONFIG_PATH', 'r') as f:
        config = yaml.safe_load(f)
    print('✅ Configuration YAML is valid')
    print(f'Model: {config.get(\"model\", {}).get(\"name\", \"unknown\")}')
    print(f'Device: {config.get(\"model\", {}).get(\"device\", \"unknown\")}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"
    else
        echo "⚠️ Configuration file not found: $VITGIF_CONFIG_PATH"
        echo "Using built-in defaults"
    fi
}

# Function to display system info
display_system_info() {
    echo "📊 System Information:"
    echo "  Device: $VITGIF_DEVICE"
    echo "  Config: $VITGIF_CONFIG_PATH"
    echo "  Python Path: $PYTHONPATH"
    echo "  Working Directory: $(pwd)"
    echo "  User: $(whoami)"
    
    if [ "$VITGIF_DEVICE" = "cuda" ]; then
        echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
    fi
    
    echo ""
}

# Function to run health check
health_check() {
    echo "🏥 Running health check..."
    
    python -c "
import torch
from src.models.model_factory import ModelFactory

# Basic imports test
try:
    from src.core.pipeline import InMemoryPipeline
    from src.core.video_decoder import OptimizedVideoDecoder
    print('✅ Core modules imported successfully')
except Exception as e:
    print(f'❌ Core module import error: {e}')
    exit(1)

# Model factory test
try:
    factory = ModelFactory()
    models = factory.list_available_models()
    print(f'✅ Available models: {len(models)}')
except Exception as e:
    print(f'⚠️ Model factory warning: {e}')

# Device test
if '$VITGIF_DEVICE' == 'cuda':
    if not torch.cuda.is_available():
        print('❌ CUDA requested but not available')
        exit(1)
    else:
        print('✅ CUDA available and working')
else:
    print('✅ CPU mode configured')

print('✅ Health check passed')
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Health check failed"
        exit 1
    fi
}

# Function to handle graceful shutdown
cleanup() {
    echo "🛑 Shutting down gracefully..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Clear GPU memory if using CUDA
    if [ "$VITGIF_DEVICE" = "cuda" ]; then
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('🧹 GPU memory cleared')
"
    fi
    
    echo "✅ Cleanup completed"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    # Run setup steps
    detect_hardware
    validate_environment
    setup_directories
    check_configuration
    display_system_info
    
    # Run health check unless disabled
    if [ "${SKIP_HEALTH_CHECK:-false}" != "true" ]; then
        health_check
    fi
    
    echo "🚀 Starting ViT-GIF Highlight..."
    echo ""
    
    # Execute the provided command
    if [ "$#" -eq 0 ]; then
        # No command provided, show help
        echo "No command provided. Available commands:"
        echo ""
        echo "  Process a video:"
        echo "    docker run vitgif process input.mp4 output.gif"
        echo ""
        echo "  Batch processing:"
        echo "    docker run vitgif batch /input /output"
        echo ""
        echo "  Start API server:"
        echo "    docker run -p 8000:8000 vitgif api"
        echo ""
        echo "  Start Streamlit UI:"
        echo "    docker run -p 8501:8501 vitgif ui"
        echo ""
        echo "  Interactive shell:"
        echo "    docker run -it vitgif bash"
        echo ""
        exec python -m src.cli --help
    elif [ "$1" = "api" ]; then
        # Start FastAPI server
        echo "🌐 Starting API server on port 8000..."
        exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2
    elif [ "$1" = "ui" ]; then
        # Start Streamlit UI
        echo "🎨 Starting Streamlit UI on port 8501..."
        exec streamlit run ui/streamlit_demo.py --server.address 0.0.0.0 --server.port 8501
    elif [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
        # Interactive shell
        echo "🐚 Starting interactive shell..."
        exec "$@"
    else
        # Pass through to CLI
        exec python -m src.cli "$@"
    fi
}

# Run main function
main "$@" 