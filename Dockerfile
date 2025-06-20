# Multi-stage Dockerfile for ViT-GIF Highlight
# Supports both CPU and GPU deployments

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy dependency files
WORKDIR /build
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --extras "all"

# ============================================
# Stage 2: CPU Runtime - Lightweight for CPU-only deployment
# ============================================
FROM python:3.11-slim as cpu-runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory and user
RUN useradd --create-home --shell /bin/bash vitgif
WORKDIR /app
RUN chown vitgif:vitgif /app

# Copy application code
COPY --chown=vitgif:vitgif . .

# Create necessary directories
RUN mkdir -p logs config/weights && \
    chown -R vitgif:vitgif logs config

USER vitgif

# Set environment variables
ENV PYTHONPATH=/app
ENV VITGIF_CONFIG_PATH=/app/config/mvp1.yaml
ENV VITGIF_DEVICE=cpu

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.cli", "--help"]

# ============================================
# Stage 3: GPU Runtime - CUDA support
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu-runtime

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy Python packages from builder (excluding torch which we just installed)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Create app directory and user
RUN useradd --create-home --shell /bin/bash vitgif
WORKDIR /app
RUN chown vitgif:vitgif /app

# Copy application code
COPY --chown=vitgif:vitgif . .

# Create necessary directories
RUN mkdir -p logs config/weights && \
    chown -R vitgif:vitgif logs config

USER vitgif

# Set environment variables
ENV PYTHONPATH=/app
ENV VITGIF_CONFIG_PATH=/app/config/mvp2.yaml
ENV VITGIF_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8000 8501

# Health check with GPU detection
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU available:' if torch.cuda.is_available() else 'CPU only'); import src; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.cli", "--help"]

# ============================================
# Stage 4: Development - Full development environment
# ============================================
FROM gpu-runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    jupyter \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    pytest \
    pytest-cov \
    black \
    ruff \
    isort \
    pre-commit

USER vitgif

# Set development environment variables
ENV VITGIF_ENV=development
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose additional ports for development
EXPOSE 8000 8501 8888

# Development command
CMD ["bash"]

# ============================================
# Build arguments and final stage selection
# ============================================
ARG BUILD_TARGET=cpu-runtime
FROM cpu-runtime as final

# Add build metadata
LABEL maintainer="ViT-GIF Team"
LABEL version="2.0.0"
LABEL description="ViT-GIF Highlight: Intelligent GIF generation from videos"
LABEL org.opencontainers.image.source="https://github.com/your-org/vit-gif-highlight"
LABEL org.opencontainers.image.documentation="https://vit-gif-highlight.readthedocs.io"

# Build information
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_BRANCH

LABEL build.date=${BUILD_DATE}
LABEL build.git.commit=${GIT_COMMIT}
LABEL build.git.branch=${GIT_BRANCH}

# Final setup
USER vitgif
WORKDIR /app

# Add entrypoint script for auto-detection
COPY --chown=vitgif:vitgif scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "src.cli", "--help"] 