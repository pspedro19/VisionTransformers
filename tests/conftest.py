"""
PyTest configuration and shared fixtures.
"""

import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Global pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a sample video file for testing."""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
    
    # Generate 30 frames (3 seconds at 10 FPS)
    for i in range(30):
        # Create frame with moving gradient
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create a moving pattern
        x_offset = int(50 * np.sin(i * 0.2))
        y_offset = int(30 * np.cos(i * 0.15))
        
        # Draw a colored rectangle that moves
        cv2.rectangle(
            frame, 
            (50 + x_offset, 50 + y_offset), 
            (150 + x_offset, 150 + y_offset), 
            (255, 100, 50), 
            -1
        )
        
        # Add some noise for realism
        noise = np.random.randint(0, 20, (224, 224, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    
    return video_path


@pytest.fixture
def small_video_path(temp_dir):
    """Create a very small video for quick tests."""
    video_path = temp_dir / "small_video.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 64))
    
    # Generate 10 frames (2 seconds at 5 FPS)
    for i in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    return video_path


@pytest.fixture
def sample_frames():
    """Generate sample frame tensors for testing."""
    # Create 5 frames of 64x64 RGB
    frames = torch.rand(5, 64, 64, 3)
    return frames


@pytest.fixture
def sample_attention_maps():
    """Generate sample attention maps for testing."""
    # Create 5 attention maps of 64x64
    attention = torch.rand(5, 64, 64)
    # Normalize to 0-1
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention


@pytest.fixture
def config_mvp1():
    """MVP-1 configuration for testing."""
    return {
        "model": {
            "name": "videomae-base",
            "device": "cpu",  # Use CPU for testing to avoid GPU requirements
            "precision": "fp32"
        },
        "limits": {
            "max_resolution": 720,
            "max_duration": 60,
            "max_file_size": 100,
            "allowed_formats": ["mp4", "avi", "mov", "webm"]
        },
        "processing": {
            "adaptive_stride": True,
            "min_stride": 2,
            "max_stride": 5  # Reduced for testing
        },
        "gif": {
            "fps": 3,  # Lower FPS for faster testing
            "max_frames": 10,  # Fewer frames for testing
            "optimization_level": 1,
            "overlay_style": "heatmap",
            "overlay_intensity": 0.5
        },
        "metrics": {
            "track_performance": False  # Disable for testing
        }
    }


@pytest.fixture
def mock_model():
    """Mock model for testing without loading real transformers."""
    class MockModel:
        def __init__(self):
            self.device = "cpu"
            
        def __call__(self, x, output_attentions=False):
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            
            if output_attentions:
                # Mock attention output
                seq_len = 196  # 14x14 patches
                attention = torch.rand(batch_size, 8, seq_len + 1, seq_len + 1)  # 8 heads
                
                class MockOutput:
                    def __init__(self):
                        self.attentions = [attention]
                        self.logits = torch.rand(batch_size, 1000)
                        
                return MockOutput()
            else:
                class MockOutput:
                    def __init__(self):
                        self.logits = torch.rand(batch_size, 1000)
                        self.last_hidden_state = torch.rand(batch_size, seq_len + 1, 768)
                        
                return MockOutput()
                
        def eval(self):
            return self
            
        def to(self, device):
            self.device = device
            return self
            
        def half(self):
            return self
            
        def cpu(self):
            return self
    
    return MockModel()


# Skip markers for CI/CD
def pytest_runtest_setup(item):
    """Skip GPU tests if no GPU available."""
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("GPU not available")


# Performance tracking
@pytest.fixture(autouse=True)
def track_performance(request):
    """Track test performance automatically."""
    import time
    start_time = time.time()
    
    yield
    
    duration = time.time() - start_time
    test_name = request.node.name
    
    # Log slow tests
    if duration > 10.0:  # 10 seconds
        print(f"\nSLOW TEST: {test_name} took {duration:.2f}s")


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_torch():
    """Clean up torch resources after each test."""
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Reduce logging noise during testing
    import logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("decord").setLevel(logging.WARNING)
    
    yield
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 