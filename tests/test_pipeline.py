"""
End-to-end tests for the main pipeline.
"""

import pytest
import torch
from pathlib import Path
import tempfile

from src.core.pipeline import InMemoryPipeline


class TestInMemoryPipeline:
    """Test suite for the complete pipeline."""
    
    def test_pipeline_initialization(self, config_mvp1):
        """Test pipeline initialization with config."""
        # Save config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_mvp1, f)
            config_path = f.name
            
        try:
            pipeline = InMemoryPipeline(config_path)
            assert pipeline.config is not None
            assert pipeline.decoder is not None
            assert pipeline.attention_engine is not None
            assert pipeline.gif_composer is not None
        finally:
            Path(config_path).unlink()
    
    def test_pipeline_with_default_config(self):
        """Test pipeline with non-existent config (should use defaults)."""
        pipeline = InMemoryPipeline("non_existent_config.yaml")
        assert pipeline.config is not None
        assert "model" in pipeline.config
        assert "limits" in pipeline.config
    
    @pytest.mark.slow
    def test_end_to_end_processing(self, sample_video_path, temp_dir, config_mvp1):
        """Test complete video to GIF processing."""
        output_path = temp_dir / "output.gif"
        
        # Save config
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        result = pipeline.process_video(
            str(sample_video_path),
            str(output_path)
        )
        
        # Check result structure
        assert result["success"] is True
        assert "total_frames" in result
        assert "selected_frames" in result
        assert "processing_time" in result
        assert "gif_stats" in result
        
        # Check output file exists
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Check that frames were selected
        assert result["selected_frames"] > 0
        assert result["selected_frames"] <= result["total_frames"]
    
    def test_process_video_with_overrides(self, sample_video_path, temp_dir, config_mvp1):
        """Test processing with configuration overrides."""
        output_path = temp_dir / "output_override.gif"
        
        # Save config
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        override_config = {
            "gif": {
                "fps": 2,
                "max_frames": 5,
                "overlay_intensity": 0.3
            }
        }
        
        result = pipeline.process_video(
            str(sample_video_path),
            str(output_path),
            override_config
        )
        
        assert result["success"] is True
        assert result["selected_frames"] <= 5  # Respects max_frames override
        assert result["fps_gif"] == 2  # Respects FPS override
    
    def test_video_limits_validation(self, temp_dir, config_mvp1):
        """Test that video limits are enforced."""
        # Create oversized video by modifying config
        config_mvp1["limits"]["max_resolution"] = 32  # Very small limit
        
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        # Create a video that exceeds the limit
        video_path = temp_dir / "large_video.mp4"
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (128, 128))  # Larger than limit
        
        for i in range(5):
            frame = torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8).numpy()
            out.write(frame)
        out.release()
        
        result = pipeline.process_video(
            str(video_path),
            str(temp_dir / "output.gif")
        )
        
        assert result["success"] is False
        assert "Resolution exceeds limit" in result.get("error", "")
    
    def test_get_video_preview(self, sample_video_path, config_mvp1, temp_dir):
        """Test video preview functionality."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        preview = pipeline.get_video_preview(str(sample_video_path))
        
        assert "video_info" in preview
        assert "within_limits" in preview
        assert "can_process" in preview
        assert "estimated_processing_time" in preview
        
        video_info = preview["video_info"]
        assert "width" in video_info
        assert "height" in video_info
        assert "duration" in video_info
        assert "fps" in video_info
        
        # For our test video, should be processable
        assert preview["can_process"] is True
    
    def test_batch_processing(self, temp_dir, config_mvp1):
        """Test batch processing of multiple videos."""
        # Create multiple small test videos
        video_paths = []
        for i in range(3):
            video_path = temp_dir / f"test_video_{i}.mp4"
            
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 64))
            
            for j in range(5):
                frame = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy()
                out.write(frame)
            out.release()
            
            video_paths.append(str(video_path))
        
        # Save config
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        output_dir = temp_dir / "batch_output"
        
        results = pipeline.process_batch(video_paths, str(output_dir))
        
        assert len(results) == 3
        
        # Check that all succeeded
        successful = sum(1 for r in results if r.get("success", False))
        assert successful == 3
        
        # Check output files exist
        for i in range(3):
            expected_output = output_dir / f"test_video_{i}_highlight.gif"
            assert expected_output.exists()
    
    def test_select_key_frames(self, config_mvp1, temp_dir):
        """Test key frame selection algorithm."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        # Create mock importance scores
        importance_scores = [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.1, 0.5, 0.6, 0.4]
        fps = 10.0
        max_frames = 5
        min_distance_frames = 2
        
        selected_indices = pipeline._select_key_frames(
            importance_scores, fps, max_frames, min_distance_frames
        )
        
        # Should select frames with highest importance
        assert len(selected_indices) <= max_frames
        assert len(selected_indices) > 0
        
        # Should be sorted chronologically
        assert selected_indices == sorted(selected_indices)
        
        # Should respect minimum distance
        for i in range(len(selected_indices) - 1):
            assert selected_indices[i + 1] - selected_indices[i] >= min_distance_frames
    
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_device_compatibility(self, device, sample_video_path, temp_dir, config_mvp1):
        """Test pipeline works on both CPU and GPU."""
        config_mvp1["model"]["device"] = device
        
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        output_path = temp_dir / f"output_{device}.gif"
        
        result = pipeline.process_video(
            str(sample_video_path),
            str(output_path)
        )
        
        assert result["success"] is True
        assert output_path.exists()
    
    def test_config_merging(self, config_mvp1, temp_dir):
        """Test configuration merging logic."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        base_config = {"a": {"b": 1, "c": 2}, "d": 3}
        override_config = {"a": {"b": 10}, "e": 4}
        
        merged = pipeline._merge_configs(base_config, override_config)
        
        assert merged["a"]["b"] == 10  # Overridden
        assert merged["a"]["c"] == 2   # Preserved
        assert merged["d"] == 3        # Preserved
        assert merged["e"] == 4        # Added
    
    def test_processing_time_estimation(self, config_mvp1, temp_dir):
        """Test processing time estimation."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        video_info = {
            "width": 640,
            "height": 480,
            "duration": 10.0,
            "fps": 30.0
        }
        
        estimated_time = pipeline._estimate_processing_time(video_info)
        
        assert isinstance(estimated_time, float)
        assert estimated_time > 0
        assert estimated_time >= 1.0  # Minimum is 1 second
    
    def test_error_handling_invalid_video(self, temp_dir, config_mvp1):
        """Test error handling for invalid video files."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        # Try to process non-existent file
        result = pipeline.process_video(
            "nonexistent_video.mp4",
            str(temp_dir / "output.gif")
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_recommended_settings(self, config_mvp1, temp_dir):
        """Test settings recommendation based on video characteristics."""
        config_path = temp_dir / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_mvp1, f)
            
        pipeline = InMemoryPipeline(str(config_path))
        
        # Test short video
        short_video_info = {"duration": 3.0, "width": 640, "height": 480}
        rec_short = pipeline._recommend_settings(short_video_info)
        
        # Test long video
        long_video_info = {"duration": 30.0, "width": 1920, "height": 1080}
        rec_long = pipeline._recommend_settings(long_video_info)
        
        # Short videos should have higher FPS
        assert rec_short["gif"]["fps"] >= rec_long["gif"]["fps"]
        
        # High-res videos should have lower overlay intensity
        assert rec_long["gif"]["overlay_intensity"] <= rec_short["gif"]["overlay_intensity"] 