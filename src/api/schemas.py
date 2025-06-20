"""
Pydantic schemas for the ViT-GIF Highlight API.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VideoMetadata(BaseModel):
    """Video metadata information."""
    width: int
    height: int
    total_frames: int
    fps: float
    duration: float
    file_size: int


class VideoUploadResponse(BaseModel):
    """Response for video upload."""
    job_id: str
    filename: str
    metadata: VideoMetadata
    can_process: bool
    estimated_time: float
    recommended_settings: Dict[str, Any]


class ProcessRequest(BaseModel):
    """Request to process a video segment."""
    job_id: str
    start_time: float = Field(ge=0, description="Start time in seconds")
    duration: float = Field(gt=0, le=60, description="Duration in seconds (max 60)")
    fps: int = Field(default=5, ge=1, le=15, description="Output GIF FPS")
    max_frames: int = Field(default=20, ge=5, le=50, description="Maximum frames")
    overlay_style: str = Field(default="heatmap", regex="^(heatmap|highlight|glow|pulse)$")
    overlay_intensity: float = Field(default=0.7, ge=0.0, le=1.0)
    optimization_level: int = Field(default=2, ge=0, le=3)
    model_name: Optional[str] = Field(default=None, description="Optional model override")


class ProcessResponse(BaseModel):
    """Response for process request."""
    job_id: str
    status: str
    message: str


class ProcessStatus(BaseModel):
    """Status of a processing job."""
    job_id: str
    status: str  # uploaded, processing, completed, failed
    progress: int = Field(ge=0, le=100)
    message: str = ""
    result_url: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    active_jobs: int
    websocket_connections: int 