"""
FastAPI application for video attention processing with comprehensive logging.
"""

import logging
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import shutil
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..core.pipeline import InMemoryPipeline

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global storage for tasks
TASKS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Vision GIF Generator", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize pipeline with detailed logging
try:
    logger.info("Initializing pipeline...")
    pipeline = InMemoryPipeline()
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    raise

# Pydantic models for request validation
class ProcessRequest(BaseModel):
    file_id: str
    start_time: float = 0.0
    duration: Optional[float] = None
    fps: int = 10
    max_frames: int = 50
    overlay_style: str = "transparent"
    overlay_intensity: float = 0.5
    overlay_color: str = "blue"
    model_name: str = "videomae-base"

@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        logger.error(f"Failed to serve index: {e}")
        raise HTTPException(status_code=500, detail="Failed to load interface")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file with detailed validation."""
    logger.info(f"Upload request received: {file.filename}")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            logger.warning(f"Invalid file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid file format. Use MP4, AVI, MOV, or WEBM")
        
        # Get file size
        file_size = 0
        if hasattr(file, 'size') and file.size:
            file_size = file.size
        else:
            # Read content to get size
            content = await file.read()
            file_size = len(content)
            await file.seek(0)  # Reset file pointer
        
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Validate file size (200MB max)
        if file_size > 200 * 1024 * 1024:
            logger.warning(f"File too large: {file_size / 1024 / 1024:.2f} MB")
            raise HTTPException(status_code=413, detail="File too large. Maximum 200MB")
        
        # Create temporary file
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_path = temp_dir / f"{file_id}_{file.filename}"
        
        logger.info(f"Saving file to: {file_path}")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            if hasattr(file, 'size') and file.size:
                shutil.copyfileobj(file.file, buffer)
            else:
                buffer.write(content)
        
        # Verify file was saved
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="Failed to save file")
            
        actual_size = file_path.stat().st_size
        logger.info(f"File saved successfully. Actual size: {actual_size / 1024 / 1024:.2f} MB")
        
        # Get video metadata
        try:
            video_info = pipeline.get_video_preview(str(file_path))
            logger.info(f"Video metadata: {video_info}")
        except Exception as e:
            logger.warning(f"Failed to get video metadata: {e}")
            video_info = {"error": "Could not read video metadata"}
        
        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size_mb": actual_size / 1024 / 1024,
            "video_info": video_info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    duration: float = Form(10.0),
    model: str = Form("videomae-base"),
    color: str = Form("azul"),
    fps: int = Form(10),
    intensity: float = Form(0.4)
):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        file_size = 0
        
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                buffer.write(chunk)
                
        logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Validate file size (200MB max)
        if file_size > 200 * 1024 * 1024:
            logger.warning(f"File too large: {file_size / 1024 / 1024:.2f} MB")
            raise HTTPException(status_code=413, detail="File too large. Maximum 200MB")
        
        # Get video info for validation
        video_info = pipeline.get_video_preview(str(file_path))
        
        if not video_info.get("can_process", False):
            error_details = []
            limits = video_info.get("within_limits", {})
            
            if not limits.get("resolution", True):
                error_details.append("Resolución excede el límite")
            if not limits.get("duration", True):
                error_details.append("Duración excede el límite")
            if not limits.get("file_size", True):
                error_details.append("Tamaño de archivo excede el límite")
                
            raise HTTPException(
                status_code=400, 
                detail=f"Video no puede ser procesado: {', '.join(error_details)}"
            )
            
        # Validate and adjust duration
        total_duration = video_info["video_info"]["duration"]
        if start_time >= total_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Tiempo de inicio ({start_time}s) excede la duración del video ({total_duration}s)"
            )
            
        # Ensure duration doesn't exceed video length or 10s limit
        max_duration = min(10.0, total_duration - start_time)
        duration = min(duration, max_duration)
        
        # Process video
        output_path = Path("data/output") / f"{uuid.uuid4()}_tracked.gif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_override = {
            "model": {
                "name": model
            },
            "gif": {
                "fps": fps,
                "overlay_intensity": intensity,
                "overlay_color": color
            },
            "video_segment": {
                "start_time": start_time,
                "duration": duration
            }
        }
        
        result = pipeline.process_video(
            str(file_path),
            str(output_path),
            config_override
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "gif_url": f"/download/{output_path.name}",
                "metadata": result.get("metadata", {})
            }
        else:
            error_msg = result.get("error", "Error desconocido durante el procesamiento del video.")
            logger.error(f"Processing failed for {file.filename}: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
    except HTTPException as e:
        logger.error(f"HTTP error processing video {file.filename}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error inesperado al procesar el video: {str(e)}"
        )
        
    finally:
        # Cleanup
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up temp file: {e}")

@app.get("/download/{filename}")
async def download_gif(filename: str):
    file_path = Path("data/output") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="GIF no encontrado")
        
    return FileResponse(
        file_path,
        media_type="image/gif",
        filename=f"vision_tracked_{filename}"
    )

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Get processing status with detailed information."""
    if task_id not in TASKS:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = TASKS[task_id].copy()
    # Remove sensitive information
    if "result" in task_info and isinstance(task_info["result"], dict):
        task_info["result"] = {k: v for k, v in task_info["result"].items() 
                             if k not in ["video_path", "file_path"]}
    
    return JSONResponse(task_info)

@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """Download processed GIF."""
    logger.info(f"Download request for task: {task_id}")
    
    if task_id not in TASKS:
        logger.warning(f"Task not found for download: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        logger.warning(f"Task not completed for download: {task_id}, status: {task['status']}")
        raise HTTPException(status_code=400, detail="Task not completed")
    
    output_path = Path(task["output_path"])
    
    if not output_path.exists():
        logger.error(f"Output file not found: {output_path}")
        raise HTTPException(status_code=404, detail="Output file not found")
    
    logger.info(f"Serving download: {output_path}")
    
    return FileResponse(
        path=str(output_path),
        filename=f"attention_gif_{task_id}.gif",
        media_type="image/gif"
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint with system information."""
    import torch
    
    active_tasks = len([t for t in TASKS.values() if t["status"] == "processing"])
    
    return JSONResponse({
        "status": "healthy",
        "active_tasks": active_tasks,
        "total_tasks": len(TASKS),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

@app.get("/api/models")
async def get_available_models():
    """Get available models and their information."""
    try:
        models = pipeline.model_factory.list_available_models()
        return JSONResponse({"models": models})
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return JSONResponse({"models": {"videomae-base": "Default model"}})

@app.delete("/api/cleanup")
async def cleanup_old_files():
    """Cleanup old uploaded files and completed tasks."""
    logger.info("Starting cleanup")
    
    try:
        # Clean up upload directory
        upload_dir = Path("data/uploads")
        if upload_dir.exists():
            for file in upload_dir.glob("*"):
                if file.is_file():
                    file.unlink()
                    
        # Clean up completed tasks older than 1 hour (simplified)
        completed_tasks = [k for k, v in TASKS.items() if v["status"] in ["completed", "failed"]]
        for task_id in completed_tasks[:10]:  # Keep only recent 10
            if task_id in TASKS:
                del TASKS[task_id]
                
        logger.info(f"Cleanup completed. Removed {len(completed_tasks)} old tasks")
        
        return JSONResponse({
            "success": True,
            "cleaned_tasks": len(completed_tasks),
            "message": "Cleanup completed"
        })
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 