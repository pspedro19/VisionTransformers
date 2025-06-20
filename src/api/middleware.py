"""
Middleware for security and rate limiting.
"""

import time
import magic
import logging
from typing import Dict, Any
from fastapi import HTTPException, UploadFile
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Security middleware for file validation."""
    
    ALLOWED_MIME_TYPES = [
        'video/mp4',
        'video/avi', 
        'video/quicktime',
        'video/x-msvideo',
        'video/webm'
    ]
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    async def validate_upload(self, file: UploadFile) -> bool:
        """
        Validate uploaded file for security.
        
        Args:
            file: Uploaded file object
            
        Returns:
            True if valid
            
        Raises:
            HTTPException: If validation fails
        """
        # Check file size
        if hasattr(file, 'size') and file.size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Read first chunk to detect MIME type
        chunk = await file.read(2048)
        await file.seek(0)  # Reset file pointer
        
        # Detect MIME type
        try:
            detected_mime = magic.from_buffer(chunk, mime=True)
        except Exception as e:
            logger.error(f"MIME detection failed: {e}")
            raise HTTPException(
                status_code=400,
                detail="Could not detect file type"
            )
        
        # Validate MIME type
        if detected_mime not in self.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {detected_mime}. "
                       f"Allowed types: {', '.join(self.ALLOWED_MIME_TYPES)}"
            )
        
        # Additional filename validation
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        # Check for suspicious filename patterns
        suspicious_patterns = ['..', '/', '\\', '<', '>', '|', ':']
        if any(pattern in file.filename for pattern in suspicious_patterns):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename"
            )
        
        logger.info(f"File validation passed: {file.filename} ({detected_mime})")
        return True


class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self, requests_per_minute: int = 30, requests_per_hour: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute per IP
            requests_per_hour: Max requests per hour per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests per IP
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limits.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if within limits
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        current_time = time.time()
        
        # Clean old entries and check minute limit
        minute_queue = self.minute_requests[client_ip]
        while minute_queue and current_time - minute_queue[0] > 60:
            minute_queue.popleft()
        
        if len(minute_queue) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: too many requests per minute"
            )
        
        # Clean old entries and check hour limit
        hour_queue = self.hour_requests[client_ip]
        while hour_queue and current_time - hour_queue[0] > 3600:
            hour_queue.popleft()
        
        if len(hour_queue) >= self.requests_per_hour:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: too many requests per hour"
            )
        
        # Add current request
        minute_queue.append(current_time)
        hour_queue.append(current_time)
        
        return True
    
    def get_rate_limit_info(self, client_ip: str) -> Dict[str, Any]:
        """Get rate limit information for client."""
        current_time = time.time()
        
        # Count current requests
        minute_queue = self.minute_requests[client_ip]
        hour_queue = self.hour_requests[client_ip]
        
        # Clean old entries
        while minute_queue and current_time - minute_queue[0] > 60:
            minute_queue.popleft()
        while hour_queue and current_time - hour_queue[0] > 3600:
            hour_queue.popleft()
        
        return {
            "requests_per_minute": {
                "current": len(minute_queue),
                "limit": self.requests_per_minute,
                "remaining": max(0, self.requests_per_minute - len(minute_queue))
            },
            "requests_per_hour": {
                "current": len(hour_queue),
                "limit": self.requests_per_hour,
                "remaining": max(0, self.requests_per_hour - len(hour_queue))
            }
        } 