"""
Custom ASGI wrapper to increase request body size limits.
This allows handling large file uploads (>1MB default limit).
"""
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
import os

# Get max file size from environment (default 100MB)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_REQUEST_BODY_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

class LargeRequestMiddleware:
    """Middleware to increase request body size limit."""
    
    def __init__(self, app, max_request_body_size: int = MAX_REQUEST_BODY_SIZE):
        self.app = app
        self.max_request_body_size = max_request_body_size
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Increase the request body size limit
            scope["_max_request_body_size"] = self.max_request_body_size
        await self.app(scope, receive, send)

def wrap_app(app, max_request_body_size: int = MAX_REQUEST_BODY_SIZE):
    """
    Wrap FastAPI app to increase request body size limit.
    
    Args:
        app: FastAPI application instance
        max_request_body_size: Maximum request body size in bytes
        
    Returns:
        Wrapped ASGI application
    """
    return LargeRequestMiddleware(app, max_request_body_size)


