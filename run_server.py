"""
Server startup script with increased request size limits.
Use this instead of direct uvicorn command for better control.
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
env_file = ".env.production" if os.path.exists(".env.production") else ".env"
load_dotenv(env_file, override=True)

# Get configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_REQUEST_BODY_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

if __name__ == "__main__":
    # Configure uvicorn with increased limits
    config = uvicorn.Config(
        "chatServer:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        limit_concurrency=1000,
        timeout_keep_alive=5,
        limit_max_requests=10000,
        # Increase request line and field size limits
        limit_request_line=8190,
        limit_request_fields=8190,
        # Note: Uvicorn doesn't have a direct max_request_body_size parameter
        # The actual limit is handled by the ASGI server and can be configured
        # via middleware or by using a reverse proxy like nginx
    )
    
    server = uvicorn.Server(config)
    server.run()


