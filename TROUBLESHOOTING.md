# Troubleshooting: 413 Request Entity Too Large

## Problem
Getting "413 Request Entity Too Large" error when uploading files.

## Solutions

### Solution 1: Configure Uvicorn (Recommended for Development)

The server.sh script has been updated with increased limits. Restart your server:

```bash
pm2 restart LLM_3.0
# Or if not using PM2:
uvicorn chatServer:app --host 0.0.0.0 --port 8000 --limit-request-line 8190 --limit-request-field_size 8190
```

### Solution 2: Use Nginx as Reverse Proxy (Recommended for Production)

1. Install nginx:
```bash
sudo apt-get install nginx
```

2. Create/edit nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/your-app
```

3. Add configuration (see `nginx_config_example.conf`):
```nginx
client_max_body_size 100M;
```

4. Restart nginx:
```bash
sudo systemctl restart nginx
```

### Solution 3: Increase Environment Variable

Set the maximum file size in your `.env` file:
```env
MAX_FILE_SIZE_MB=200  # Increase to 200MB or higher
```

### Solution 4: Check Current File Size

The application now validates file size and returns a clear error message if the file exceeds the limit. Check the error message for the actual file size.

### Solution 5: For Very Large Files (>100MB)

1. Use nginx as reverse proxy (see Solution 2)
2. Increase `client_max_body_size` in nginx config
3. Increase `MAX_FILE_SIZE_MB` in environment variables
4. Consider chunked uploads for very large files

## Current Limits

- Default max file size: 100MB (configurable via `MAX_FILE_SIZE_MB`)
- Uvicorn request line limit: 8190 bytes
- Uvicorn request field size: 8190 bytes

## Verification

After applying changes, test with:
```bash
curl -X POST "http://localhost:8000/chat1" \
  -F "select=input" \
  -F "message=test" \
  -F "file=@your_large_file.pdf"
```


