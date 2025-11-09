#pm2 start gunicorn --name "fastapi-app" -- main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
#source venv/bin/activate
# Increase request size limits
# Note: For very large files (>100MB), use nginx as reverse proxy with client_max_body_size
# See nginx_config_example.conf for nginx configuration
pm2 start "uvicorn chatServer:app --host 0.0.0.0 --port 8000 --limit-concurrency 1000 --timeout-keep-alive 5 --limit-max-requests 10000 --limit-request-line 8190 --limit-request-field_size 8190" --name LLM_3.0
