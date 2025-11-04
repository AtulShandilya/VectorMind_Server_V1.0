#pm2 start gunicorn --name "fastapi-app" -- main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
pm2 start "uvicorn Server:app --host 0.0.0.0 --port 8000" --name LLM_3.0
