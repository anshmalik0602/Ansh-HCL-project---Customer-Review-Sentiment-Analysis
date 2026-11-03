#!/usr/bin/env bash
# Startup script for Sentiment Analysis Web Server (Backend + Frontend)

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv virtual environment not found. Please create it first."
    exit 1
fi

echo "=========================================================="
echo " Starting Customer Review Sentiment Analysis Web App"
echo "=========================================================="
echo " Backend: FastAPI + PyTorch (Apple Silicon GPU / MPS)"
echo " Frontend: Single-Page Dashboard (Tailwind CSS)"
echo " URL: http://127.0.0.1:8000"
echo " API Docs: http://127.0.0.1:8000/docs"
echo "=========================================================="

python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
