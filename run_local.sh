#!/bin/bash
# Quick start script for local development

cd "$(dirname "$0")"

echo "Starting Price Negotiation UI..."
echo "Open http://localhost:8000 in your browser"
echo ""

ENABLE_WEB_INTERFACE=true uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
