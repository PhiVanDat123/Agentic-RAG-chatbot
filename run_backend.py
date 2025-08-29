import asyncio
import uvicorn
from backend import app
from config import config, validate_environment

async def start_backend():
    """Start the FastAPI backend server"""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()