from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from config import settings

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register Socket.IO handlers
    from core.adk.chat_socket import register_chat_handlers
    from core.adk.training_socket import register_training_handlers
    register_chat_handlers(sio)
    register_training_handlers(sio)
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="ESG Portfolio Optimization API",
    description="AI-powered ESG investment chatbot backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGIN_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "esg-portfolio-backend"}


# Mount Socket.IO on the FastAPI ASGI app
socket_app = socketio.ASGIApp(sio, app)
