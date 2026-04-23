from dotenv import load_dotenv
load_dotenv()
from config import settings
import traceback
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.router import api_router
from core.logging import configure_logging, get_logger
from core.response import error_response

configure_logging()
log = get_logger(__name__)

IS_DEV = settings.IS_DEV

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")



@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.adk.chat_socket import register_chat_handlers
    from core.adk.training_socket import register_training_handlers
    from core.adk.training_runner import init_training_runner
    from core.adk.chat_runner import init_chat_runner

    init_chat_runner() 
    init_training_runner()
    register_chat_handlers(sio)
    register_training_handlers(sio)
    yield


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


# ═══════════════════════════════════════
# Global Error Handlers
# ═══════════════════════════════════════


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "field": " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return error_response(
        message="Validation Error",
        details=details,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return error_response(
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        details=None,
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("unhandled exception method=%s path=%s", request.method, request.url.path)

    return error_response(
        message=str(exc) if IS_DEV else "An unexpected error occurred",
        details=traceback.format_exc() if IS_DEV else None,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ═══════════════════════════════════════
# Routes
# ═══════════════════════════════════════

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"success": True, "message": "Service is running", "data": {"service": "esg-portfolio-backend"}}


# Mount Socket.IO on the FastAPI ASGI app
socket_app = socketio.ASGIApp(sio, app)