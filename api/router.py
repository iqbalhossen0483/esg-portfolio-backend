from fastapi import APIRouter

from .admin import router as admin_router
from .auth import router as auth_router
from .chat import router as chat_router
from .companies import router as companies_router
from .portfolio import router as portfolio_router
from .sectors import router as sectors_router
from .training import router as training_router

api_router = APIRouter(prefix="/api")

api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(training_router)
api_router.include_router(sectors_router)
api_router.include_router(companies_router)
api_router.include_router(portfolio_router)
api_router.include_router(admin_router)
