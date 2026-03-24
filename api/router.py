from fastapi import APIRouter

from .auth import router as auth_router
from .training import router as training_router

api_router = APIRouter(prefix="/api")

api_router.include_router(auth_router)
api_router.include_router(training_router)

# Additional routers will be added as they are built:
# from .chat import router as chat_router
# from .sectors import router as sectors_router
# from .companies import router as companies_router
# from .portfolio import router as portfolio_router
# from .admin import router as admin_router
