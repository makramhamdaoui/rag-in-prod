from fastapi import APIRouter

from api.routes.health import router as health_router
from api.routes.sessions import router as sessions_router
from api.routes.documents import router as documents_router
from api.routes.chat import router as chat_router
from api.routes.agentic import router as agentic_router

# combined router for main app
api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(sessions_router)
api_router.include_router(documents_router)
api_router.include_router(chat_router)
api_router.include_router(agentic_router)
