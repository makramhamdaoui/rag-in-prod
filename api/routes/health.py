from fastapi import APIRouter
from src.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "model": settings.ollama_model_name}
