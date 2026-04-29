"""FastAPI app factory."""
import logging
from fastapi import FastAPI

from src.database.connection import create_tables
from src.services.search.ingestion import create_index, create_search_pipeline
from src.services.search.client import get_opensearch_client
from src.logging import setup_logging

from api.routes import api_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(title="RAG in Production", version="1.0.0")
    app.include_router(api_router)

    @app.on_event("startup")
    def startup():
        create_tables()
        client = get_opensearch_client()
        create_index(client)
        create_search_pipeline(client)
        logger.info("Startup complete.")

    return app
