import json
import logging
from typing import Dict

import ollama
from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.database.connection import SessionLocal
from src.database.models.document import Document, DocumentStatus
from src.services.agents.prompts import REWRITE_PROMPT
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def get_indexed_documents() -> str:
    """Fetch indexed document names from PostgreSQL."""
    try:
        db = SessionLocal()
        docs = db.query(Document.name).filter(
            Document.status == DocumentStatus.indexed
        ).all()
        db.close()
        if not docs:
            return "No documents currently indexed."
        return "\n".join([f"- {doc.name}" for doc in docs])
    except Exception as e:
        logger.warning(f"Could not fetch documents from DB: {e}")
        return "Documents available in the knowledge base."


def rewrite_node(state: AgentState) -> Dict:
    """Rewrite query using grading reason and document context."""
    logger.info("NODE: rewrite_query")
    settings = get_settings()

    original = state.get("original_query", "")
    grading_reason = state.get("grading_reasoning", "chunks were not relevant")
    document_topics = get_indexed_documents()

    prompt = REWRITE_PROMPT.format(
        question=original,
        grading_reason=grading_reason,
        document_topics=document_topics,
    )

    try:
        response = ollama.chat(
            model=settings.ollama_model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        content = response["message"]["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        rewritten = data.get("rewritten_query", original)
        reasoning = data.get("reasoning", "")
        logger.info(f"Rewritten: '{original[:50]}' → '{rewritten[:50]}'")
        logger.info(f"Reason: {reasoning}")
    except Exception as e:
        logger.warning(f"Rewrite LLM failed: {e} — keeping original")
        rewritten = original

    return {
        "rewritten_query": rewritten,
        "messages": [HumanMessage(content=rewritten)],
    }
