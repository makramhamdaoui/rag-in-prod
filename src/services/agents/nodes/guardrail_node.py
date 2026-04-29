import json
import logging
from typing import Dict

import ollama

from src.config import get_settings
from src.database.connection import SessionLocal
from src.database.models.document import Document, DocumentStatus
from src.services.agents.prompts import GUARDRAIL_PROMPT
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def get_latest_query(state: AgentState) -> str:
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


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


def guardrail_node(state: AgentState) -> Dict:
    """Evaluate if query is relevant to our knowledge base."""
    logger.info("NODE: guardrail")
    query = get_latest_query(state)
    settings = get_settings()

    # get real document names from DB
    document_topics = get_indexed_documents()
    logger.info(f"Guardrail checking against documents:\n{document_topics}")

    prompt = GUARDRAIL_PROMPT.format(
        question=query,
        document_topics=document_topics,
    )
    try:
        response = ollama.chat(
            model=settings.ollama_model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        score = int(data.get("score", 50))
        reason = data.get("reason", "")
        logger.info(f"Guardrail score={score}: {reason}")
    except Exception as e:
        logger.warning(f"Guardrail LLM failed: {e} — defaulting to 50")
        score, reason = 50, f"LLM error: {e}"

    return {
        "guardrail_score": score,
        "guardrail_reason": reason,
        "original_query": query,
    }


def continue_after_guardrail(state: AgentState) -> str:
    score = state.get("guardrail_score", 50)
    return "continue" if score >= 75 else "out_of_scope"
