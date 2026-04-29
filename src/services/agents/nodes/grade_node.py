import json
import logging
from typing import Dict

import ollama

from src.config import get_settings
from src.services.agents.prompts import GRADE_DOCUMENTS_PROMPT
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def grade_node(state: AgentState) -> Dict:
    """Grade retrieved chunks for relevance."""
    logger.info("NODE: grade_documents")
    settings = get_settings()

    query = state.get("rewritten_query") or state.get("original_query", "")
    context = state.get("context", "")
    attempts = state.get("retrieval_attempts", 0)

    if not context.strip():
        logger.warning("No context — routing to rewrite")
        return {"routing_decision": "rewrite_query", "grading_reasoning": "no context"}

    prompt = GRADE_DOCUMENTS_PROMPT.format(question=query, context=context[:2000])
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
        is_relevant = data.get("binary_score", "no") == "yes"
        reasoning = data.get("reasoning", "")
        logger.info(f"Grade: {'relevant' if is_relevant else 'not relevant'} — {reasoning}")
    except Exception as e:
        logger.warning(f"Grading LLM failed: {e} — defaulting to relevant")
        is_relevant = True
        reasoning = f"LLM error fallback: {e}"

    # route to rewrite only if not relevant AND attempts < max
    if not is_relevant and attempts < 2:
        route = "rewrite_query"
    else:
        route = "generate_answer"

    return {"routing_decision": route, "grading_reasoning": reasoning}
