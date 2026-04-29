import logging
from typing import Dict

import ollama
from langchain_core.messages import AIMessage

from src.config import get_settings
from src.services.agents.prompts import RAG_PROMPT, OUT_OF_SCOPE_PROMPT
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def generate_node(state: AgentState) -> Dict:
    """Generate final answer from context."""
    logger.info("NODE: generate_answer")
    settings = get_settings()

    query = state.get("rewritten_query") or state.get("original_query", "")
    context = state.get("context", "")

    prompt = RAG_PROMPT.format(context=context, question=query)
    try:
        response = ollama.chat(
            model=settings.ollama_model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        answer = response["message"]["content"]
        logger.info(f"Generated answer ({len(answer)} chars)")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = f"I encountered an error generating the answer: {e}"

    return {"messages": [AIMessage(content=answer)]}


def out_of_scope_node(state: AgentState) -> Dict:
    """Return out of scope message."""
    logger.info("NODE: out_of_scope")
    score = state.get("guardrail_score", 0)
    reason = state.get("guardrail_reason", "")
    msg = f"{OUT_OF_SCOPE_PROMPT}\n\n(Relevance score: {score}/100 — {reason})"
    return {"messages": [AIMessage(content=msg)]}
