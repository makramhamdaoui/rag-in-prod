import uuid
from fastapi import APIRouter

from src.services.agents.graph import run_agentic_rag
from api.schemas import AgenticRequest

router = APIRouter(tags=["agentic"])


@router.post("/agentic")
def agentic_chat(request: AgenticRequest):
    """Full agentic RAG pipeline: guardrail → retrieve → grade → rewrite (if needed) → generate."""
    session_id = request.session_id or str(uuid.uuid4())
    result = run_agentic_rag(query=request.query, session_id=session_id)
    return result
