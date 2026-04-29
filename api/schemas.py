"""Request/response schemas for all routes."""
from typing import Optional
from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    use_hybrid_search: bool = True
    num_results: int = 3
    temperature: float = 0.3


class AgenticRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
