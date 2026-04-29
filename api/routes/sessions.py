import uuid
from fastapi import APIRouter, HTTPException

from api.schemas import SessionResponse
from api.dependencies import sessions

router = APIRouter(tags=["sessions"])


@router.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return SessionResponse(session_id=session_id)


@router.get("/session/{session_id}/history")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"deleted": session_id}
