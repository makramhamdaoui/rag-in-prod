import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chat import generate_response_streaming
from src.constants import OLLAMA_MODEL_NAME, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
from src.ingestion import create_index, create_search_pipeline, bulk_index_documents, delete_documents_by_name
from src.opensearch import get_opensearch_client
from src.ocr import extract_text_from_pdf
from src.embeddings import generate_embeddings
from src.utils import chunk_text, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG in Production", version="1.0.0")

# resolve project root relative to this file
PROJECT_ROOT = Path(__file__).parent

# in-memory session store
sessions: Dict[str, List[Dict[str, str]]] = {}


# ─── startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    client = get_opensearch_client()
    create_index(client)
    create_search_pipeline(client)
    logger.info("Startup complete.")


# ─── health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL_NAME}


# ─── sessions ─────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str


@app.post("/session", response_model=SessionResponse)
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return SessionResponse(session_id=session_id)


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"deleted": session_id}


# ─── documents ────────────────────────────────────────────────────────────────

@app.post("/documents/ingest")
def ingest_document(file_path: str):
    """
    Ingest a PDF into OpenSearch.
    Accepts absolute paths or paths relative to the project root.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    text = extract_text_from_pdf(str(path))
    chunks = chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=TEXT_CHUNK_OVERLAP)
    embeddings = generate_embeddings(chunks)
    document_name = path.name

    documents = [
        {
            "doc_id": str(uuid.uuid4()),
            "text": chunk,
            "embedding": emb,
            "document_name": document_name,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    success, errors = bulk_index_documents(documents)
    return {"indexed": success, "errors": len(errors), "document": document_name}


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    response = delete_documents_by_name(document_name)
    return {"deleted": response.get("deleted", 0), "document": document_name}


# ─── chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    use_hybrid_search: bool = True
    num_results: int = 3
    temperature: float = 0.3


@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    stream = generate_response_streaming(
        query=request.query,
        use_hybrid_search=request.use_hybrid_search,
        num_results=request.num_results,
        temperature=request.temperature,
        chat_history=history,
    )

    if stream is None:
        raise HTTPException(status_code=500, detail="Model failed to respond")

    full_response = ""

    def generate():
        nonlocal full_response
        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            yield token
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": full_response})

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )