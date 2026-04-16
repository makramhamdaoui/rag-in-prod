import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chat import generate_response_streaming
from src.config import get_settings
from src.ingestion import create_index, create_search_pipeline, bulk_index_chunks, delete_chunks_by_document_id
from src.opensearch import get_opensearch_client
from src.services.pdf_parser.docling_parser import DoclingPDFParser
from src.embeddings import generate_embeddings
from src.utils import chunk_document, setup_logging
from src.database.connection import create_tables, get_db
from src.models.document import Document, DocumentStatus

setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="RAG in Production", version="1.0.0")
PROJECT_ROOT = Path(__file__).parent

# in-memory session store
sessions: Dict[str, List[Dict[str, str]]] = {}


# ─── startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    create_tables()
    client = get_opensearch_client()
    create_index(client)
    create_search_pipeline(client)
    logger.info("Startup complete.")


# ─── health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": settings.ollama_model_name}


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

@app.get("/documents")
def list_documents():
    """List all indexed documents from PostgreSQL."""
    db = next(get_db())
    docs = db.query(Document).all()
    return {"documents": [d.to_dict() for d in docs]}


@app.post("/documents/ingest")
def ingest_document(file_path: str):
    """Ingest a PDF — extract, chunk, embed and index into OpenSearch + PostgreSQL."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    db = next(get_db())

    # check if already indexed
    existing = db.query(Document).filter(Document.name == path.name).first()
    if existing and existing.status == DocumentStatus.indexed:
        return {"message": f"Already indexed", "document": existing.to_dict()}

    # create DB record
    doc_record = Document(
        id=str(uuid.uuid4()),
        name=path.name,
        path=str(path),
        status=DocumentStatus.pending,
        file_size=path.stat().st_size,
    )
    db.add(doc_record)
    db.commit()

    try:
        # parse
        parser = DoclingPDFParser()
        parsed = parser.parse(str(path))
        if not parsed:
            raise ValueError("PDF parsing failed")

        # chunk with metadata
        chunks = chunk_document(
            parsed_doc=parsed,
            document_id=doc_record.id,
            document_name=path.name,
            chunk_size=settings.text_chunk_size,
            overlap=settings.text_chunk_overlap,
        )

        # embed
        embeddings = generate_embeddings([c["text"] for c in chunks])

        # index into OpenSearch
        success, errors = bulk_index_chunks(chunks, embeddings)

        # update DB record
        doc_record.status = DocumentStatus.indexed
        doc_record.num_chunks = success
        doc_record.num_pages = parsed.num_pages
        doc_record.indexed_at = datetime.now(timezone.utc)
        db.commit()

        return {
            "indexed": success,
            "errors": len(errors),
            "document": doc_record.to_dict(),
        }

    except Exception as e:
        doc_record.status = DocumentStatus.failed
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete document from OpenSearch and PostgreSQL."""
    db = next(get_db())
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_chunks_by_document_id(document_id)
    db.delete(doc)
    db.commit()
    return {"deleted": document_id}


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


# ─── cached chat ──────────────────────────────────────────────────────────────

@app.post("/chat/cached")
def chat_cached(request: ChatRequest):
    """
    Chat with semantic caching.
    - Exact match: same query → instant response
    - Semantic match: similar query (cosine > 0.92) → instant response
    - Miss: full RAG pipeline → store in cache
    """
    from src.services.cache.client import get_cached_response, store_cached_response
    from src.embeddings import get_embedding_model

    # embed query once — used for both semantic search and cache lookup
    model = get_embedding_model()
    query_embedding = model.encode(request.query).tolist()

    # check cache
    cached = get_cached_response(
        query=request.query,
        num_results=request.num_results,
        use_hybrid_search=request.use_hybrid_search,
        query_embedding=query_embedding,
    )
    if cached:
        return {
            "response": cached["response"],
            "cached": True,
            "match_type": cached["match_type"],
            "similarity": cached["similarity"],
        }

    # cache miss — run full RAG
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
    for chunk in stream:
        full_response += chunk["message"]["content"]

    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": full_response})

    # store with embedding for semantic matching
    store_cached_response(
        query=request.query,
        num_results=request.num_results,
        use_hybrid_search=request.use_hybrid_search,
        response=full_response,
        query_embedding=query_embedding,
    )

    return {
        "response": full_response,
        "cached": False,
        "session_id": session_id,
    }
