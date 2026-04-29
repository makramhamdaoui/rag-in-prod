import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.config import get_settings
from src.database.connection import get_db
from src.database.models.document import Document, DocumentStatus
from src.services.pdf_parser.docling_parser import DoclingPDFParser
from src.services.embeddings.client import generate_embeddings
from src.services.chunking.chunker import chunk_document
from src.services.search.ingestion import bulk_index_chunks, delete_chunks_by_document_id

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

PROJECT_ROOT = Path(__file__).parent.parent.parent


@router.get("/documents")
def list_documents():
    db = next(get_db())
    docs = db.query(Document).all()
    return {"documents": [d.to_dict() for d in docs]}


@router.post("/documents/ingest")
def ingest_document(file_path: str):
    settings = get_settings()
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    db = next(get_db())

    existing = db.query(Document).filter(Document.name == path.name).first()
    if existing and existing.status == DocumentStatus.indexed:
        return {"message": "Already indexed", "document": existing.to_dict()}

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
        parser = DoclingPDFParser()
        parsed = parser.parse(str(path))
        if not parsed:
            raise ValueError("PDF parsing failed")

        chunks = chunk_document(
            parsed_doc=parsed,
            document_id=doc_record.id,
            document_name=path.name,
            chunk_size=settings.text_chunk_size,
            overlap=settings.text_chunk_overlap,
        )
        embeddings = generate_embeddings([c["text"] for c in chunks])
        success, errors = bulk_index_chunks(chunks, embeddings)

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


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    db = next(get_db())
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_chunks_by_document_id(document_id)
    db.delete(doc)
    db.commit()
    return {"deleted": document_id}
