import logging
import re
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    """Remove hyphenated line breaks, collapse newlines, strip extra whitespace."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_section(
    section_title: str,
    section_content: str,
    page_number: int,
    document_id: str,
    document_name: str,
    chunk_size: int,
    overlap: int,
    section_index: int,
    total_sections: int,
) -> List[Dict[str, Any]]:
    """
    Chunk a single section into overlapping word-based chunks with metadata.

    Returns a list of chunk dicts ready for OpenSearch indexing.
    """
    text = clean_text(section_content)
    tokens = text.split()
    step = chunk_size - overlap
    chunks = []
    chunk_index = 0

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_text = " ".join(tokens[start:end])
        if not chunk_text.strip():
            continue

        chunks.append({
            "text":          chunk_text,
            "section_title": section_title,
            "page_number":   page_number,
            "chunk_index":   chunk_index,
            "char_count":    len(chunk_text),
            "document_id":   document_id,
            "document_name": document_name,
            "section_index": section_index,
            "total_sections": total_sections,
        })
        chunk_index += 1

        if end >= len(tokens):
            break

    return chunks


def chunk_document(
    parsed_doc,
    document_id: str,
    document_name: str,
    chunk_size: int = 300,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Chunk all sections of a ParsedDocument into metadata-rich chunks.

    Args:
        parsed_doc   : ParsedDocument from DoclingPDFParser
        document_id  : UUID from PostgreSQL documents table
        document_name: filename
        chunk_size   : words per chunk
        overlap      : overlapping words between chunks

    Returns:
        List of chunk dicts with full metadata
    """
    all_chunks = []
    total_sections = len(parsed_doc.sections)

    for section_index, section in enumerate(parsed_doc.sections):
        section_chunks = chunk_section(
            section_title=section.title,
            section_content=section.content,
            page_number=section.page_number,
            document_id=document_id,
            document_name=document_name,
            chunk_size=chunk_size,
            overlap=overlap,
            section_index=section_index,
            total_sections=total_sections,
        )
        all_chunks.extend(section_chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["global_chunk_index"] = i
        chunk["total_chunks"] = len(all_chunks)

    logging.info(
        f"Chunked '{document_name}': {total_sections} sections → "
        f"{len(all_chunks)} chunks"
    )
    return all_chunks
