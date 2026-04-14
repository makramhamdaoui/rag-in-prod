import logging
import re
from typing import List


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler(),
        ],
    )


def clean_text(text: str) -> str:
    """Remove hyphenated line breaks, collapse newlines, strip extra whitespace."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: input text
        chunk_size: number of words per chunk
        overlap: number of words shared between consecutive chunks

    Returns:
        list of text chunks
    """
    text = clean_text(text)
    tokens = text.split(" ")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
    logging.info(f"Text split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
