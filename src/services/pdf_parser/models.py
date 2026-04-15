from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Section(BaseModel):
    """A section extracted from a PDF."""
    title: str = Field(..., description="Section title/header")
    content: str = Field(..., description="Section text content")
    page_number: int = Field(default=0, description="Page where section starts")
    level: int = Field(default=1, description="Header level (1=top, 2=sub, etc)")


class ParsedDocument(BaseModel):
    """Full structured output from PDF parsing."""
    sections: List[Section] = Field(default_factory=list)
    raw_text: str = Field(default="", description="Full raw text")
    num_pages: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
