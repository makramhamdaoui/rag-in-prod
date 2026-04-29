"""Agent state and prompt tests."""
from src.services.agents.state import AgentState
from src.services.agents.prompts import (
    GUARDRAIL_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    REWRITE_PROMPT,
    RAG_PROMPT,
)


def test_guardrail_prompt_has_placeholders():
    assert "{question}" in GUARDRAIL_PROMPT
    assert "{document_topics}" in GUARDRAIL_PROMPT


def test_grade_prompt_has_placeholders():
    assert "{question}" in GRADE_DOCUMENTS_PROMPT
    assert "{context}" in GRADE_DOCUMENTS_PROMPT


def test_rewrite_prompt_has_placeholders():
    assert "{question}" in REWRITE_PROMPT
    assert "{grading_reason}" in REWRITE_PROMPT
    assert "{document_topics}" in REWRITE_PROMPT


def test_rag_prompt_has_placeholders():
    assert "{context}" in RAG_PROMPT
    assert "{question}" in RAG_PROMPT


def test_guardrail_prompt_formats_correctly():
    result = GUARDRAIL_PROMPT.format(
        question="test query",
        document_topics="- doc1.pdf",
    )
    assert "test query" in result
    assert "doc1.pdf" in result
