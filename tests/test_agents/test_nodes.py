"""Agentic RAG node tests."""
from langchain_core.messages import HumanMessage

from src.services.agents.nodes import (
    get_latest_query,
    continue_after_guardrail,
)


def test_get_latest_query_empty_state():
    state = {"messages": []}
    assert get_latest_query(state) == ""


def test_get_latest_query_from_human_message():
    state = {"messages": [HumanMessage(content="hello")]}
    assert get_latest_query(state) == "hello"


def test_get_latest_query_returns_most_recent_human():
    state = {
        "messages": [
            HumanMessage(content="first query"),
            HumanMessage(content="rewritten query"),
        ]
    }
    assert get_latest_query(state) == "rewritten query"


def test_continue_after_guardrail_high_score():
    assert continue_after_guardrail({"guardrail_score": 95}) == "continue"


def test_continue_after_guardrail_low_score():
    assert continue_after_guardrail({"guardrail_score": 30}) == "out_of_scope"


def test_continue_after_guardrail_threshold_boundary():
    assert continue_after_guardrail({"guardrail_score": 75}) == "continue"
    assert continue_after_guardrail({"guardrail_score": 74}) == "out_of_scope"


def test_continue_after_guardrail_missing_score():
    # defaults to 50 → out_of_scope
    assert continue_after_guardrail({}) == "out_of_scope"
