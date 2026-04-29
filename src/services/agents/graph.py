import logging
import time
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.services.agents.state import AgentState
from src.services.agents.nodes import (
    guardrail_node,
    continue_after_guardrail,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    out_of_scope_node,
)

logger = logging.getLogger(__name__)


def build_agentic_rag_graph():
    """Build and compile the agentic RAG LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # add nodes
    workflow.add_node("guardrail",       guardrail_node)
    workflow.add_node("out_of_scope",    out_of_scope_node)
    workflow.add_node("retrieve",        retrieve_node)
    workflow.add_node("grade_documents", grade_node)
    workflow.add_node("rewrite_query",   rewrite_node)
    workflow.add_node("generate_answer", generate_node)

    # edges
    workflow.add_edge(START, "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        continue_after_guardrail,
        {"continue": "retrieve", "out_of_scope": "out_of_scope"},
    )

    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("routing_decision", "generate_answer"),
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def run_agentic_rag(query: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Run the full agentic RAG pipeline.

    Args:
        query: user question
        session_id: for tracking

    Returns:
        dict with answer, reasoning_steps, sources, metadata
    """
    graph = build_agentic_rag_graph()
    start_time = time.time()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "rewritten_query": None,
        "retrieval_attempts": 0,
        "guardrail_score": None,
        "guardrail_reason": None,
        "routing_decision": None,
        "retrieved_chunks": [],
        "grading_reasoning": None,
        "context": None,
    }

    result = graph.invoke(initial_state)
    execution_time = time.time() - start_time

    # extract final answer
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No answer generated."

    # build reasoning steps
    reasoning_steps = []
    score = result.get("guardrail_score")
    if score is not None:
        reasoning_steps.append(f"Guardrail: score={score}/100 — {result.get('guardrail_reason', '')}")

    attempts = result.get("retrieval_attempts", 0)
    if attempts:
        reasoning_steps.append(f"Retrieved documents ({attempts} attempt(s))")

    grading = result.get("grading_reasoning")
    if grading:
        reasoning_steps.append(f"Grading: {grading}")

    rewritten = result.get("rewritten_query")
    if rewritten and rewritten != query:
        reasoning_steps.append(f"Query rewritten: '{rewritten}'")

    reasoning_steps.append("Generated answer from context")

    # extract sources
    sources = [
        {
            "section_title": hit.get("_source", {}).get("section_title"),
            "page_number":   hit.get("_source", {}).get("page_number"),
            "document_name": hit.get("_source", {}).get("document_name"),
            "score":         hit.get("_score"),
        }
        for hit in result.get("retrieved_chunks", [])
    ]

    return {
        "query":            query,
        "answer":           answer,
        "reasoning_steps":  reasoning_steps,
        "sources":          sources,
        "rewritten_query":  rewritten,
        "retrieval_attempts": attempts,
        "guardrail_score":  score,
        "execution_time":   round(execution_time, 2),
    }
