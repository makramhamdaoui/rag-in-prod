"""Full agentic RAG graph integration tests."""
import pytest

from src.services.agents.graph import build_agentic_rag_graph


def test_graph_compiles():
    graph = build_agentic_rag_graph()
    assert graph is not None


def test_graph_has_all_nodes():
    graph = build_agentic_rag_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__", "__end__",
        "guardrail", "out_of_scope", "retrieve",
        "grade_documents", "rewrite_query", "generate_answer",
    }
    assert expected.issubset(node_names)
