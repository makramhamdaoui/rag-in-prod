from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:            Annotated[list[AnyMessage], add_messages]
    original_query:      Optional[str]
    rewritten_query:     Optional[str]
    retrieval_attempts:  int
    guardrail_score:     Optional[int]
    guardrail_reason:    Optional[str]
    routing_decision:    Optional[str]
    retrieved_chunks:    List[Dict[str, Any]]
    grading_reasoning:   Optional[str]
    context:             Optional[str]
