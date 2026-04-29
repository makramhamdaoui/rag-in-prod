"""Shared dependencies and in-memory state."""
from typing import Dict, List

# in-memory session store
# in real production this would be Redis or PostgreSQL
sessions: Dict[str, List[Dict[str, str]]] = {}
