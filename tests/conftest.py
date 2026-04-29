"""Shared pytest fixtures."""
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_query():
    return "What is the Transformer architecture?"


@pytest.fixture
def off_topic_query():
    return "What is the best pizza recipe?"


@pytest.fixture
def vague_query():
    return "how does it work?"
