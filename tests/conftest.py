"""
Pytest configuration and shared fixtures for the llm_agents test suite.
"""
import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, UTC
from pathlib import Path


# Test environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(request):
    """Set up test environment variables."""

    # Skip auto environment setup for tests marked with no_auto_env
    if hasattr(request, 'node') and request.node.get_closest_marker('no_auto_env'):
        yield {}
        return

    test_env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "llm_agents",
        "EMBEDDING_MODEL_NAME": "nomic-embed-text",
        "EMBEDDING_MODEL_API_BASE": "http://localhost:11435",
        "AGENT_MODEL_NAME": "gpt-oss:20b",
        "AGENT_MODEL_API_BASE": "http://localhost:11436",
        "OLLAMA_API_KEY": "test-api-key",
        "AGENTIC_NAV_LOG_LEVEL": "DEBUG",
        "POPULATE_DATABASE_NIPS2025": "false"
    }
    
    # Temporarily set test environment variables
    old_env = {}
    for key, value in test_env.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def mock_datetime():
    """Mock datetime to return a fixed timestamp."""
    fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    with patch('llm_agents.agents.base.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.UTC = UTC
        yield mock_dt


@pytest.fixture
def sample_message():
    """Sample message for testing agent interactions."""
    return {
        "role": "user", 
        "content": "Test message",
        "_ts": "2024-01-01T12:00:00+00:00"
    }


@pytest.fixture
def sample_tool_call():
    """Sample tool call for testing."""
    return {
        "id": "test_call_id",
        "function": {
            "name": "search_similar_papers",
            "arguments": '{"query": "machine learning", "num_results": 5}'
        }
    }


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM response for testing."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].message.tool_calls = None
    return mock_response


@pytest.fixture
def mock_litellm_stream():
    """Mock LiteLLM streaming response."""
    chunks = [
        {"choices": [{"delta": {"content": "Test "}}]},
        {"choices": [{"delta": {"content": "response"}}]},
        {"choices": [{"delta": {}}]}  # End of stream
    ]
    return iter(chunks)


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for database tests."""
    with patch('neo4j.GraphDatabase.driver') as mock_driver:
        mock_session = Mock()
        mock_driver.return_value.session.return_value = mock_session
        yield mock_driver, mock_session


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for vector tests."""
    with patch('sentence_transformers.SentenceTransformer') as mock_model:
        mock_instance = Mock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3]]  # Sample embedding
        mock_model.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_paper_data():
    """Sample paper data for knowledge graph tests."""
    return [
        {
            "id": "paper1",
            "title": "Test Paper 1",
            "abstract": "This is a test abstract for paper 1",
            "authors": ["Author 1", "Author 2"],
            "keywords": ["machine learning", "AI"]
        },
        {
            "id": "paper2", 
            "title": "Test Paper 2",
            "abstract": "This is a test abstract for paper 2",
            "authors": ["Author 2", "Author 3"],
            "keywords": ["deep learning", "neural networks"]
        }
    ]


@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface for frontend tests."""
    with patch('gradio.Interface') as mock_interface:
        mock_instance = Mock()
        mock_interface.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_prompt_session():
    """Mock prompt_toolkit session for CLI tests."""
    with patch('prompt_toolkit.PromptSession') as mock_session:
        mock_instance = Mock()
        mock_instance.prompt.return_value = "test input"
        mock_session.return_value = mock_instance
        yield mock_instance
