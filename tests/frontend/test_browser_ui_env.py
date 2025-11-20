"""
Isolated test for browser UI environment variable usage.
This test is in a separate file to avoid interference from global mocks in other test files.
"""
import pytest
import sys
from unittest.mock import MagicMock


@pytest.mark.no_auto_env
def test_environment_variable_usage_isolated(monkeypatch):
    """Test that environment variables are used correctly in complete isolation."""
    
    # Clear only the specific browser_ui module to force fresh import
    # Don't clear parent packages to avoid breaking the import structure
    if 'llm_agents.frontend.browser_ui' in sys.modules:
        del sys.modules['llm_agents.frontend.browser_ui']

    monkeypatch.setenv('EMBEDDING_MODEL_NAME', 'test-embed-model')
    monkeypatch.setenv('EMBEDDING_MODEL_API_BASE', 'http://test-embed.com')
    monkeypatch.setenv('AGENT_MODEL_API_BASE', 'http://test-agent.com')
    monkeypatch.setenv('OLLAMA_API_KEY', 'test-key')

    import llm_agents.frontend.browser_ui as browser_ui
    
    # Verify these are NOT mocks
    assert not isinstance(browser_ui.EMBEDDING_MODEL_NAME, MagicMock), f"EMBEDDING_MODEL_NAME is a mock: {type(browser_ui.EMBEDDING_MODEL_NAME)}"
    assert not isinstance(browser_ui.EMBEDDING_MODEL_API_BASE, MagicMock), f"EMBEDDING_MODEL_API_BASE is a mock: {type(browser_ui.EMBEDDING_MODEL_API_BASE)}"
    assert not isinstance(browser_ui.AGENT_MODEL_API_BASE, MagicMock), f"AGENT_MODEL_API_BASE is a mock: {type(browser_ui.AGENT_MODEL_API_BASE)}"
    assert not isinstance(browser_ui.OLLAMA_API_KEY, MagicMock), f"OLLAMA_API_KEY is a mock: {type(browser_ui.OLLAMA_API_KEY)}"
    
    # Test that environment variables are correctly loaded
    assert browser_ui.EMBEDDING_MODEL_NAME == 'test-embed-model', f"Expected 'test-embed-model', got {browser_ui.EMBEDDING_MODEL_NAME}"
    assert browser_ui.EMBEDDING_MODEL_API_BASE == 'http://test-embed.com', f"Expected 'http://test-embed.com', got {browser_ui.EMBEDDING_MODEL_API_BASE}"
    assert browser_ui.AGENT_MODEL_API_BASE == 'http://test-agent.com', f"Expected 'http://test-agent.com', got {browser_ui.AGENT_MODEL_API_BASE}"
    assert browser_ui.OLLAMA_API_KEY == 'test-key', f"Expected 'test-key', got {browser_ui.OLLAMA_API_KEY}"
