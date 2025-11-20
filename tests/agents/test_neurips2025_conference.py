"""
Tests for the NeurIPS2025Agent class.
"""
import pytest
from unittest.mock import patch

from llm_agents.agents.neurips2025_conference import NeurIPS2025Agent, DEFAULT_NEURIPS2025_AGENT_ARGS


class TestNeurIPS2025Agent:
    """Test the NeurIPS2025Agent class."""

    def test_default_args(self):
        """Test default agent arguments configuration."""
        args = DEFAULT_NEURIPS2025_AGENT_ARGS
        
        assert "model" in args
        assert "api_base" in args
        assert "api_key" in args
        assert "llm_args" in args
        assert "global_tool_args" in args
        
        # Check LLM args structure
        assert "temperature" in args["llm_args"]
        assert "max_tokens" in args["llm_args"]
        assert "num_ctx" in args["llm_args"]
        
        # Check tool args
        assert "max_num_papers" in args["global_tool_args"]

    def test_agent_initialization_default(self):
        """Test agent initialization with default configuration."""
        agent = NeurIPS2025Agent()
        
        # Should have system message pre-configured
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert "NeurIPS 2025 papers" in agent.messages[0]["content"]
        assert "search tool" in agent.messages[0]["content"]
        
        # Should have the right tools
        assert len(agent.tools) == 3
        tool_names = [tool.__name__ for tool in agent.tools]
        assert "search_similar_papers" in tool_names
        assert "find_neighboring_papers" in tool_names
        assert "traverse_graph" in tool_names

    def test_agent_initialization_custom_args(self):
        """Test agent initialization with custom arguments."""
        custom_args = {
            "model": "custom-model",
            "api_base": "http://custom.com",
            "api_key": "custom-key",
            "llm_args": {"temperature": 0.8},
            "global_tool_args": {"max_num_papers": 20}
        }
        
        agent = NeurIPS2025Agent(**custom_args)
        
        assert agent.model == "custom-model"
        assert agent.api_base == "http://custom.com"
        assert agent.api_key == "custom-key"
        assert agent.llm_args == {"temperature": 0.8}
        assert agent.global_tool_args == {"max_num_papers": 20}

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance."""
        agent = NeurIPS2025Agent()
        system_msg = agent.messages[0]["content"]
        
        # Check key instruction components
        assert "NeurIPS 2025 papers" in system_msg
        assert "search tool" in system_msg
        assert "paper titles and abstracts as input keywords" in system_msg
        assert "cite titles, abstracts, and OpenReview URLs" in system_msg

    def test_agent_inherits_base_functionality(self):
        """Test that agent properly inherits from LLMAgent."""
        agent = NeurIPS2025Agent()
        
        # Should inherit base methods
        assert hasattr(agent, 'setup_session')
        assert hasattr(agent, 'interact')
        assert hasattr(agent, 'interact_stateless')
        assert hasattr(agent, 'test_llm_connection')
        assert hasattr(agent, 'set_history')
        assert hasattr(agent, 'get_history')

    @patch('llm_agents.tools.search_similar_papers')
    @patch('llm_agents.tools.find_neighboring_papers')
    @patch('llm_agents.tools.traverse_graph')
    def test_tools_import(self, mock_traverse, mock_neighboring, mock_search):
        """Test that tools are properly imported and available."""
        agent = NeurIPS2025Agent()
        agent.setup_session()
        
        # Tools should be registered
        assert "search_similar_papers" in agent.tool_registry
        assert "find_neighboring_papers" in agent.tool_registry
        assert "traverse_graph" in agent.tool_registry

    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        import sys
        import importlib

        with patch.dict('os.environ', {
            'AGENT_MODEL_NAME': 'env-model',
            'AGENT_MODEL_API_BASE': 'http://env-base.com',
            'OLLAMA_API_KEY': 'env-key'
        }):
            # Remove from cache and reimport
            if 'llm_agents.agents.neurips2025_conference' in sys.modules:
                del sys.modules['llm_agents.agents.neurips2025_conference']

            from llm_agents.agents.neurips2025_conference import DEFAULT_NEURIPS2025_AGENT_ARGS

            assert DEFAULT_NEURIPS2025_AGENT_ARGS["model"] == "env-model"
            assert DEFAULT_NEURIPS2025_AGENT_ARGS["api_base"] == "http://env-base.com"
            assert DEFAULT_NEURIPS2025_AGENT_ARGS["api_key"] == "env-key"
