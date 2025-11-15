"""
Tests for the base LLMAgent class.
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
from datetime import datetime, UTC

from llm_agents.agents.base import LLMAgent


class TestLLMAgent:
    """Test the base LLMAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return LLMAgent(
            model="test-model",
            api_base="http://test.com",
            api_key="test-key",
            llm_args={"temperature": 0.5},
            global_tool_args={"num_records": 5}
        )

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        def mock_tool_1(query: str, num_results: int = 10):
            """Mock search tool."""
            return {"results": f"Found {num_results} results for '{query}'"}

        def mock_tool_2(paper_id: str):
            """Mock retrieval tool."""
            return {"paper": f"Details for paper {paper_id}"}

        return [mock_tool_1, mock_tool_2]

    def test_agent_initialization(self, agent):
        """Test agent initialization with default values."""
        assert agent.model == "test-model"
        assert agent.api_base == "http://test.com"
        assert agent.api_key == "test-key"
        assert agent.llm_args == {"temperature": 0.5}
        assert agent.global_tool_args == {"num_records": 5}
        assert agent.max_interaction_rounds == 10
        assert agent.messages == []

    def test_agent_default_initialization(self):
        """Test agent initialization with default values."""
        agent = LLMAgent()
        assert agent.model == "ollama_chat/gpt-oss:20b"
        assert agent.api_base == "http://localhost:11434"
        assert agent.api_key is None
        assert "temperature" in agent.llm_args
        assert "max_tokens" in agent.llm_args

    @patch('llm_agents.agents.base.litellm')
    def test_test_llm_connection_success(self, mock_litellm, agent):
        """Test successful LLM connection test."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_litellm.completion.return_value = mock_response

        agent.test_llm_connection()
        
        mock_litellm.completion.assert_called_once()
        call_args = mock_litellm.completion.call_args
        assert call_args.kwargs['model'] == "test-model"
        assert call_args.kwargs['api_base'] == "http://test.com"
        assert call_args.kwargs['api_key'] == "test-key"

    @patch('llm_agents.agents.base.litellm')
    def test_test_llm_connection_failure(self, mock_litellm, agent, caplog):
        """Test failed LLM connection test."""
        mock_litellm.completion.side_effect = Exception("Connection failed")

        agent.test_llm_connection()
        
        assert "Model not available or connection failed" in caplog.text

    def test_setup_session(self, agent, mock_tools):
        """Test session setup with tools."""
        agent.tools = mock_tools
        agent.setup_session()

        assert agent.tool_registry is not None
        assert len(agent.tool_registry) == 2
        assert "mock_tool_1" in agent.tool_registry
        assert "mock_tool_2" in agent.tool_registry
        assert agent.tool_descriptions is not None

    def test_setup_session_custom_tools(self, agent, mock_tools):
        """Test session setup with custom tool functions."""
        agent.setup_session(tool_funcs=mock_tools[:1])  # Only first tool

        assert len(agent.tool_registry) == 1
        assert "mock_tool_1" in agent.tool_registry
        assert "mock_tool_2" not in agent.tool_registry

    @patch('llm_agents.agents.base.litellm')
    def test_send_to_llm_text_response(self, mock_litellm, agent):
        """Test _send_to_llm with text-only response."""
        # Mock streaming response
        mock_chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world!"}}]},
            {"choices": [{"delta": {}}]}  # End marker
        ]
        mock_litellm.completion.return_value = iter(mock_chunks)

        messages = [{"role": "user", "content": "test"}]
        collected, calls = agent._send_to_llm(messages, "test-model", "http://test.com", "test-key")

        assert collected == "Hello world!"
        assert calls == []

    @patch('llm_agents.agents.base.litellm')
    def test_send_to_llm_with_tool_calls(self, mock_litellm, agent):
        """Test _send_to_llm with tool calls in response."""
        mock_tool_calls = [{
            "id": "call_1",
            "function": {"name": "search_papers", "arguments": "{}"}
        }]
        
        mock_chunks = [
            {"choices": [{"delta": {"content": "I'll search for papers."}}]},
            {"choices": [{"delta": {"tool_calls": mock_tool_calls}}]},
            {"choices": [{"delta": {}}]}
        ]
        mock_litellm.completion.return_value = iter(mock_chunks)

        messages = [{"role": "user", "content": "search papers"}]
        collected, calls = agent._send_to_llm(messages, "test-model", "http://test.com", "test-key")

        assert collected == "I'll search for papers."
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"

    def test_call_tool_success(self, agent, mock_tools):
        """Test successful tool execution."""
        agent.tools = mock_tools
        agent.setup_session()

        tool_call = {
            "id": "call_1",
            "function": {
                "name": "mock_tool_1",
                "arguments": '{"query": "test query", "num_results": 3}'
            }
        }

        result = agent.call_tool(tool_call)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_1"
        assert result["name"] == "mock_tool_1"
        
        content = json.loads(result["content"])
        assert "Found 3 results for 'test query'" in content["results"]

    def test_call_tool_invalid_json_args(self, agent, mock_tools):
        """Test tool call with invalid JSON arguments."""
        agent.tools = mock_tools
        agent.setup_session()

        tool_call = {
            "id": "call_1",
            "function": {
                "name": "mock_tool_1",
                "arguments": "invalid json"
            }
        }

        result = agent.call_tool(tool_call)

        # Should still work with empty args
        assert result["role"] == "tool"
        assert result["name"] == "mock_tool_1"

    def test_set_get_history(self, agent, sample_message):
        """Test setting and getting message history."""
        messages = [sample_message]
        agent.set_history(messages)
        
        assert agent.get_history() == messages

    def test_set_system_prompt(self, agent):
        """Test setting system prompt."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        new_system_prompt = "You are a helpful assistant."
        updated_messages = agent.set_system_prompt(new_system_prompt, messages)

        assert len(updated_messages) == 3
        assert updated_messages[0]["role"] == "system"
        assert updated_messages[0]["content"] == new_system_prompt
        assert updated_messages[1]["role"] == "user"
        assert updated_messages[2]["role"] == "assistant"

    def test_set_system_prompt_replaces_existing(self, agent):
        """Test that setting system prompt replaces existing system message."""
        messages = [
            {"role": "system", "content": "Old system prompt"},
            {"role": "user", "content": "Hello"}
        ]
        
        new_system_prompt = "New system prompt"
        updated_messages = agent.set_system_prompt(new_system_prompt, messages)

        # Should still have 2 messages, with system prompt replaced
        assert len(updated_messages) == 2
        assert updated_messages[0]["role"] == "system" 
        assert updated_messages[0]["content"] == new_system_prompt
        assert updated_messages[1]["role"] == "user"

    def test_get_system_prompt(self, agent):
        """Test getting system prompt from messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"}
        ]
        agent.set_history(messages)
        
        system_msg = agent.get_system_prompt()
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "System prompt"

    def test_get_system_prompt_none(self, agent):
        """Test getting system prompt when none exists."""
        messages = [{"role": "user", "content": "Hello"}]
        agent.set_history(messages)
        
        system_msg = agent.get_system_prompt()
        assert system_msg is None

    def test_get_most_recent_assistant_message(self, agent):
        """Test getting most recent assistant message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
        agent.set_history(messages)
        
        recent_msg = agent.get_most_recent_assistant_message()
        assert recent_msg["content"] == "I'm doing well!"

    def test_get_most_recent_assistant_message_none(self, agent):
        """Test getting most recent assistant message when none exists."""
        messages = [{"role": "user", "content": "Hello"}]
        agent.set_history(messages)
        
        recent_msg = agent.get_most_recent_assistant_message()
        assert recent_msg is None

    @patch('llm_agents.agents.base.litellm')
    def test_interact_single_round(self, mock_litellm, agent, mock_tools, sample_message):
        """Test single interaction round without tool calls."""
        agent.tools = mock_tools
        agent.setup_session()

        mock_chunks = [
            {"choices": [{"delta": {"content": "Hello there!"}}]},
            {"choices": [{"delta": {}}]}
        ]
        mock_litellm.completion.return_value = iter(mock_chunks)

        result_messages = agent.interact(sample_message)
        
        assert len(result_messages) == 2  # user + assistant
        assert result_messages[0] == sample_message
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[1]["content"] == "Hello there!"

    @patch('llm_agents.agents.base.litellm')
    def test_interact_with_tool_calls(self, mock_litellm, agent, mock_tools, sample_message):
        """Test interaction with tool calls."""
        agent.tools = mock_tools
        agent.setup_session()

        # First call - assistant makes tool call
        mock_tool_calls = [{
            "id": "call_1",
            "function": {"name": "mock_tool_1", "arguments": '{"query": "test"}'}
        }]
        
        first_chunks = [
            {"choices": [{"delta": {"content": "I'll search for that."}}]},
            {"choices": [{"delta": {"tool_calls": mock_tool_calls}}]},
            {"choices": [{"delta": {}}]}
        ]
        
        # Second call - assistant responds to tool result
        second_chunks = [
            {"choices": [{"delta": {"content": "Here are the results!"}}]},
            {"choices": [{"delta": {}}]}
        ]
        
        mock_litellm.completion.side_effect = [iter(first_chunks), iter(second_chunks)]

        result_messages = agent.interact(sample_message)
        
        # Should have: user message, assistant response with tool call, tool result, final assistant response
        assert len(result_messages) == 4
        assert result_messages[0] == sample_message
        assert result_messages[1]["role"] == "assistant"
        assert "tool_calls" in result_messages[1]
        assert result_messages[2]["role"] == "tool"
        assert result_messages[3]["role"] == "assistant"
        assert result_messages[3]["content"] == "Here are the results!"

    def test_interact_assertions(self, agent):
        """Test that interact method validates inputs properly."""
        with pytest.raises(AssertionError, match="Make sure to call 'setup_agent'"):
            agent.interact({"role": "user", "content": "test"})
        
        agent.tool_registry = {}
        agent.tool_descriptions = []
        
        with pytest.raises(AssertionError, match="must contain a 'role' key"):
            agent.interact({"content": "test"})
            
        with pytest.raises(AssertionError, match="must contain a 'content' key"):
            agent.interact({"role": "user"})

    @patch('llm_agents.agents.base.litellm')
    def test_interact_stateless(self, mock_litellm, agent, mock_tools):
        """Test stateless interaction generator."""
        agent.tools = mock_tools
        agent.setup_session()

        mock_chunks = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world!"}}]},
            {"choices": [{"delta": {}}]}
        ]
        mock_litellm.completion.return_value = iter(mock_chunks)

        messages = [{"role": "user", "content": "test"}]
        
        # Collect all yielded message states
        states = list(agent.interact_stateless(messages, "test-model", "http://test.com", "test-key"))
        
        # Should have multiple states as content streams in
        assert len(states) > 0
        final_state = states[-1]
        assert len(final_state) == 2  # user + assistant
        assert final_state[1]["role"] == "assistant"
        assert final_state[1]["content"] == "Hello world!"

    def test_interact_stateless_assertions(self, agent):
        """Test that interact_stateless validates setup properly."""
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(AssertionError, match="Make sure to call 'setup_agent'"):
            list(agent.interact_stateless(messages, "model", "api_base", "api_key"))

    @patch('llm_agents.agents.base.datetime')
    def test_message_timestamp_addition(self, mock_datetime, agent, mock_tools):
        """Test that messages get timestamps added automatically."""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_datetime.UTC = UTC
        
        agent.tools = mock_tools
        agent.setup_session()

        message = {"role": "user", "content": "test"}  # No timestamp
        
        with patch.object(agent, '_send_to_llm', return_value=("response", [])):
            agent.interact(message)
        
        # Message should have timestamp added
        assert "_ts" in agent.messages[0]
        assert agent.messages[0]["_ts"] == "2024-01-01 12:00:00+00:00"