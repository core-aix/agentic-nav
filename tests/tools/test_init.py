"""
Tests for the tools __init__.py module.
"""
import pytest
from unittest.mock import patch, Mock

from llm_agents.tools import get_all_tools


class TestGetAllTools:
    """Test the get_all_tools function."""

    @patch('llm_agents.tools.search_similar_papers')
    @patch('llm_agents.tools.find_neighboring_papers') 
    @patch('llm_agents.tools.traverse_graph')
    def test_get_all_tools_returns_functions(self, mock_traverse, mock_neighbors, mock_search):
        """Test that get_all_tools returns the imported tool functions."""
        tools = get_all_tools()
        
        assert len(tools) == 3
        tool_names = [tool.__name__ for tool in tools]
        assert 'search_similar_papers' in tool_names
        assert 'find_neighboring_papers' in tool_names
        assert 'traverse_graph' in tool_names

    @patch('llm_agents.tools.search_similar_papers')
    def test_get_all_tools_filters_correctly(self, mock_search):
        """Test that get_all_tools only returns callable functions."""
        # Add some non-function attributes to the module
        tools = get_all_tools()
        
        # Should only contain actual functions, not modules or classes
        for tool in tools:
            assert callable(tool)
            assert not tool.__name__.startswith('_')
            assert tool.__name__ != 'get_all_tools'

    def test_get_all_tools_no_duplicates(self):
        """Test that get_all_tools doesn't return duplicate functions."""
        tools = get_all_tools()
        tool_names = [tool.__name__ for tool in tools]
        
        assert len(tool_names) == len(set(tool_names))  # No duplicates