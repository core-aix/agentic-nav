from unittest.mock import patch, Mock
import pytest

from llm_agents.tools import get_all_tools
from llm_agents.tools.knowledge_graph import search_similar_papers, find_neighboring_papers, traverse_graph

class TestGetAllTools:
    """Test the get_all_tools function."""

    @patch('llm_agents.tools.knowledge_graph.search_similar_papers')
    @patch('llm_agents.tools.knowledge_graph.find_neighboring_papers')
    @patch('llm_agents.tools.knowledge_graph.traverse_graph')
    def test_get_all_tools_returns_list(self, mock_traverse, mock_neighbors, mock_search):
        """Test that get_all_tools returns a list of all tools."""
        # Mock the tool functions
        mock_search_func = Mock()
        mock_neighbors_func = Mock()
        mock_traverse_func = Mock()
        
        mock_search.return_value = mock_search_func
        mock_neighbors.return_value = mock_neighbors_func
        mock_traverse.return_value = mock_traverse_func

        # Call the function
        result = get_all_tools()

        # Verify return type is list
        assert isinstance(result, list)

    @patch('llm_agents.tools.knowledge_graph.search_similar_papers')
    @patch('llm_agents.tools.knowledge_graph.find_neighboring_papers')
    @patch('llm_agents.tools.knowledge_graph.traverse_graph')
    def test_get_all_tools_returns_correct_count(self, mock_traverse, mock_neighbors, mock_search):
        """Test that get_all_tools returns the correct number of tools."""
        # Call the function
        result = get_all_tools()

        # Verify we get exactly 3 tools
        assert len(result) == 3

    @patch('llm_agents.tools.knowledge_graph.search_similar_papers')
    @patch('llm_agents.tools.knowledge_graph.find_neighboring_papers')
    @patch('llm_agents.tools.knowledge_graph.traverse_graph')
    def test_get_all_tools_contains_all_expected_tools(self, mock_traverse, mock_neighbors, mock_search):
        """Test that get_all_tools contains all expected tool functions."""
        # Call the function
        result = get_all_tools()

        # Verify all expected tools are in the result
        assert search_similar_papers in result
        assert find_neighboring_papers in result
        assert traverse_graph in result

    @patch('llm_agents.tools.knowledge_graph.search_similar_papers')
    @patch('llm_agents.tools.knowledge_graph.find_neighboring_papers')
    @patch('llm_agents.tools.knowledge_graph.traverse_graph')
    def test_get_all_tools_order_matches_all_declaration(self, mock_traverse, mock_neighbors, mock_search):
        """Test that get_all_tools returns tools in the same order as __all__."""
        # Call the function
        result = get_all_tools()

        # Verify order matches __all__
        assert result[0] == search_similar_papers
        assert result[1] == find_neighboring_papers
        assert result[2] == traverse_graph

    def test_get_all_tools_no_duplicates(self):
        """Test that get_all_tools returns no duplicate tools."""
        # Call the function
        result = get_all_tools()

        # Verify no duplicates
        assert len(result) == len(set(result))

    def test_get_all_tools_all_callable(self):
        """Test that all returned tools are callable."""
        # Call the function
        result = get_all_tools()

        # Verify all items are callable
        for tool in result:
            assert callable(tool)
