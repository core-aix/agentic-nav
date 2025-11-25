"""
Tests for the file handler utility for knowledge graphs.
"""
import pytest
import tempfile
import networkx as nx
from pathlib import Path

from agentic_nav.tools.knowledge_graph.file_handler import save_graph, load_graph


class TestSaveGraph:
    """Test the save_graph function."""

    def test_save_graph_basic(self, capsys):
        """Test basic graph saving functionality."""
        # Create a simple graph
        graph = nx.Graph()
        graph.add_node("paper1", title="Test Paper 1")
        graph.add_node("paper2", title="Test Paper 2")
        graph.add_edge("paper1", "paper2", weight=0.85)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_graph.pkl"

            # Save the graph
            save_graph(graph, str(output_path))

            # Verify file was created
            assert output_path.exists()
            assert output_path.is_file()

            # Verify output message
            captured = capsys.readouterr()
            assert f"Graph saved to {output_path}" in captured.out

    def test_save_graph_with_complex_attributes(self):
        """Test saving graph with complex node and edge attributes."""
        graph = nx.Graph()
        graph.add_node("paper1",
                      title="Complex Paper",
                      authors=["Author A", "Author B"],
                      embedding=[0.1, 0.2, 0.3])
        graph.add_node("paper2",
                      title="Another Paper",
                      metadata={"year": 2024, "venue": "NeurIPS"})
        graph.add_edge("paper1", "paper2",
                      similarity=0.92,
                      relationship_type="SIMILAR_TO")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "complex_graph.pkl"

            # Save the graph
            save_graph(graph, str(output_path))

            # Verify file exists
            assert output_path.exists()

    def test_save_graph_empty_graph(self):
        """Test saving an empty graph."""
        graph = nx.Graph()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_graph.pkl"

            save_graph(graph, str(output_path))

            assert output_path.exists()


class TestLoadGraph:
    """Test the load_graph function."""

    def test_load_graph_basic(self, capsys):
        """Test basic graph loading functionality."""
        # Create and save a graph
        original_graph = nx.Graph()
        original_graph.add_node("paper1", title="Test Paper 1")
        original_graph.add_node("paper2", title="Test Paper 2")
        original_graph.add_edge("paper1", "paper2", weight=0.85)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_graph.pkl"
            save_graph(original_graph, str(file_path))

            # Load the graph
            loaded_graph = load_graph(str(file_path))

            # Verify the graph was loaded correctly
            assert isinstance(loaded_graph, nx.Graph)
            assert loaded_graph.number_of_nodes() == 2
            assert loaded_graph.number_of_edges() == 1
            assert "paper1" in loaded_graph.nodes()
            assert "paper2" in loaded_graph.nodes()
            assert loaded_graph.nodes["paper1"]["title"] == "Test Paper 1"
            assert loaded_graph.has_edge("paper1", "paper2")
            assert loaded_graph["paper1"]["paper2"]["weight"] == 0.85

            # Verify output message
            captured = capsys.readouterr()
            assert f"Graph loaded from {file_path}" in captured.out

    def test_load_graph_with_complex_attributes(self):
        """Test loading graph with complex attributes."""
        original_graph = nx.Graph()
        original_graph.add_node("paper1",
                               title="Complex Paper",
                               authors=["Author A", "Author B"],
                               embedding=[0.1, 0.2, 0.3])
        original_graph.add_edge("paper1", "paper2",
                               similarity=0.92,
                               metadata={"type": "citation"})

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "complex_graph.pkl"
            save_graph(original_graph, str(file_path))

            loaded_graph = load_graph(str(file_path))

            # Verify complex attributes are preserved
            assert loaded_graph.nodes["paper1"]["authors"] == ["Author A", "Author B"]
            assert loaded_graph.nodes["paper1"]["embedding"] == [0.1, 0.2, 0.3]
            assert loaded_graph["paper1"]["paper2"]["metadata"]["type"] == "citation"

    def test_load_graph_nonexistent_file(self):
        """Test loading from a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_graph("/nonexistent/path/graph.pkl")

    def test_save_load_roundtrip(self):
        """Test that saving and loading preserves graph structure."""
        original_graph = nx.Graph()
        original_graph.add_nodes_from([
            ("paper1", {"title": "Paper 1", "year": 2023}),
            ("paper2", {"title": "Paper 2", "year": 2024}),
            ("paper3", {"title": "Paper 3", "year": 2024}),
        ])
        original_graph.add_edges_from([
            ("paper1", "paper2", {"weight": 0.8}),
            ("paper2", "paper3", {"weight": 0.9}),
            ("paper1", "paper3", {"weight": 0.7}),
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "roundtrip_graph.pkl"

            # Save and load
            save_graph(original_graph, str(file_path))
            loaded_graph = load_graph(str(file_path))

            # Verify complete equality
            assert nx.is_isomorphic(original_graph, loaded_graph)
            assert loaded_graph.number_of_nodes() == original_graph.number_of_nodes()
            assert loaded_graph.number_of_edges() == original_graph.number_of_edges()

            # Verify all node attributes
            for node in original_graph.nodes():
                assert loaded_graph.nodes[node] == original_graph.nodes[node]

            # Verify all edge attributes
            for edge in original_graph.edges():
                assert loaded_graph.edges[edge] == original_graph.edges[edge]
