"""
Tests for the Neo4jGraphWorker class.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from llm_agents.tools.knowledge_graph.retriever import Neo4jGraphWorker


class TestNeo4jGraphWorker:
    """Test the Neo4jGraphWorker class."""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__.return_value = session
        return driver, session

    @pytest.fixture
    def worker(self, mock_driver):
        """Create worker instance with mocked driver."""
        driver, session = mock_driver
        with patch('llm_agents.tools.knowledge_graph.retriever.GraphDatabase.driver', return_value=driver):
            worker = Neo4jGraphWorker(
                uri="bolt://localhost:7687",
                username="neo4j", 
                password="test_password"
            )
        return worker, session

    def test_initialization(self):
        """Test worker initialization."""
        with patch('llm_agents.tools.knowledge_graph.retriever.GraphDatabase.driver') as mock_driver:
            worker = Neo4jGraphWorker(
                uri="bolt://test:7687",
                username="test_user",
                password="test_pass"
            )
            
            mock_driver.assert_called_once_with(
                "bolt://test:7687",
                auth=("test_user", "test_pass")
            )

    @patch('llm_agents.tools.knowledge_graph.retriever.batch_embed_documents')
    def test_similarity_search(self, mock_embed, worker):
        """Test similarity search functionality."""
        worker_instance, mock_session = worker
        
        # Mock embedding generation
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock database query results
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'id': 'paper1',
                'name': 'Test Paper 1', 
                'abstract': 'Test abstract 1',
                'topic': 'ML',
                'score': 0.95
            },
            {
                'id': 'paper2',
                'name': 'Test Paper 2',
                'abstract': 'Test abstract 2', 
                'topic': 'AI',
                'score': 0.90
            }
        ]
        mock_session.run.return_value = mock_result
        
        # Call similarity search
        results = worker_instance.similarity_search(
            user_query="machine learning",
            top_k=5,
            min_similarity=0.8
        )
        
        # Verify embedding generation was called
        mock_embed.assert_called_once_with(
            documents=["machine learning"],
            model="test-embed-model",  # From test env
            ollama_server_url="http://localhost:11435"
        )
        
        # Verify database query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "db.index.vector.queryNodes" in call_args[0][0]
        assert call_args[1]['top_k'] == 5
        assert np.array_equal(call_args[1]['query_embedding'], [0.1, 0.2, 0.3])
        
        # Verify results filtering by min_similarity
        assert len(results) == 2
        assert results[0]['id'] == 'paper1'
        assert results[0]['score'] == 0.95

    @patch('llm_agents.tools.knowledge_graph.retriever.batch_embed_documents')
    def test_similarity_search_no_min_similarity(self, mock_embed, worker):
        """Test similarity search without minimum similarity filtering."""
        worker_instance, mock_session = worker
        
        mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
        
        mock_result = Mock()
        mock_result.data.return_value = [
            {'id': 'paper1', 'name': 'Test', 'abstract': 'Test', 'topic': 'ML', 'score': 0.5},
            {'id': 'paper2', 'name': 'Test', 'abstract': 'Test', 'topic': 'AI', 'score': 0.3}
        ]
        mock_session.run.return_value = mock_result
        
        results = worker_instance.similarity_search(
            user_query="test",
            top_k=10,
            min_similarity=None
        )
        
        # Should return all results when no min_similarity filter
        assert len(results) == 2

    def test_neighborhood_search(self, worker):
        """Test neighborhood search functionality."""
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'source_paper_id': 'paper1',
                'neighbor': {'id': 'paper2', 'name': 'Neighbor Paper'},
                'relationship_type': 'SIMILAR_TO',
                'relationship_properties': {'similarity': 0.85},
                'neighbor_labels': ['Paper']
            },
            {
                'source_paper_id': 'paper1', 
                'neighbor': {'fullname': 'Author Name'},
                'relationship_type': 'AUTHORED_BY',
                'relationship_properties': {},
                'neighbor_labels': ['Author']
            }
        ]
        mock_session.run.return_value = mock_result
        
        results = worker_instance.neighborhood_search(
            paper_id="paper1",
            relationship_types=["SIMILAR_TO", "AUTHORED_BY"]
        )
        
        # Verify query was constructed and executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "MATCH (p:Paper)" in query
        assert "WHERE p.id IN $paper_ids" in query
        assert call_args[1]['paper_ids'] == ["paper1"]
        
        # Verify results structure
        assert 'similar_papers' in results
        assert 'authors' in results
        assert len(results['similar_papers']) == 1
        assert len(results['authors']) == 1

    def test_neighborhood_search_relationship_filter(self, worker):
        """Test neighborhood search with relationship type filtering.""" 
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        
        worker_instance.neighborhood_search(
            paper_id="paper1",
            relationship_types=["SIMILAR_TO"]
        )
        
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        
        # Should include relationship type filter
        assert ":SIMILAR_TO" in query

    @patch('llm_agents.tools.knowledge_graph.retriever._graph_traversal_bfs_random')
    def test_graph_traversal_bfs_random(self, mock_bfs, worker):
        """Test graph traversal with breadth-first random strategy."""
        worker_instance, mock_session = worker
        
        mock_papers = [{'id': 'paper1', 'name': 'Traversed Paper'}]
        mock_bfs.return_value = mock_papers
        
        results = worker_instance.graph_traversal(
            start_paper_id="start_paper",
            n_hops=2,
            relationship_type="SIMILAR_TO",
            max_results=30,
            strategy="breadth_first_random",
            max_branches=3,
            random_seed=42
        )
        
        # Verify strategy function was called
        mock_bfs.assert_called_once_with(
            session=mock_session,
            start_paper_id="start_paper", 
            n_hops=2,
            relationship_type="SIMILAR_TO",
            max_results=30,
            max_branches=3,
            random_seed=42
        )
        
        assert results == mock_papers

    @patch('llm_agents.tools.knowledge_graph.retriever._graph_traversal_cypher')
    def test_graph_traversal_breadth_first(self, mock_cypher, worker):
        """Test graph traversal with breadth-first strategy."""
        worker_instance, mock_session = worker
        
        mock_papers = [{'id': 'paper1'}]
        mock_cypher.return_value = mock_papers
        
        results = worker_instance.graph_traversal(
            start_paper_id="start",
            strategy="breadth_first"
        )
        
        mock_cypher.assert_called_once()
        assert results == mock_papers

    def test_graph_traversal_invalid_strategy(self, worker):
        """Test graph traversal with invalid strategy raises error."""
        worker_instance, mock_session = worker
        
        with pytest.raises(ValueError, match="Unsupported traversal strategy"):
            worker_instance.graph_traversal(
                start_paper_id="start",
                strategy="invalid_strategy"
            )

    def test_papers_by_author(self, worker):
        """Test papers by author search."""
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'id': 'paper1',
                'name': 'Paper by Author',
                'abstract': 'Abstract',
                'topic': 'ML',
                'author_name': 'Test Author'
            }
        ]
        mock_session.run.return_value = mock_result
        
        results = worker_instance.papers_by_author("Test Author")
        
        # Verify exact match query was used
        call_args = mock_session.run.call_args
        assert "a.fullname = $author_name" in call_args[0][0]
        assert call_args[1]['author_name'] == "Test Author"
        
        assert len(results) == 1
        assert results[0]['author_name'] == 'Test Author'

    def test_papers_by_author_fuzzy(self, worker):
        """Test fuzzy papers by author search."""
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        
        worker_instance.papers_by_author("Test Author", fuzzy=True)
        
        # Verify fuzzy query was used
        call_args = mock_session.run.call_args
        assert "CONTAINS" in call_args[0][0]
        assert "toLower" in call_args[0][0]

    def test_papers_by_topic(self, worker):
        """Test papers by topic search.""" 
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'id': 'paper1',
                'name': 'Topic Paper',
                'abstract': 'Abstract',
                'topic': 'Machine Learning'
            }
        ]
        mock_session.run.return_value = mock_result
        
        results = worker_instance.papers_by_topic("Machine Learning")
        
        call_args = mock_session.run.call_args
        assert "t:Topic {name: $topic_name}" in call_args[0][0]
        assert call_args[1]['topic_name'] == "Machine Learning"
        
        assert len(results) == 1

    def test_papers_by_topic_with_subtopics(self, worker):
        """Test papers by topic including subtopics."""
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        
        worker_instance.papers_by_topic("ML", include_subtopics=True)
        
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "SUBTOPIC_OF" in query
        assert "collect(DISTINCT subtopic)" in query

    def test_get_similar_papers(self, worker):
        """Test get similar papers functionality."""
        worker_instance, mock_session = worker
        
        mock_result = Mock()
        mock_result.data.return_value = [
            {
                'id': 'similar1',
                'name': 'Similar Paper', 
                'abstract': 'Abstract',
                'topic': 'ML',
                'similarity': 0.92
            }
        ]
        mock_session.run.return_value = mock_result
        
        results = worker_instance.get_similar_papers("paper1", min_similarity=0.8)
        
        call_args = mock_session.run.call_args
        assert "SIMILAR_TO" in call_args[0][0]
        assert call_args[1]['paper_id'] == "paper1"
        assert call_args[1]['min_similarity'] == 0.8
        
        assert len(results) == 1
        assert results[0]['similarity'] == 0.92

    def test_close(self, worker):
        """Test worker close method."""
        worker_instance, mock_session = worker
        
        # Mock the driver
        mock_driver = Mock()
        worker_instance.driver = mock_driver
        
        worker_instance.close()
        
        mock_driver.close.assert_called_once()