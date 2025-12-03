"""
Tests for the Neo4jGraphWorker class.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from agentic_nav.tools.knowledge_graph.retriever import Neo4jGraphWorker


class TestNeo4jGraphWorker:
    """Test the Neo4jGraphWorker class."""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        # Properly mock the context manager behavior
        driver.session.return_value = MagicMock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver, session

    @pytest.fixture
    def worker(self, mock_driver):
        """Create worker instance with mocked driver."""
        driver, session = mock_driver
        with patch('agentic_nav.tools.knowledge_graph.retriever.GraphDatabase.driver', return_value=driver):
            worker = Neo4jGraphWorker(
                uri="bolt://localhost:7687",
                username="neo4j", 
                password="test_password"
            )
        return worker, session

    def test_initialization(self):
        """Test worker initialization."""
        with patch('agentic_nav.tools.knowledge_graph.retriever.GraphDatabase.driver') as mock_driver:
            worker = Neo4jGraphWorker(
                uri="bolt://test:7687",
                username="test_user",
                password="test_pass",
                max_connection_lifetime=1800,
                max_connection_pool_size=25,
                connection_acquisition_timeout=30
            )

            mock_driver.assert_called_once_with(
                "bolt://test:7687",
                auth=("test_user", "test_pass"),
                max_connection_lifetime=1800,
                max_connection_pool_size=25,
                connection_acquisition_timeout=30
            )

    @patch.object(Neo4jGraphWorker, 'embed_user_query')
    def test_similarity_search(self, mock_embed, worker):
        """Test similarity search functionality."""
        worker_instance, mock_session = worker

        # Mock embedding generation
        mock_embed.return_value = [0.1, 0.2, 0.3]

        # Mock database query results - authors are now a list of strings
        def create_mock_record(id, name, abstract, topic, score, paper_url, decisions,
                               session, session_start_time, session_end_time,
                               presentation_type, presentation_category, room_name,
                               project_url, poster_position, sourceid, virtualsite_url, authors):
            """Helper to create a mock record with dict-like access."""
            record_data = {
                'id': id, 'name': name, 'abstract': abstract, 'topic': topic,
                'score': score, 'paper_url': paper_url, 'decisions': decisions,
                'session': session, 'session_start_time': session_start_time,
                'session_end_time': session_end_time, 'presentation_type': presentation_type,
                'presentation_category': presentation_category, 'room_name': room_name,
                'project_url': project_url, 'poster_position': poster_position,
                'sourceid': sourceid, 'virtualsite_url': virtualsite_url, 'authors': authors
            }
            record = Mock()
            record.__getitem__ = lambda self, key: record_data[key]
            return record

        mock_records = [
            create_mock_record('paper1', 'Test Paper 1', 'Test abstract 1', 'ML', 0.95,
                               'http://example.com/1', 'Accept', 'S1', '09:00', '10:00',
                               'Oral', 'Main', 'Room A', 'http://proj1.com', 'A1',
                               -1, 'http://virtual1.com', ['Author A', 'Author B']),
            create_mock_record('paper2', 'Test Paper 2', 'Test abstract 2', 'AI', 0.90,
                               'http://example.com/2', 'Accept', 'S2', '10:00', '11:00',
                               'Poster', 'Main', 'Room B', 'http://proj2.com', 'B1',
                               1, 'http://virtual2.com', ['Author C'])
        ]
        mock_session.run.return_value = mock_records

        # Call similarity search
        results = worker_instance.similarity_search(
            user_query="machine learning",
            day=None,
            timeslots=None,
            top_k=5,
            min_similarity=0.8
        )

        # Verify embedding generation was called
        mock_embed.assert_called_once_with("machine learning")

        # Verify database query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "db.index.vector.queryNodes" in call_args[0][0]
        assert call_args[1]['top_k'] == 5
        assert call_args[1]['query_embedding'] == [0.1, 0.2, 0.3]

        # Verify results filtering by min_similarity
        assert len(results) == 2
        assert results[0]['id'] == 'paper1'
        assert results[0]['authors'] == ['Author A', 'Author B']
        # project_url is mapped to github_url in _build_paper_dict
        assert results[0]['github_url'] == 'http://proj1.com'

    @patch.object(Neo4jGraphWorker, 'embed_user_query')
    def test_similarity_search_no_min_similarity(self, mock_embed, worker):
        """Test similarity search without minimum similarity filtering."""
        worker_instance, mock_session = worker

        mock_embed.return_value = [0.1, 0.2, 0.3]

        def create_mock_record(id, name, abstract, topic, score, paper_url, decisions,
                               session, session_start_time, session_end_time,
                               presentation_type, presentation_category, room_name,
                               project_url, poster_position, sourceid, virtualsite_url, authors):
            """Helper to create a mock record with dict-like access."""
            record_data = {
                'id': id, 'name': name, 'abstract': abstract, 'topic': topic,
                'score': score, 'paper_url': paper_url, 'decisions': decisions,
                'session': session, 'session_start_time': session_start_time,
                'session_end_time': session_end_time, 'presentation_type': presentation_type,
                'presentation_category': presentation_category, 'room_name': room_name,
                'project_url': project_url, 'poster_position': poster_position,
                'sourceid': sourceid, 'virtualsite_url': virtualsite_url, 'authors': authors
            }
            record = Mock()
            record.__getitem__ = lambda self, key: record_data[key]
            return record

        mock_records = [
            create_mock_record('paper1', 'Test', 'Test', 'ML', 0.5,
                               'http://example.com/1', 'Accept', 'S1', '09:00', '10:00',
                               'Oral', 'Main', 'Room A', 'http://proj1.com', 'A1',
                               -1, 'http://virtual1.com', ['Author A']),
            create_mock_record('paper2', 'Test', 'Test', 'AI', 0.3,
                               'http://example.com/2', 'Accept', 'S2', '10:00', '11:00',
                               'Poster', 'Main', 'Room B', 'http://proj2.com', 'B1',
                               2, 'http://virtual2.com', ['Author B'])
        ]
        mock_session.run.return_value = mock_records

        results = worker_instance.similarity_search(
            user_query="test",
            day=None,
            timeslots=None,
            top_k=10,
            min_similarity=None
        )

        # Should return all results when no min_similarity filter
        assert len(results) == 2

    def test_neighborhood_search(self, worker):
        """Test neighborhood search functionality."""
        worker_instance, mock_session = worker

        def create_mock_record(id, name, abstract, topic, paper_url, decisions,
                               session, session_start_time, session_end_time,
                               presentation_type, presentation_category, room_name,
                               project_url, poster_position, sourceid, virtualsite_url,
                               authors, source_paper_id, relationship_type, similarity=None):
            """Helper to create a mock record with dict-like access."""
            record_data = {
                'id': id, 'name': name, 'abstract': abstract, 'topic': topic,
                'paper_url': paper_url, 'decisions': decisions,
                'session': session, 'session_start_time': session_start_time,
                'session_end_time': session_end_time, 'presentation_type': presentation_type,
                'presentation_category': presentation_category, 'room_name': room_name,
                'project_url': project_url, 'poster_position': poster_position,
                'sourceid': sourceid, 'virtualsite_url': virtualsite_url,
                'authors': authors, 'source_paper_id': source_paper_id,
                'relationship_type': relationship_type, 'similarity': similarity
            }
            record = Mock()
            record.__getitem__ = lambda self, key: record_data[key]
            return record

        mock_records = [
            create_mock_record('paper2', 'Neighbor Paper 1', 'Test abstract 1', 'ML',
                               'http://example.com/1', 'Accept', 'S1', '09:00', '10:00',
                               'Oral', 'Main', 'Room A', 'http://proj1.com', 'A1',
                               -2, 'http://virtual1.com', ['Author A'], 'paper1',
                               'SIMILAR_TO', 0.85),
            create_mock_record('paper3', 'Neighbor Paper 2', 'Test abstract 2', 'AI',
                               'http://example.com/2', 'Accept', 'S2', '10:00', '11:00',
                               'Poster', 'Main', 'Room B', 'http://proj2.com', 'B1',
                               3, 'http://virtual2.com', ['Author B'], 'paper1',
                               'SIMILAR_TO', 0.90)
        ]
        mock_session.run.return_value = mock_records

        results = worker_instance.neighborhood_search(
            paper_id="paper1",
            relationship_types=["SIMILAR_TO"],
            min_similarity=0.7
        )

        # Verify query was constructed and executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "MATCH (p:Paper {id: $paper_id})" in query
        assert "type(r) IN $allowed_rel_types" in query
        assert call_args[1]['paper_id'] == "paper1"
        assert call_args[1]['allowed_rel_types'] == ["SIMILAR_TO"]

        # Verify results structure - keys are relationship types
        assert 'SIMILAR_TO' in results
        assert len(results['SIMILAR_TO']) == 2
        assert results['SIMILAR_TO'][0]['id'] == 'paper2'
        assert results['SIMILAR_TO'][1]['id'] == 'paper3'

    def test_neighborhood_search_relationship_filter(self, worker):
        """Test neighborhood search with relationship type filtering.""" 
        worker_instance, mock_session = worker
        
        mock_session.run.return_value = []
        
        worker_instance.neighborhood_search(
            paper_id="paper1",
            relationship_types=["SIMILAR_TO"]
        )
        
        call_args = mock_session.run.call_args
        query = call_args[0][0]

        # Should include relationship type filter in WHERE clause
        assert "type(r) IN $allowed_rel_types" in query
        assert call_args[1]['allowed_rel_types'] == ["SIMILAR_TO"]

    @patch('agentic_nav.tools.knowledge_graph.retriever._graph_traversal_bfs_random')
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
        
        # Verify strategy function was called (it uses driver, not session)
        mock_bfs.assert_called_once_with(
            worker_instance.driver,
            "start_paper", 
            2,
            "SIMILAR_TO",
            30,
            3
        )
        
        assert results == mock_papers

    @patch('agentic_nav.tools.knowledge_graph.retriever._graph_traversal_cypher')
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

        def create_mock_record(id, name, abstract, topic, paper_url, decisions,
                               session, session_start_time, session_end_time,
                               presentation_type, presentation_category, room_name,
                               project_url, poster_position, sourceid, virtualsite_url, authors):
            """Helper to create a mock record with dict-like access."""
            record_data = {
                'id': id, 'name': name, 'abstract': abstract, 'topic': topic,
                'paper_url': paper_url, 'decisions': decisions,
                'session': session, 'session_start_time': session_start_time,
                'session_end_time': session_end_time, 'presentation_type': presentation_type,
                'presentation_category': presentation_category, 'room_name': room_name,
                'project_url': project_url, 'poster_position': poster_position,
                'sourceid': sourceid, 'virtualsite_url': virtualsite_url, 'authors': authors
            }
            record = Mock()
            record.__getitem__ = lambda self, key: record_data[key]
            return record

        mock_records = [
            create_mock_record('paper1', 'Paper by Author', 'Abstract', 'ML',
                               'http://example.com/1', 'Accept', 'S1', '09:00', '10:00',
                               'Oral', 'Main', 'Room A', 'http://proj1.com', 'A1',
                               -1, 'http://virtual1.com', ['Test Author'])
        ]
        mock_session.run.return_value = mock_records

        results = worker_instance.search_papers_by_author("Test Author", fuzzy=False)

        # Verify exact match query was used
        call_args = mock_session.run.call_args
        assert "a.fullname" in call_args[0][0] or "Author" in call_args[0][0]
        assert call_args[1]['author_name'] == "Test Author"

        assert len(results) == 1
        assert results[0]['authors'] == ['Test Author']
        assert results[0]['github_url'] == 'http://proj1.com'

    def test_papers_by_author_fuzzy(self, worker):
        """Test fuzzy papers by author search."""
        worker_instance, mock_session = worker
        
        mock_session.run.return_value = []
        
        worker_instance.search_papers_by_author("Test Author", fuzzy=True)
        
        # Verify fuzzy query was used
        call_args = mock_session.run.call_args
        assert "CONTAINS" in call_args[0][0]
        assert "toLower" in call_args[0][0]

    def test_papers_by_topic(self, worker):
        """Test papers by topic search."""
        worker_instance, mock_session = worker

        def create_mock_record(id, name, abstract, topic, paper_url, decisions,
                               session, session_start_time, session_end_time,
                               presentation_type, presentation_category, room_name,
                               project_url, poster_position, sourceid, virtualsite_url, authors):
            """Helper to create a mock record with dict-like access."""
            record_data = {
                'id': id, 'name': name, 'abstract': abstract, 'topic': topic,
                'paper_url': paper_url, 'decisions': decisions,
                'session': session, 'session_start_time': session_start_time,
                'session_end_time': session_end_time, 'presentation_type': presentation_type,
                'presentation_category': presentation_category, 'room_name': room_name,
                'project_url': project_url, 'poster_position': poster_position,
                'sourceid': sourceid, 'virtualsite_url': virtualsite_url, 'authors': authors
            }
            record = Mock()
            record.__getitem__ = lambda self, key: record_data[key]
            return record

        mock_records = [
            create_mock_record('paper1', 'Topic Paper', 'Abstract', 'Machine Learning',
                               'http://example.com/1', 'Accept', 'S1', '09:00', '10:00',
                               'Oral', 'Main', 'Room A', 'http://proj1.com', 'A1',
                               -1, 'http://virtual1.com', ['Author A'])
        ]
        mock_session.run.return_value = mock_records

        results = worker_instance.search_papers_by_topic("Machine Learning")

        call_args = mock_session.run.call_args
        assert "t:Topic {name: $topic_name}" in call_args[0][0]
        assert call_args[1]['topic_name'] == "Machine Learning"

        assert len(results) == 1
        assert results[0]['topic'] == 'Machine Learning'
        assert results[0]['github_url'] == 'http://proj1.com'

    def test_papers_by_topic_with_subtopics(self, worker):
        """Test papers by topic including subtopics."""
        worker_instance, mock_session = worker
        
        mock_session.run.return_value = []
        
        worker_instance.search_papers_by_topic("ML", include_subtopics=True)
        
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        assert "SUBTOPIC_OF" in query
        assert "collect(DISTINCT subtopic)" in query

    # Commented out - method find_similar_papers_direct no longer exists
    # def test_get_similar_papers(self, worker):
    #     """Test get similar papers functionality."""
    #     worker_instance, mock_session = worker
    #
    #     mock_records = [
    #         Mock(id='similar1', name='Similar Paper', abstract='Abstract', topic='ML', similarity=0.92)
    #     ]
    #     # Configure record access as dict-like
    #     mock_records[0].__getitem__ = lambda self, key: {
    #         'id': 'similar1',
    #         'name': 'Similar Paper',
    #         'abstract': 'Abstract',
    #         'topic': 'ML',
    #         'similarity': 0.92
    #     }[key]
    #     mock_session.run.return_value = mock_records
    #
    #     results = worker_instance.find_similar_papers_direct("paper1", min_similarity=0.8)
    #
    #     call_args = mock_session.run.call_args
    #     assert "SIMILAR_TO" in call_args[0][0]
    #     assert call_args[1]['paper_id'] == "paper1"
    #     assert call_args[1]['min_similarity'] == 0.8
    #
    #     assert len(results) == 1
    #     assert results[0]['similarity'] == 0.92

    def test_close(self, worker):
        """Test worker close method."""
        worker_instance, mock_session = worker
        
        # Mock the driver
        mock_driver = Mock()
        worker_instance.driver = mock_driver
        
        worker_instance.close()
        
        mock_driver.close.assert_called_once()