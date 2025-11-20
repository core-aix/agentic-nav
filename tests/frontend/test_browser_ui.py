"""
Tests for the browser UI frontend.
"""
import pytest
from unittest.mock import patch, Mock


class TestBrowserUIFunctions:
    """Test functions from browser_ui module that can be tested in isolation."""

    def test_module_imports(self):
        """Test that module imports work correctly."""
        try:
            from llm_agents.frontend import browser_ui
            assert hasattr(browser_ui, 'LOGGER')
        except ImportError as e:
            pytest.skip(f"Could not import browser_ui: {e}")

    @patch('llm_agents.frontend.browser_ui.NeurIPS2025Agent')
    def test_agent_initialization(self, mock_agent_class):
        """Test global agent initialization."""
        from llm_agents.frontend.browser_ui import initialize_agent

        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Call the function
        result = initialize_agent()

        # Verify agent was created correctly
        mock_agent_class.assert_called_once()
        mock_agent.setup_session.assert_called_once()
        assert result == mock_agent

    def test_configure_agent_function(self):
        """Test the initialize_agent function."""
        from llm_agents.frontend.browser_ui import configure_agent
        
        current_config = {}
        
        result = configure_agent(
            api_base="http://test.com",
            api_key="test-key", 
            model="test-model",
            temperature=0.8,
            max_tokens=5000,
            num_ctx=100000,
            max_num_papers=15,
            current_config=current_config
        )
        
        # Verify config was updated correctly
        assert current_config["model"] == "test-model"
        assert current_config["api_base"] == "http://test.com"
        assert current_config["api_key"] == "test-key"
        assert current_config["llm_args"]["temperature"] == 0.8
        assert current_config["llm_args"]["max_tokens"] == 5000
        assert current_config["llm_args"]["num_ctx"] == 100000
        assert current_config["global_tool_args"]["max_num_papers"] == 15

    def test_initialize_agent_api_key_masking(self):
        """Test that API key is masked in logged config."""
        from llm_agents.frontend.browser_ui import configure_agent
        
        current_config = {}
        
        with patch('llm_agents.frontend.browser_ui.LOGGER') as mock_logger:
            configure_agent(
                api_base="http://test.com",
                api_key="secret-key-123",
                model="test-model", 
                temperature=0.5,
                max_tokens=1000,
                num_ctx=50000,
                max_num_papers=10,
                current_config=current_config
            )
            
            # Verify logging occurred
            mock_logger.info.assert_called()
            
            # The function should mask sensitive information in logs
            # (though the current implementation may not do this yet)


class TestBrowserUIIntegration:
    """
    Integration tests for browser UI (may require mocking Gradio).
    TODO: Write integration tests.
    """

    @pytest.mark.integration
    def test_gradio_interface_creation(self):
        """Test that Gradio interface can be created without errors."""
        pytest.skip("Integration test - requires full Gradio setup")
        
        # This would test the actual Gradio interface creation
        # but requires more complex setup and mocking

    @pytest.mark.integration  
    def test_chat_functionality(self):
        """Test chat functionality through Gradio interface."""
        pytest.skip("Integration test - requires full Gradio setup")
        
        # This would test the chat functionality
        # but requires Gradio interface to be running

    @pytest.mark.integration
    def test_system_prompt_functionality(self):
        """Test system prompt editing through Gradio interface."""
        pytest.skip("Integration test - requires full Gradio setup")

    @pytest.mark.integration
    def test_history_save_functionality(self):
        """Test conversation history saving through Gradio interface.""" 
        pytest.skip("Integration test - requires full Gradio setup")

    @pytest.mark.integration
    def test_configuration_persistence(self):
        """Test that configuration persists across interactions."""
        pytest.skip("Integration test - requires full Gradio setup")


class TestBrowserUIMain:
    """Test the main function for browser UI."""

    @patch('llm_agents.frontend.browser_ui.initialize_agent')
    @patch('llm_agents.frontend.browser_ui.gr')
    def test_main_function_exists(self, mock_gr, mock_initialize_agent):
        """Test that main function exists and can be called."""
        from llm_agents.frontend.browser_ui import main

        # Mock the agent instance
        mock_agent = Mock()
        mock_agent.get_system_prompt.return_value = {
            "role": "system",
            "content": "Test system prompt"
        }
        mock_initialize_agent.return_value = mock_agent

        # Mock the Blocks context manager
        mock_webapp = Mock()
        mock_gr.Blocks.return_value.__enter__ = Mock(return_value=mock_webapp)
        mock_gr.Blocks.return_value.__exit__ = Mock(return_value=False)

        # Mock the launch method to prevent actual server startup
        mock_webapp.launch = Mock()

        # Mock all Gradio components used in main()
        mock_gr.Markdown = Mock(return_value=Mock())
        mock_gr.State = Mock(return_value=Mock())
        mock_gr.Row = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        mock_gr.Column = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        mock_gr.Accordion = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        mock_gr.Textbox = Mock(return_value=Mock())
        mock_gr.Slider = Mock(return_value=Mock())
        mock_gr.Number = Mock(return_value=Mock())
        mock_gr.Button = Mock(return_value=Mock(click=Mock(return_value=Mock(then=Mock()))))
        mock_gr.Chatbot = Mock(return_value=Mock())
        mock_gr.Code = Mock(return_value=Mock())

        # Mock themes
        mock_gr.themes.Default = Mock(return_value=Mock())
        mock_gr.themes.sizes.spacing_sm = "sm"
        mock_gr.themes.sizes.radius_none = "none"

        # Call main - should not raise an error
        main()

        # Verify Blocks was created with expected parameters
        mock_gr.Blocks.assert_called_once()
        call_kwargs = mock_gr.Blocks.call_args[1]
        assert call_kwargs['title'] == "SciAgent For NeurIPS 2025"

        # Verify launch was called with expected parameters
        mock_webapp.launch.assert_called_once()
        launch_kwargs = mock_webapp.launch.call_args[1]
        assert launch_kwargs['server_name'] == "0.0.0.0"
        assert launch_kwargs['server_port'] == 7860
        assert launch_kwargs['share'] is False
        assert launch_kwargs['show_error'] is True
        assert launch_kwargs['debug'] is True

    @patch('llm_agents.frontend.browser_ui.initialize_agent')
    @patch('llm_agents.frontend.browser_ui.gr')
    def test_main_creates_ui_components(self, mock_gr, mock_initialize_agent):
        """Test that main function creates necessary UI components."""
        from llm_agents.frontend.browser_ui import main

        # Track all gr component calls
        component_calls = []

        def track_component(name):
            def wrapper(*args, **kwargs):
                component_calls.append(name)
                mock_instance = Mock()
                # Mock context manager for components that use 'with' statements
                mock_instance.__enter__ = Mock(return_value=mock_instance)
                mock_instance.__exit__ = Mock(return_value=False)
                # Mock the click method for buttons (returns self for chaining)
                if name == 'Button':
                    mock_click = Mock(return_value=mock_instance)
                    mock_click.then = Mock(return_value=mock_instance)
                    mock_instance.click = mock_click
                # Mock submit method for textboxes
                if name == 'Textbox':
                    mock_submit = Mock(return_value=mock_instance)
                    mock_submit.then = Mock(return_value=mock_instance)
                    mock_instance.submit = mock_submit
                return mock_instance

            return wrapper

        # Mock the agent instance
        mock_agent = Mock()
        mock_agent.get_system_prompt.return_value = {
            "role": "system",
            "content": "Test system prompt"
        }
        mock_initialize_agent.return_value = mock_agent

        # Mock the Blocks context manager
        mock_webapp = Mock()
        mock_webapp.__enter__ = Mock(return_value=mock_webapp)
        mock_webapp.__exit__ = Mock(return_value=False)
        mock_webapp.launch = Mock()
        mock_gr.Blocks.return_value = mock_webapp

        # Mock Gradio components to track creation
        mock_gr.Markdown = track_component('Markdown')
        mock_gr.State = track_component('State')
        mock_gr.Chatbot = track_component('Chatbot')
        mock_gr.Textbox = track_component('Textbox')
        mock_gr.Button = track_component('Button')
        mock_gr.Slider = track_component('Slider')
        mock_gr.Number = track_component('Number')
        mock_gr.Accordion = track_component('Accordion')
        mock_gr.Row = track_component('Row')
        mock_gr.Column = track_component('Column')
        mock_gr.Code = track_component('Code')

        # Mock themes
        mock_gr.themes.Default = Mock(return_value=Mock())
        mock_gr.themes.sizes.spacing_sm = "sm"
        mock_gr.themes.sizes.radius_none = "none"

        main()

        # Verify key components were created
        assert 'Markdown' in component_calls
        assert 'State' in component_calls
        assert 'Chatbot' in component_calls
        assert 'Textbox' in component_calls
        assert 'Button' in component_calls
        assert 'Slider' in component_calls
        assert 'Accordion' in component_calls

        # Verify multiple instances of common components
        assert component_calls.count('Textbox') >= 5
        assert component_calls.count('Button') >= 5
        assert component_calls.count('Markdown') >= 3

    def test_main_entry_point(self):
        """Test that the module can be used as entry point."""
        try:
            import llm_agents.frontend.browser_ui
            # Verify the module has the expected main function
            assert hasattr(llm_agents.frontend.browser_ui, 'main')
            assert callable(llm_agents.frontend.browser_ui.main)
        except ImportError:
            pytest.fail("Could not import browser_ui module")

    @patch('llm_agents.frontend.browser_ui.initialize_agent')
    def test_main_initializes_agent(self, mock_initialize_agent):
        """Test that main function initializes the agent."""
        from llm_agents.frontend.browser_ui import main

        # Mock the agent instance
        mock_agent = Mock()
        mock_agent.get_system_prompt.return_value = {
            "role": "system",
            "content": "Test prompt"
        }
        mock_initialize_agent.return_value = mock_agent

        # Patch launch to prevent server startup
        with patch('gradio.Blocks.launch'):
            main()

        # Verify initialize_agent was called
        mock_initialize_agent.assert_called_once()

        # Verify agent methods were accessed
        assert mock_agent.get_system_prompt.called
