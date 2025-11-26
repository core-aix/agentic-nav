"""
Tests for the CLI frontend.
"""
import pytest
import asyncio
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from pathlib import Path

# Import the module under test
from agentic_nav.frontend.cli import (
    create_prompt_session,
    render_markdown,
    stream_agent_response_sync,
    async_interact,
    print_welcome,
    main
)


class TestCreatePromptSession:
    """Test the create_prompt_session function."""

    @patch('agentic_nav.frontend.cli.PromptSession')
    @patch('agentic_nav.frontend.cli.FileHistory')
    def test_create_prompt_session_basic(self, mock_file_history, mock_prompt_session):
        """Test basic prompt session creation."""
        mock_session_instance = Mock()
        mock_prompt_session.return_value = mock_session_instance
        
        result = create_prompt_session()
        
        # Verify FileHistory was created with correct path
        home_path = str(Path.home() / ".llm_agents_history")
        mock_file_history.assert_called_once_with(home_path)
        
        # Verify PromptSession was configured correctly
        mock_prompt_session.assert_called_once()
        call_kwargs = mock_prompt_session.call_args.kwargs
        
        assert 'history' in call_kwargs
        assert 'auto_suggest' in call_kwargs
        assert 'completer' in call_kwargs
        assert call_kwargs['complete_while_typing'] is True
        assert call_kwargs['enable_open_in_editor'] is True
        assert call_kwargs['multiline'] is False
        
        assert result == mock_session_instance


class TestRenderMarkdown:
    """Test the render_markdown function."""

    @patch('agentic_nav.frontend.cli.console')
    def test_render_markdown_without_title(self, mock_console):
        """Test markdown rendering without title."""
        render_markdown("# Test Markdown")
        
        mock_console.print.assert_called_once()
        # Should be called with Markdown object, not Panel
        call_args = mock_console.print.call_args[0][0]
        assert hasattr(call_args, 'markup')  # Markdown object

    @patch('agentic_nav.frontend.cli.console')
    @patch('agentic_nav.frontend.cli.Panel')
    def test_render_markdown_with_title(self, mock_panel, mock_console):
        """Test markdown rendering with title."""
        mock_panel_instance = Mock()
        mock_panel.return_value = mock_panel_instance
        
        render_markdown("# Test Markdown", title="Test Title")
        
        # Should create Panel with title
        mock_panel.assert_called_once()
        call_args = mock_panel.call_args
        assert call_args.kwargs['title'] == "Test Title"
        assert call_args.kwargs['border_style'] == "blue"
        
        mock_console.print.assert_called_once_with(mock_panel_instance)


class TestStreamAgentResponse:
    """Test the stream_agent_response_sync function."""

    def test_stream_agent_response_basic(self):
        """Test basic streaming response functionality."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.get_history.return_value = [
            {"role": "user", "content": "Hello"}
        ]
        
        # Mock interact_stateless to yield message updates
        message_updates = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ],
            [
                {"role": "user", "content": "Hello"}, 
                {"role": "assistant", "content": "Hi there!"}
            ]
        ]
        mock_agent.interact_stateless.return_value = iter(message_updates)
        
        # Mock Live context manager
        with patch('agentic_nav.frontend.cli.Live') as mock_live:
            mock_live_instance = Mock()
            mock_live.return_value.__enter__.return_value = mock_live_instance
            
            message = {"role": "user", "content": "test"}
            stream_agent_response_sync(mock_agent, message)
            
            # Verify agent's set_history was called with final messages
            mock_agent.set_history.assert_called_once_with(message_updates[-1])
            
            # Verify Live was used for streaming display
            mock_live_instance.update.assert_called()

    def test_stream_agent_response_keyboard_interrupt(self):
        """Test handling of keyboard interrupt during streaming."""
        mock_agent = Mock()
        mock_agent.get_history.return_value = []
        
        # Mock interact_stateless to raise KeyboardInterrupt
        mock_agent.interact_stateless.side_effect = KeyboardInterrupt()
        
        with patch('agentic_nav.frontend.cli.Live') as mock_live:
            mock_live_instance = Mock()
            mock_live.return_value.__enter__.return_value = mock_live_instance
            
            message = {"role": "user", "content": "test"}
            
            with pytest.raises(KeyboardInterrupt):
                stream_agent_response_sync(mock_agent, message)
            
            # Verify Live was stopped
            mock_live_instance.stop.assert_called_once()

    def test_stream_agent_response_with_tool_calls(self):
        """Test streaming response with tool calls."""
        mock_agent = Mock()
        mock_agent.get_history.return_value = []
        
        # Mock response with tool calls
        message_updates = [
            [
                {"role": "assistant", "content": "I'll help with that.", "tool_calls": [
                    {"function": {"name": "search_papers"}}
                ]}
            ]
        ]
        mock_agent.interact_stateless.return_value = iter(message_updates)
        
        with patch('agentic_nav.frontend.cli.Live') as mock_live:
            mock_live_instance = Mock()
            mock_live.return_value.__enter__.return_value = mock_live_instance
            
            message = {"role": "user", "content": "test"}
            stream_agent_response_sync(mock_agent, message)
            
            # Should have updated with tool execution info
            mock_live_instance.update.assert_called()


class TestAsyncInteract:
    """Test the async_interact function."""

    @pytest.mark.asyncio
    async def test_async_interact_success(self):
        """Test successful async interaction."""
        mock_agent = Mock()
        message = {"role": "user", "content": "test"}
        
        with patch('agentic_nav.frontend.cli.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = None  # Successful completion
            
            await async_interact(mock_agent, message)
            
            # Verify stream_agent_response_sync was called in thread
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_interact_keyboard_interrupt(self):
        """Test async interaction with keyboard interrupt."""
        mock_agent = Mock()
        message = {"role": "user", "content": "test"}
        
        with patch('agentic_nav.frontend.cli.asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = KeyboardInterrupt()
            
            # Should handle KeyboardInterrupt gracefully
            await async_interact(mock_agent, message)

    @pytest.mark.asyncio
    async def test_async_interact_exception(self):
        """Test async interaction with general exception."""
        mock_agent = Mock()
        message = {"role": "user", "content": "test"}
        
        with patch('agentic_nav.frontend.cli.asyncio.to_thread') as mock_to_thread:
            with patch('agentic_nav.frontend.cli.console') as mock_console:
                mock_to_thread.side_effect = Exception("Test error")
                
                await async_interact(mock_agent, message)
                
                # Should print error message
                mock_console.print.assert_called()


class TestPrintWelcome:
    """Test the print_welcome function."""

    @patch('agentic_nav.frontend.cli.console')
    def test_print_welcome(self, mock_console):
        """Test that welcome message is printed."""
        print_welcome()
        
        mock_console.print.assert_called_once()
        # Verify welcome text contains expected elements
        call_args = mock_console.print.call_args[0][0]
        welcome_text = str(call_args)
        
        assert "LLM Agent Chat Interface" in welcome_text
        assert "/help" in welcome_text
        assert "/exit" in welcome_text
        assert "Ctrl+C" in welcome_text


class TestMain:
    """Test the main CLI function."""

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_main_initialization(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test main function initialization."""
        # Mock agent instance
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock prompt session to exit immediately
        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = EOFError()
        mock_session.return_value = mock_session_instance
        
        # Import click for testing
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        # Verify initialization steps
        mock_setup_logging.assert_called_once()
        mock_welcome.assert_called_once()
        mock_agent_class.assert_called_once()
        mock_agent.setup_session.assert_called_once()

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_main_with_custom_params(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test main function with custom CLI parameters."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = EOFError()
        mock_session.return_value = mock_session_instance
        
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, [
            '--temperature', '0.8',
            '--max-tokens', '4000',
            '--num-ctx', '65536', 
            '--max-num-papers', '20'
        ])
        
        # Verify agent was created with custom parameters
        call_args = mock_agent_class.call_args
        llm_config = call_args.kwargs['llm_args']
        tool_args = call_args.kwargs['global_tool_args']
        
        assert llm_config['temperature'] == 0.8
        assert llm_config['max_tokens'] == 4000
        assert llm_config['num_ctx'] == 65536
        assert tool_args['num_records'] == 20

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    @patch('agentic_nav.frontend.cli.asyncio')
    def test_main_user_interaction(self, mock_asyncio, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test main function user interaction loop."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock session to return user input then exit
        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = [
            "test question",  # First input
            EOFError()  # Exit
        ]
        mock_session.return_value = mock_session_instance
        
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        # Verify asyncio.run was called for user interaction
        mock_asyncio.run.assert_called()

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_main_help_command(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test main function with help command."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock session to return help command then exit
        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = [
            "/help",  # Help command
            EOFError()  # Exit
        ]
        mock_session.return_value = mock_session_instance
        
        with patch('agentic_nav.frontend.cli.print_help') as mock_print_help:
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(main, [])
            
            # Verify help was printed
            mock_print_help.assert_called_once()

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_main_exit_command(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test main function with exit command."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Mock session to return exit command
        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = ["/exit"]
        mock_session.return_value = mock_session_instance
        
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        # Should exit cleanly
        assert result.exit_code == 0


class TestCommandProcessing:
    """Test command processing in main loop."""

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_save_command(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test /save command."""
        mock_agent = Mock()
        mock_agent.get_history.return_value = [{"role": "user", "content": "test"}]
        mock_agent_class.return_value = mock_agent

        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = [
            "/save test_history.json",
            EOFError()
        ]
        mock_session.return_value = mock_session_instance

        with patch('agentic_nav.frontend.cli.save_chat_history') as mock_save:
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [])

            # Verify save was called
            mock_save.assert_called_once()

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_history_command(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test /history command."""
        mock_agent = Mock()
        mock_agent.get_history.return_value = [
            {"role": "user", "content": "test"}
        ]
        mock_agent_class.return_value = mock_agent

        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = [
            "/history",
            EOFError()
        ]
        mock_session.return_value = mock_session_instance

        with patch('agentic_nav.frontend.cli.show_history') as mock_show:
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [])

            # Verify show_history was called
            mock_show.assert_called_once()

    @patch('agentic_nav.frontend.cli.setup_logging')
    @patch('agentic_nav.frontend.cli.NeurIPS2025Agent')
    @patch('agentic_nav.frontend.cli.create_prompt_session')
    @patch('agentic_nav.frontend.cli.print_welcome')
    def test_system_command(self, mock_welcome, mock_session, mock_agent_class, mock_setup_logging):
        """Test /system command."""
        mock_agent = Mock()
        mock_agent.get_system_prompt.return_value = {"role": "system", "content": "Test prompt"}
        mock_agent_class.return_value = mock_agent

        mock_session_instance = Mock()
        mock_session_instance.prompt.side_effect = [
            "/system",
            EOFError()
        ]
        mock_session.return_value = mock_session_instance

        with patch('agentic_nav.frontend.cli.console') as mock_console:
            from click.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(main, [])

            # Verify console.print was called to show system prompt
            assert mock_console.print.called

