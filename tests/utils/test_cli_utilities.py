"""
Tests for CLI utility functions.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from agentic_nav.utils.cli.editor import open_editor
from agentic_nav.utils.cli.help import print_help
from agentic_nav.utils.cli.history import show_history
from agentic_nav.utils.file_handlers import save_chat_history


class TestOpenEditor:
    """Test the open_editor function."""

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    def test_open_editor_with_custom_editor(self, mock_env_get, mock_system):
        """Test opening editor with custom EDITOR environment variable."""
        mock_env_get.return_value = "vim"
        mock_system.return_value = 0

        # Mock the file read to return edited content
        with patch('builtins.open', mock_open(read_data="edited content")):
            result = open_editor("initial text")

            # Verify editor was called
            mock_system.assert_called_once()
            call_args = mock_system.call_args[0][0]
            assert "vim" in call_args

            assert result == "edited content"

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    @patch('agentic_nav.utils.cli.editor.os.name', 'posix')
    def test_open_editor_default_unix(self, mock_env_get, mock_system):
        """Test opening editor with default Unix editor (nano)."""
        mock_env_get.return_value = None  # No EDITOR set
        mock_system.return_value = 0

        with patch('builtins.open', mock_open(read_data="content")):
            result = open_editor()

            # Verify nano was used as default
            call_args = mock_system.call_args[0][0]
            assert "nano" in call_args

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    @patch('agentic_nav.utils.cli.editor.os.name', 'nt')
    def test_open_editor_default_windows(self, mock_env_get, mock_system):
        """Test opening editor with default Windows editor (notepad)."""
        mock_env_get.return_value = None  # No EDITOR set
        mock_system.return_value = 0

        with patch('builtins.open', mock_open(read_data="content")):
            result = open_editor()

            # Verify notepad was used as default
            call_args = mock_system.call_args[0][0]
            assert "notepad" in call_args

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    @patch('agentic_nav.utils.cli.editor.tempfile.NamedTemporaryFile')
    def test_open_editor_with_initial_text(self, mock_temp, mock_env_get, mock_system):
        """Test that initial text is written to temp file."""
        mock_env_get.return_value = "nano"
        mock_system.return_value = 0

        initial_text = "This is initial text"

        # Mock the temporary file
        mock_file = mock_open(read_data="modified text")()
        mock_file.name = "/tmp/test.md"
        mock_temp.return_value.__enter__.return_value = mock_file

        with patch('builtins.open', mock_open(read_data="modified text")):
            result = open_editor(initial_text)

            # Verify temp file was created and written to
            mock_temp.assert_called_once()
            assert result == "modified text"

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    def test_open_editor_strips_whitespace(self, mock_env_get, mock_system):
        """Test that returned content is stripped of whitespace."""
        mock_env_get.return_value = "nano"
        mock_system.return_value = 0

        with patch('builtins.open', mock_open(read_data="  content with spaces  \n")):
            result = open_editor()

            assert result == "content with spaces"

    @patch('agentic_nav.utils.cli.editor.os.system')
    @patch('agentic_nav.utils.cli.editor.os.environ.get')
    def test_open_editor_nonzero_exit_code(self, mock_env_get, mock_system, capsys):
        """Test handling of non-zero editor exit code."""
        mock_env_get.return_value = "nano"
        mock_system.return_value = 1  # Non-zero exit

        with patch('builtins.open', mock_open(read_data="content")):
            result = open_editor()

            # Verify exit code was printed
            captured = capsys.readouterr()
            assert "(editor exit code 1)" in captured.out


class TestPrintHelp:
    """Test the print_help function."""

    def test_print_help_output(self, capsys):
        """Test that help text is printed correctly."""
        print_help()

        captured = capsys.readouterr()
        output = captured.out

        # Verify key commands are present
        assert "Commands:" in output
        assert "/help" in output
        assert "/exit" in output
        assert "/system" in output
        assert "/edit" in output
        assert "/history" in output
        assert "/save" in output

    def test_print_help_describes_commands(self, capsys):
        """Test that help includes command descriptions."""
        print_help()

        captured = capsys.readouterr()
        output = captured.out

        # Verify descriptions are present
        assert "Show this help" in output
        assert "Exit the chat" in output
        assert "system prompt" in output
        assert "conversation history" in output


class TestShowHistory:
    """Test the show_history function."""

    def test_show_history_basic(self, capsys):
        """Test basic history display."""
        messages = [
            {"role": "user", "content": "Hello", "_ts": "2024-01-01 12:00:00"},
            {"role": "assistant", "content": "Hi there!", "_ts": "2024-01-01 12:00:01"}
        ]

        show_history(messages)

        captured = capsys.readouterr()
        output = captured.out

        # Verify all messages are displayed
        assert "[0] user 2024-01-01 12:00:00" in output
        assert "Hello" in output
        assert "[1] assistant 2024-01-01 12:00:01" in output
        assert "Hi there!" in output

    def test_show_history_missing_fields(self, capsys):
        """Test history display with missing optional fields."""
        messages = [
            {"role": "user", "content": "Hello"},  # No timestamp
            {"content": "World"}  # No role
        ]

        show_history(messages)

        captured = capsys.readouterr()
        output = captured.out

        # Should still display without errors
        assert "[0] user" in output
        assert "Hello" in output
        assert "[1]" in output
        assert "World" in output

    def test_show_history_empty_list(self, capsys):
        """Test history display with empty message list."""
        messages = []

        show_history(messages)

        captured = capsys.readouterr()
        # Should not crash, output will be empty
        assert captured.out == ""

    def test_show_history_formatting(self, capsys):
        """Test that history formatting includes separators."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant", "_ts": "2024-01-01"}
        ]

        show_history(messages)

        captured = capsys.readouterr()
        output = captured.out

        # Verify formatting elements
        assert "---" in output  # Separator line
        assert "[0]" in output


class TestSaveChatHistory:
    """Test the save_chat_history function."""

    def test_save_chat_history_basic(self, capsys):
        """Test basic chat history saving."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "history.json"

            save_chat_history(messages, str(file_path))

            # Verify file was created
            assert file_path.exists()

            # Verify content
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)

            assert loaded_messages == messages

            # Verify success message
            captured = capsys.readouterr()
            assert f"Saved to {file_path}" in captured.out

    def test_save_chat_history_with_unicode(self):
        """Test saving history with unicode characters."""
        messages = [
            {"role": "user", "content": "Hello 世界 🌍"},
            {"role": "assistant", "content": "Bonjour café"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "unicode_history.json"

            save_chat_history(messages, str(file_path))

            # Verify unicode is preserved
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)

            assert loaded_messages[0]["content"] == "Hello 世界 🌍"
            assert loaded_messages[1]["content"] == "Bonjour café"

    def test_save_chat_history_empty_list(self):
        """Test saving empty chat history."""
        messages = []

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "empty_history.json"

            save_chat_history(messages, str(file_path))

            assert file_path.exists()

            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)

            assert loaded_messages == []

    def test_save_chat_history_complex_messages(self):
        """Test saving history with complex message structures."""
        messages = [
            {
                "role": "assistant",
                "content": "Here's the result",
                "tool_calls": [{"id": "call1", "function": {"name": "search"}}],
                "_ts": "2024-01-01 12:00:00"
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "complex_history.json"

            save_chat_history(messages, str(file_path))

            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)

            assert loaded_messages[0]["tool_calls"] == messages[0]["tool_calls"]

    def test_save_chat_history_invalid_path(self, capsys):
        """Test error handling for invalid save path."""
        messages = [{"role": "user", "content": "test"}]

        # Try to save to invalid path
        save_chat_history(messages, "/invalid/path/that/does/not/exist/history.json")

        # Verify error message
        captured = capsys.readouterr()
        assert "Save failed:" in captured.out

    def test_save_chat_history_formatting(self):
        """Test that saved JSON is properly formatted (indented)."""
        messages = [
            {"role": "user", "content": "test"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "formatted_history.json"

            save_chat_history(messages, str(file_path))

            # Verify formatting by checking file content directly
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Should have indentation (not single line)
            assert "\n" in content
            assert "  " in content  # Should have 2-space indentation
