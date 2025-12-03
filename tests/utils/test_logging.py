"""
Tests for the logging utility.
"""
import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from agentic_nav.utils.logger import setup_logging


class TestSetupLogging:
    """Test the setup_logging function."""

    def test_setup_logging_creates_log_directory(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            assert not log_dir.exists()
            
            setup_logging(log_dir=str(log_dir), level="INFO")
            
            assert log_dir.exists()
            assert log_dir.is_dir()

    def test_setup_logging_existing_directory(self):
        """Test that function works with existing log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "existing_logs"
            log_dir.mkdir()
            
            # Should not raise error
            setup_logging(log_dir=str(log_dir), level="INFO")

    @patch('agentic_nav.utils.logger.datetime')
    def test_setup_logging_creates_handlers(self, mock_datetime):
        """Test that console handler is created (file handler is disabled)."""
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-00"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Clear any existing handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            setup_logging(log_dir=temp_dir, level="DEBUG")

            # Check that only console handler was added (file handler is commented out)
            assert len(root_logger.handlers) == 1

            # Check handler types
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert "StreamHandler" in handler_types

    def test_setup_logging_sets_log_levels(self):
        """Test that log levels are set correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_logger = logging.getLogger()
            
            # Test DEBUG level
            setup_logging(log_dir=temp_dir, level="DEBUG")
            assert root_logger.level == logging.DEBUG
            
            # Clear handlers and test INFO level
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            setup_logging(log_dir=temp_dir, level="INFO")
            assert root_logger.level == logging.INFO

    def test_setup_logging_invalid_level(self):
        """Test behavior with invalid log level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should raise AttributeError for invalid level
            with pytest.raises(AttributeError):
                setup_logging(log_dir=temp_dir, level="INVALID")

    @patch('agentic_nav.utils.logger.datetime')
    def test_setup_logging_file_naming(self, mock_datetime):
        """Test that log files are named correctly (currently file handler is disabled)."""
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01_12-30"

        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_dir=temp_dir, level="INFO")

            # File handler is currently commented out, so no log files are created
            log_files = list(Path(temp_dir).glob("*.log"))
            assert len(log_files) == 0

    def test_setup_logging_handler_levels(self):
        """Test that console handler has correct log level (file handler is disabled)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_logger = logging.getLogger()
            # Clear existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            setup_logging(log_dir=temp_dir, level="DEBUG")

            # Find console handler
            console_handler = None

            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    console_handler = handler

            # Verify console handler level
            assert console_handler is not None
            assert console_handler.level == logging.WARNING

    def test_setup_logging_formatters(self):
        """Test that formatters are set correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_logger = logging.getLogger()
            # Clear existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            setup_logging(log_dir=temp_dir, level="INFO")
            
            for handler in root_logger.handlers:
                formatter = handler.formatter
                assert formatter is not None
                assert isinstance(formatter, logging.Formatter)
                
                # Check that format string contains expected elements
                format_string = formatter._fmt
                assert "%(asctime)s" in format_string
                assert "%(levelname)s" in format_string
                assert "%(name)s" in format_string
                assert "%(message)s" in format_string

    @patch('agentic_nav.utils.logger.logging.handlers.RotatingFileHandler')
    def test_setup_logging_rotating_file_config(self, mock_rotating_handler):
        """Test that rotating file handler is not created (currently disabled)."""
        mock_handler_instance = Mock()
        mock_rotating_handler.return_value = mock_handler_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_dir=temp_dir, level="INFO")

            # RotatingFileHandler creation is commented out, so it should not be called
            mock_rotating_handler.assert_not_called()

    def test_setup_logging_default_parameters(self):
        """Test function with default parameters."""
        # Clean up any existing log directory first
        default_log_dir = Path("logs")
        
        # Clear existing handlers to avoid interference
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        try:
            setup_logging()
            
            # Check default log directory was created
            assert default_log_dir.exists()
            
            # Check default log level
            assert root_logger.level == logging.INFO
            
        finally:
            # Clean up handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)