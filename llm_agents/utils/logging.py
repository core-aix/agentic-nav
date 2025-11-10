import logging
import logging.handlers

from datetime import datetime
from pathlib import Path



def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Configure logging for the entire application."""
    Path(log_dir).mkdir(exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Console handler - for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_format)

    # File handler - for production
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")

    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/{time_now}_llm_agents.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
