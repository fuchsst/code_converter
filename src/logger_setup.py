# logger_setup.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "conversion.log")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

def setup_logging(log_level=logging.INFO):
    """Configures logging for the application."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level) # Or set a different level for console

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level) # Set root logger level

    # Clear existing handlers (important for multiple calls or testing)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging setup complete.")

def get_logger(name):
    """Gets a logger instance for a specific module."""
    return logging.getLogger(name)

# Example usage in other modules:
# from logger_setup import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")
# logger.error("This is an error message.")