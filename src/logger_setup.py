# logger_setup.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "conversion.log")
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels for console output"""
    
    # ANSI escape codes (works on most Unix terminals)
    COLOR_CODES = {
        logging.INFO: '\033[32m',    # Green
        logging.WARNING: '\033[33m', # Yellow
        logging.ERROR: '\033[31m',   # Red
    }
    RESET_CODE = '\033[0m'

    def format(self, record):
        # Add color codes to the record
        record.color_start = self.COLOR_CODES.get(record.levelno, '')
        record.color_end = self.RESET_CODE if record.color_start else ''
        return super().format(record)

def setup_logging(log_level=logging.DEBUG):
    """Configures logging with colored console output"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logging.basicConfig(encoding='utf-8')

    # Formatter for file handler (no colors)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Formatter for console handler (with colors)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(color_start)s%(levelname)s%(color_end)s - %(message)s'
    )

    # File Handler (rotating logs)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # Console Handler (colored output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clean up existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging setup complete.")

def get_logger(name):
    """Gets a logger instance for a specific module."""
    return logging.getLogger(name)