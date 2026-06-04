import logging
import sys
from typing import Optional


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Sets up a standardized logger with console output.
    
    Args:
        name (str): The name of the logger (typically __name__).
        log_level (str): The string representation of logging level (e.g. INFO, DEBUG).
        
    Returns:
        logging.Logger: Standard python logger configured with custom format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers if setup_logger is called multiple times for the same name
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
