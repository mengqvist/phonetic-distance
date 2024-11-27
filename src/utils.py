import logging
import sys
from pathlib import Path
import numpy as np

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger with consistent formatting.
    
    Args:
        name: Name of the logger, typically __name__ from the calling module
        log_file: Optional path to log file. If None, logs only to console
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optionally add file handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_root() -> Path:
    """Get project root directory from any script location"""
    # Start with current working directory
    current_dir = Path.cwd()
    
    # Look for key project files/directories that indicate the root
    # For example, look for 'data' dir, requirements.yml, or .git
    while current_dir != current_dir.parent:  # stop at root
        if (current_dir / 'environment.yml').exists() or \
           (current_dir / '.git').exists() or \
           (current_dir / 'data').exists():
            return current_dir
        current_dir = current_dir.parent
    
    # If we couldn't find project root, use current directory
    return Path.cwd()
