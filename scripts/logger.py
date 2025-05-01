import logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Define log file path
LOG_FILE = "logs/project.log"

# Define logging format
LOG_FORMAT = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"

# Create logger object
logger = logging.getLogger("llm-bias")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all types of logs

# --- File Handler with Rotation ---
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1_000_000,  # 1 MB max file size
    backupCount=3        # Keep up to 3 old log files
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# --- Console Handler ---
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# --- Attach handlers ---
if not logger.handlers:  # Prevent adding multiple handlers if re-imported
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
