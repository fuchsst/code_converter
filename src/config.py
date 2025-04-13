# config.py
import os
from dotenv import load_dotenv
from logger_setup import get_logger

logger = get_logger(__name__)

# Load environment variables from.env file
load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")
    # Consider raising an error or using a default/dummy key for testing

# --- Model Configuration ---
# Allows specifying different models for different agents
DEFAULT_AGENT_MODEL = "gemini-2.0-flash-001" # Cost-effective default
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini-2.5-pro-preview-03-25")
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", DEFAULT_AGENT_MODEL)
MAPPER_MODEL = os.getenv("MAPPER_MODEL", "gemini-2.5-pro-exp-03-25")
GENERATOR_EDITOR_MODEL = os.getenv("GENERATOR_EDITOR_MODEL", DEFAULT_AGENT_MODEL) # Flash might be okay
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "gemini-2.5-pro-exp-03-25")

# --- Path Configuration ---
# These should ideally be passed via CLI arguments, but provide defaults or load from env
CPP_PROJECT_DIR = os.getenv("CPP_PROJECT_DIR", "data/cpp_project")
GODOT_PROJECT_DIR = os.getenv("GODOT_PROJECT_DIR", "data/godot_project")
GODOT_DOCS_DIR = os.getenv("GODOT_DOCS_DIR", "data/godot_docs")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output/godot_converted")
ANALYSIS_OUTPUT_DIR = os.getenv("ANALYSIS_OUTPUT_DIR", "analysis_output")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store/index")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# --- Tool Paths ---
# Paths to external executables or scripts needed by tools
CLANG_ANALYSIS_TOOL_PATH = os.getenv("CLANG_ANALYSIS_TOOL_PATH", "path/to/your/clang_analyzer") # Placeholder
GODOT_EXECUTABLE_PATH = os.getenv("GODOT_EXECUTABLE_PATH", "godot") # Assumes godot is in PATH

# --- RAG Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2") # Or Google's
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RAG_RESULTS_K = int(os.getenv("RAG_RESULTS_K", 5))

# --- API Rate Limit Configuration ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
INITIAL_BACKOFF_DELAY = float(os.getenv("INITIAL_BACKOFF_DELAY", 1.0)) # seconds
MAX_BACKOFF_DELAY = float(os.getenv("MAX_BACKOFF_DELAY", 60.0)) # seconds

# --- Conversion Settings ---
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "GDScript") # Default, override via CLI

# --- Ensure output/generated directories exist ---
# Input directories (CPP_PROJECT_DIR, GODOT_PROJECT_DIR, GODOT_DOCS_DIR) are expected to be provided.
dirs_to_ensure = [
    OUTPUT_DIR,
    ANALYSIS_OUTPUT_DIR,
    VECTOR_STORE_DIR,
    LOG_DIR # LOG_DIR is also created by logger_setup, but checking here is fine
]

for dir_path in dirs_to_ensure:
    if dir_path and not os.path.exists(dir_path): # Check if dir_path is not None or empty
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")

logger.info("Configuration loaded.")

# You might want to add validation logic here (e.g., check paths exist)
