# config.py
import os
from dotenv import load_dotenv
from src.logger_setup import get_logger

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
# Format: "provider/model_name" (e.g., "gemini/gemini-1.5-pro-latest")
DEFAULT_AGENT_MODEL = "gemini/gemini-2.0-flash-001" # Cost-effective default
# Environment variables (if set) should also follow the "provider/model_name" format.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini/gemini-2.5-pro-preview-03-25")
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", DEFAULT_AGENT_MODEL) # Defaults to the already formatted DEFAULT_AGENT_MODEL
MAPPER_MODEL = os.getenv("MAPPER_MODEL", "gemini/gemini-2.5-pro-exp-03-25")
GENERATOR_EDITOR_MODEL = os.getenv("GENERATOR_EDITOR_MODEL", DEFAULT_AGENT_MODEL) # Defaults to the already formatted DEFAULT_AGENT_MODEL
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "gemini/gemini-2.5-pro-exp-03-25")

# --- Path Configuration ---
# These should ideally be passed via CLI arguments, but provide defaults or load from env
CPP_PROJECT_DIR = os.getenv("CPP_PROJECT_DIR", "data/cpp_project")
GODOT_PROJECT_DIR = os.getenv("GODOT_PROJECT_DIR", "input/godot_project")
GODOT_DOCS_DIR = os.getenv("GODOT_DOCS_DIR", "input/godot_docs")
ANALYSIS_OUTPUT_DIR = os.getenv("ANALYSIS_OUTPUT_DIR", "output/analysis")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# --- Tool Paths ---
# Paths to external executables or scripts needed by tools
GODOT_EXECUTABLE_PATH = os.getenv("GODOT_EXECUTABLE_PATH", "godot") # Assumes godot is in PATH


# --- API Rate Limit Configuration ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
INITIAL_BACKOFF_DELAY = float(os.getenv("INITIAL_BACKOFF_DELAY", 1.0)) # seconds
MAX_BACKOFF_DELAY = float(os.getenv("MAX_BACKOFF_DELAY", 60.0)) # seconds
BACKOFF_JITTER = float(os.getenv("BACKOFF_JITTER", 0.1)) # Max jitter fraction (e.g., 0.1 = +/- 10%)

# --- Conversion Settings ---
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "GDScript") # Default, override via CLI

# --- Orchestrator Settings ---
TASK_ITEM_MAX_RETRIES = int(os.getenv("TASK_ITEM_MAX_RETRIES", 2)) # Max retries for a single task item in Step 5
MAX_REMAPPING_ATTEMPTS = int(os.getenv("MAX_REMAPPING_ATTEMPTS", 1)) # Max times Step 4 can be re-run for a package

# --- Context & Token Management ---
# Max tokens allowed for the context assembled *before* adding the main prompt
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 800000)) # Gemini 2.5 Pro has 1M, leave buffer
# Buffer subtracted from MAX_CONTEXT_TOKENS to leave room for prompt/response overhead
PROMPT_TOKEN_BUFFER = int(os.getenv("PROMPT_TOKEN_BUFFER", 5000))
# Ratio of MAX_CONTEXT_TOKENS allocated to file content vs other context items
CONTEXT_FILE_BUDGET_RATIO = float(os.getenv("CONTEXT_FILE_BUDGET_RATIO", 0.9)) # 90% for files

# --- Generation Settings ---
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.95)) # Example, adjust as needed
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 40)) # Example, adjust as needed

# --- Ensure output/generated directories exist ---
# Input directories (CPP_PROJECT_DIR, GODOT_PROJECT_DIR, GODOT_DOCS_DIR) are expected to be provided.
dirs_to_ensure = [
    ANALYSIS_OUTPUT_DIR,
    LOG_DIR # LOG_DIR is also created by logger_setup, but checking here is fine
]

for dir_path in dirs_to_ensure:
    if dir_path and not os.path.exists(dir_path): # Check if dir_path is not None or empty
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")

# --- Basic Validation ---
if not GODOT_EXECUTABLE_PATH:
    logger.warning("GODOT_EXECUTABLE_PATH is not set in environment or .env file. Validation tool will likely fail.")
# Simple check if it looks like a path vs just 'godot' (in PATH)
elif os.path.sep in GODOT_EXECUTABLE_PATH and not os.path.exists(GODOT_EXECUTABLE_PATH):
     logger.warning(f"GODOT_EXECUTABLE_PATH is set to '{GODOT_EXECUTABLE_PATH}', but this path does not seem to exist.")


logger.info("Configuration loaded.")

# Further validation could be added (e.g., check cpp_project_dir exists)
