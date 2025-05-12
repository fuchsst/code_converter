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
# Grouped models for different task types. Allows overriding via environment variables.
# Format: "provider/model_name" (e.g., "gemini/gemini-1.5-pro-latest")

# Manager Roles (Steps 3, 4, 5): Orchestration, delegation, review. Requires strong reasoning.
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemini/gemini-2.5-pro-preview-03-25") # Best reasoning

# Analysis Roles (Steps 2, 3, 4, 5): Understanding code, structure, context, failures.
ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gemini/gemini-2.5-flash-preview-04-17") # Good capability/cost balance, large context

# Design/Planning Roles (Steps 3, 4): Synthesizing analysis into structure, strategy, tasks.
DESIGNER_PLANNER_MODEL = os.getenv("DESIGNER_PLANNER_MODEL", "gemini/gemini-2.5-flash-preview-04-17") # Good capability/cost balance

# Code Generation/Refinement Roles (Step 5): Generating/fixing code based on instructions.
GENERATOR_REFINER_MODEL = os.getenv("GENERATOR_REFINER_MODEL", "gemini/gemini-2.5-flash-preview-04-17") # Good coding capability

# Utility Roles (Steps 3, 4, 5): Simple, constrained tasks like formatting, tool use.
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gemini/gemini-2.0-flash-lite") # Most cost-effective for simple tasks

# Default model if a specific role isn't assigned (should ideally not be needed with the above groupings)
DEFAULT_AGENT_MODEL = os.getenv("DEFAULT_AGENT_MODEL", "gemini/gemini-2.5-flash-preview-04-17")

# --- Path Configuration ---
# These should ideally be passed via CLI arguments, but provide defaults or load from env
CPP_PROJECT_DIR = os.getenv("CPP_PROJECT_DIR", "data/cpp_project")
GODOT_PROJECT_DIR = os.getenv("GODOT_PROJECT_DIR", "input/godot_project")
INSTRUCTION_DIR = os.getenv("INSTRUCTION_DIR", "input/instructions")
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

# --- Gemini Specific Settings ---
# Timeout for Google Gemini API calls in seconds
default_gemini_timeout = 60*15 # 15 minutes
try:
    VERTEX_TIMEOUT = int(os.getenv("VERTEX_TIMEOUT", default_gemini_timeout))
except ValueError:
    logger.warning(f"Invalid GEMINI_TIMEOUT value '{os.getenv('VERTEX_TIMEOUT')}'. Using default: {default_gemini_timeout}s")
    VERTEX_TIMEOUT = default_gemini_timeout


# --- Conversion Settings ---
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "GDScript") # Default, override via CLI

# --- Orchestrator Settings ---
TASK_ITEM_MAX_RETRIES = int(os.getenv("TASK_ITEM_MAX_RETRIES", 2)) # Max retries for a single task item in Step 5
MAX_REMAPPING_ATTEMPTS = int(os.getenv("MAX_REMAPPING_ATTEMPTS", 1)) # Max times Step 4 can be re-run for a package
LLM_CALL_RETRIES = int(os.getenv("LLM_CALL_RETRIES", 2)) # General retries for LLM calls within steps (e.g., Step 2 details)

# --- Dependency Analysis Settings (Step 1) ---
# Comma-separated list of folder paths relative to CPP_PROJECT_DIR to exclude
# Example: EXCLUDE_FOLDERS="build,tests,external/lib"
exclude_folders_str = os.getenv("EXCLUDE_FOLDERS", "")
EXCLUDE_FOLDERS = [folder.strip() for folder in exclude_folders_str.split(',') if folder.strip()] if exclude_folders_str else []
if EXCLUDE_FOLDERS:
    logger.info(f"Dependency analysis will exclude folders: {EXCLUDE_FOLDERS}")


# --- Context & Token Management ---
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 32*1024)) # Max tokens for the model's output
# Max tokens allowed for the context assembled *before* adding the main prompt
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 700_000)) # Gemini 2.5 Pro has 1M, leave buffer
# Buffer subtracted from MAX_CONTEXT_TOKENS to leave room for prompt/response overhead
PROMPT_TOKEN_BUFFER = int(os.getenv("PROMPT_TOKEN_BUFFER", 3000))
# Ratio of MAX_CONTEXT_TOKENS allocated to file content vs other context items
CONTEXT_FILE_BUDGET_RATIO = float(os.getenv("CONTEXT_FILE_BUDGET_RATIO", 0.9)) # 90% for files

# --- Package Identification Settings (Step 2) ---
# Max estimated tokens for a single package's content/interface (used for splitting large clusters)
MAX_PACKAGE_SIZE_TOKENS = int(os.getenv("MAX_PACKAGE_SIZE_TOKENS", 20_000))
# Minimum number of files for a cluster to be considered a valid final package
MIN_PACKAGE_SIZE_FILES = int(os.getenv("MIN_PACKAGE_SIZE_FILES", 3))
# Ratio of LLM's context window to use for description/evaluation calls
LLM_DESC_MAX_TOKENS_RATIO = float(os.getenv("LLM_DESC_MAX_TOKENS_RATIO", 0.75))
# Maximum number of iterations for the package merging loop
MAX_MERGE_ITERATIONS = int(os.getenv("MAX_MERGE_ITERATIONS", 100))
# Minimum score (based on graph connectivity) for a merge candidate pair to be considered
MERGE_SCORE_THRESHOLD = float(os.getenv("MERGE_SCORE_THRESHOLD", 0.7))

# --- Generation Settings ---
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.9))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.98)) # Example, adjust as needed
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10)) # Example, adjust as needed

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
