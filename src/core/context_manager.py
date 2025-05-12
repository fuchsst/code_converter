# src/core/context_manager.py
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union
from typing import TYPE_CHECKING # Import for type hinting StateManager
from src.logger_setup import get_logger
import src.config as config
import tiktoken
import glob

logger = get_logger(__name__)

PACKAGES_JSON_FILENAME = "packages.json" # Define filename constant
STRUCTURE_ARTIFACT_SUFFIX = "_structure.json"
MAPPING_ARTIFACT_SUFFIX = "_mapping.json"


# --- Tiktoken Initializer ---
# Attempt to get a default tokenizer. cl100k_base is common for GPT-4/3.5/Embedding models
try:
    # We could use specific model names from config to appropriate tokenizers if needed
    # For now, use a common default.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logger.info("Tiktoken tokenizer 'cl100k_base' initialized for token counting.")
except Exception as e:
    logger.error(f"Failed to initialize tiktoken tokenizer: {e}. Token counting might be inaccurate.", exc_info=True)
    tokenizer = None



# --- File Reading Utility ---

def read_file_content(file_path: str, remove_comments_blank_lines: bool = True) -> str | None:
    """
    Reads content from a file, handling potential encoding issues.
    Optionally removes C-style comments and blank lines for C/C++ files.
    """
    logger.debug(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove comments/blank lines only for C/C++ files if requested
        if remove_comments_blank_lines and file_path.lower().endswith(('.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx')):
            original_len = len(content)
            # Remove C-style block comments (/* ... */) - non-greedy
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            # Remove C++ style line comments (// ...)
            content = re.sub(r'//.*', '', content)

            # Process lines: strip, remove empty lines, condense internal whitespace
            processed_lines = []
            for line in content.splitlines():
                stripped_line = line.strip() # Remove leading/trailing whitespace
                if stripped_line: # Only process non-blank lines
                    # Condense multiple internal whitespace characters into a single space
                    condensed_line = re.sub(r'\s+', ' ', stripped_line)
                    processed_lines.append(condensed_line)

            content = "\n".join(processed_lines) # Join the processed, non-blank, condensed lines

            final_len = len(content)
            #if original_len != final_len: # Log only if changes were made
            #    logger.debug(f"Removed comments/blank lines & condensed whitespace in {os.path.basename(file_path)} (Length: {original_len} -> {final_len})")

        return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return None

# --- Token Counting Utility ---

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    """
    Counts tokens using tiktoken. Uses cl100k_base as default.
    Falls back to character count if tiktoken is unavailable or fails.

    Args:
        text (str): The text to count tokens for.
        model_name (str): Optional model name hint (currently unused, defaults to cl100k_base).

    Returns:
        int: The estimated token count.
    """
    if not text:
        return 0

    if tokenizer:
        try:
            # Using the default initialized tokenizer (cl100k_base)
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            # logger.debug(f"Counted tokens using tiktoken ({model_name} -> cl100k_base): {token_count}") # Kept debug log for reference
            return token_count
        except Exception as e:
            logger.error(f"Tiktoken encode failed: {e}. Falling back to character count.", exc_info=True)
            # Fallback to character count
            token_count = len(text)
            logger.debug(f"Using fallback character count: {token_count}")
            return token_count
    else:
        # Fallback if tokenizer failed to initialize
        token_count = len(text)
        logger.warning(f"Tiktoken tokenizer not available. Using character count: {token_count}")
        return token_count

def check_token_limit(current_tokens: int, max_tokens: int) -> bool:
     """Checks if current tokens are within the limit."""
     is_within_limit = current_tokens < max_tokens
     # logger.debug(f"Checking token limit: {current_tokens} < {max_tokens} -> {is_within_limit}") # Can be verbose
     return is_within_limit

# --- Godot File Reading Utility ---

def read_godot_file_content(file_path: str) -> str | None:
    """
    Reads content from a Godot file (.gd, .tscn, .tres), handling encoding issues,
    and removing comments (#) and blank lines.
    """
    logger.debug(f"Reading Godot file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_len = len(content)
        # Remove Godot script comments (# ...)
        content = re.sub(r'#.*', '', content)

        # Process lines: strip, remove empty lines
        processed_lines = []
        for line in content.splitlines():
            stripped_line = line.strip() # Remove leading/trailing whitespace
            if stripped_line: # Only process non-blank lines
                processed_lines.append(stripped_line)

        content = "\n".join(processed_lines) # Join the processed, non-blank lines

        final_len = len(content)
        if original_len != final_len: # Log only if changes were made
            logger.debug(f"Removed comments/blank lines in {os.path.basename(file_path)} (Length: {original_len} -> {final_len})")

        return content
    except FileNotFoundError:
        logger.error(f"Godot file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading Godot file {file_path}: {e}", exc_info=True)
        return None


# --- ContextManager Class ---

# Forward reference for type hint if StateManager is imported later or causes circular dependency
if TYPE_CHECKING:
    from .state_manager import StateManager

class ContextManager:
    def __init__(self,
                 cpp_source_dir: str,
                 godot_project_dir: str,
                 analysis_output_dir: str,
                 instruction_dir: Optional[str],
                 state_manager: 'StateManager'):
        """
        Initializes the ContextManager.

        Args:
            cpp_source_dir (str): Path to the root of the C++ source code.
            godot_project_dir (str): Path to the root of the Godot project output.
            analysis_output_dir (str): Path to the directory for analysis output files.
            instruction_dir (Optional[str]): Path to the directory containing instruction files.
            state_manager (StateManager): Instance of the StateManager to access package states.
        """
        self.cpp_source_dir = Path(cpp_source_dir).resolve()
        self.godot_project_dir = Path(godot_project_dir).resolve()
        self.analysis_dir = Path(analysis_output_dir).resolve()
        self.instruction_dir = Path(instruction_dir).resolve() if instruction_dir else None
        self.state_manager = state_manager

        # Load include graph via StateManager artifact loading
        self.include_graph = self.state_manager.load_artifact("dependencies.json", expect_json=True)
        if not self.include_graph:
             logger.warning("Include graph (dependencies.json) failed to load via StateManager or is empty.")
        else:
             logger.info(f"Context Manager initialized. Loaded include graph ({len(self.include_graph)} entries) via StateManager.")

    # --- Context Retrieval Methods ---

    def get_instruction_context(self) -> str:
        """Loads and concatenates content from files in the instruction directory."""
        instruction_context_str = ""
        if self.instruction_dir and self.instruction_dir.is_dir():
            logger.info(f"Reading instruction files from: {self.instruction_dir}")
            instruction_parts = []
            try:
                for instruction_file in sorted(self.instruction_dir.iterdir()): # Sort for consistent order
                    if instruction_file.is_file():
                        # Use the utility function, don't remove comments/blanks from instructions
                        content = read_file_content(str(instruction_file), remove_comments_blank_lines=False)
                        if content:
                            instruction_parts.append(f"--- Instruction File: {instruction_file.name} ---\n{content}")
                        else:
                            logger.warning(f"Could not read instruction file or it was empty: {instruction_file}")
                if instruction_parts:
                    instruction_context_str = "\n\n".join(instruction_parts)
                    logger.info(f"Loaded instruction context ({len(instruction_context_str)} chars).")
                else:
                    logger.info("Instruction directory exists but contains no readable files.")
            except Exception as e:
                logger.error(f"Error reading instruction files from {self.instruction_dir}: {e}", exc_info=True)
        elif self.instruction_dir:
            logger.warning(f"Instruction directory specified but not found or not a directory: {self.instruction_dir}")
        else:
            logger.info("No INSTRUCTION_DIR configured. Skipping instruction context loading.")
        return instruction_context_str

    def get_source_file_list(self, package_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the list of source files and their roles for a specific package.

        Args:
            package_id (str): The ID of the package.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each with 'file_path' and 'role'.
                                  Returns empty list if package or roles not found.
        """
        pkg_info = self.state_manager.get_package_info(package_id)
        if not pkg_info:
            logger.warning(f"Package info not found for '{package_id}' in StateManager.")
            return []

        file_roles_list = pkg_info.get("file_roles", [])
        result_list = []
        if isinstance(file_roles_list, list):
            for role_info in file_roles_list:
                if isinstance(role_info, dict) and "file_path" in role_info and "role" in role_info:
                    result_list.append({
                        "file_path": role_info["file_path"],
                        "role": role_info["role"]
                    })
                else:
                    logger.warning(f"Invalid file_role format in package {package_id}: {role_info}")
        else:
            logger.warning(f"Missing or invalid 'file_roles' list in package {package_id}.")

        logger.debug(f"Retrieved {len(result_list)} source files/roles for package '{package_id}'.")
        return result_list

    def get_target_file_list(self, package_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the list of target Godot files defined for a package,
        including their purpose and existence status.

        Args:
            package_id (str): The ID of the package.

        Returns:
            List[Dict[str, Any]]: List of dicts with 'path', 'purpose', 'exists' (bool).
                                  Returns empty list if structure artifact not found/parsable.
        """
        structure_artifact_name = f"package_{package_id}{STRUCTURE_ARTIFACT_SUFFIX}"
        structure_data = self.state_manager.load_artifact(structure_artifact_name, expect_json=True)

        if not structure_data or not isinstance(structure_data, dict):
            logger.warning(f"Could not load or parse structure artifact '{structure_artifact_name}' for package '{package_id}'.")
            return []

        target_files = []
        defined_files_details = []

        # Extract details from structure data
        for key in ["scenes", "scripts", "resources", "migration_scripts"]:
            items = structure_data.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "path" in item and "purpose" in item:
                        defined_files_details.append({
                            "path": item["path"],
                            "purpose": item["purpose"]
                        })
                    # else: logger.warning(f"Invalid item format in '{key}' list for package {package_id}: {item}")

        # Check existence in Godot project dir
        for file_detail in defined_files_details:
            godot_path_str = file_detail["path"]
            purpose = file_detail["purpose"]
            exists = False
            absolute_path = None
            if godot_path_str.startswith("res://"):
                relative_path = godot_path_str[len("res://"):]
                absolute_path = self.godot_project_dir / relative_path
            elif godot_path_str: # If it's not empty and doesn't start with res://
                logger.info(f"Assuming '{godot_path_str}' is a relative path from project root for package '{package_id}'.")
                absolute_path = self.godot_project_dir / godot_path_str
            
            if absolute_path:
                exists = absolute_path.is_file()
                if not exists:
                    logger.debug(f"Target file '{godot_path_str}' (resolved to '{absolute_path}') does not exist.")
            else:
                logger.warning(f"Could not resolve absolute path for target file '{godot_path_str}' in package '{package_id}'.")

            target_files.append({
                "path": godot_path_str,
                "purpose": purpose,
                "exists": exists
            })

        logger.debug(f"Retrieved {len(target_files)} target file details for package '{package_id}'.")
        return target_files

    def get_work_package_source_code_content(self, package_id: str, max_tokens: Optional[int] = None) -> str:
        """
        Retrieves and formats C++ source code content for a given package ID,
        respecting token limits.

        Args:
            package_id (str): The ID of the package.
            max_tokens (Optional[int]): Maximum tokens for the combined content. Defaults to config.

        Returns:
            str: A formatted string containing the source code, or empty string if no files/content.
        """
        if max_tokens is None:
            max_tokens = config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER # Default limit
            logger.debug(f"Using default max_tokens for source code content: {max_tokens}")

        pkg_info = self.state_manager.get_package_info(package_id)
        if not pkg_info:
            logger.warning(f"Package info not found for '{package_id}' in StateManager.")
            return ""

        source_files = pkg_info.get("files", []) # Get the list of relative paths
        if not isinstance(source_files, list) or not source_files:
            logger.warning(f"No source files found for package '{package_id}'.")
            return ""

        logger.info(f"Retrieving source code content for package '{package_id}' ({len(source_files)} files, max_tokens={max_tokens}).")

        # Use _get_contextual_content to read files respecting token limits
        # Need to re-implement or adapt _get_contextual_content logic here or call a helper
        content_map = self._get_cpp_content_for_paths(relative_paths=source_files, max_tokens=max_tokens)

        # Format the content map into a single string
        formatted_parts = []
        for file_path, content in content_map.items():
            lang_hint = os.path.splitext(file_path)[1].lstrip('.') or "cpp"
            formatted_parts.append(f"// File: {file_path}\n```{lang_hint}\n{content}\n```")

        final_content = "\n\n".join(formatted_parts)
        final_tokens = count_tokens(final_content)
        logger.info(f"Assembled source code content for package '{package_id}' ({len(content_map)} files included, {final_tokens} tokens).")
        return final_content

    def get_work_package_target_code_content(self, package_id: str, max_tokens: Optional[int] = None) -> str:
        """
        Retrieves and formats EXISTING Godot code/scene/resource content
        defined for a given package ID, respecting token limits.

        Args:
            package_id (str): The ID of the package.
            max_tokens (Optional[int]): Maximum tokens for the combined content. Defaults to config.

        Returns:
            str: A formatted string containing the existing target code, or empty string.
        """
        if max_tokens is None:
            max_tokens = config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER # Default limit
            logger.debug(f"Using default max_tokens for target code content: {max_tokens}")

        target_files_details = self.get_target_file_list(package_id)
        if not target_files_details:
            logger.info(f"No target files defined or structure artifact missing for package '{package_id}'.")
            return ""

        existing_target_files = [f["path"] for f in target_files_details if f["exists"]]
        if not existing_target_files:
            logger.info(f"No existing target files found for package '{package_id}'.")
            return ""

        logger.info(f"Retrieving existing target code content for package '{package_id}' ({len(existing_target_files)} files, max_tokens={max_tokens}).")

        content_map = {}
        current_tokens = 0
        processed_paths = set()

        for res_path in existing_target_files:
            if res_path in processed_paths: continue

            if not res_path.startswith("res://"):
                logger.warning(f"Target file path '{res_path}' does not start with 'res://'. Skipping.")
                processed_paths.add(res_path)
                continue

            relative_path = res_path[len("res://"):]
            absolute_path = self.godot_project_dir / relative_path

            try:
                # Use the Godot-specific reader utility
                file_content = read_godot_file_content(str(absolute_path))
                if file_content is None:
                    processed_paths.add(res_path)
                    continue

                content_tokens = count_tokens(file_content)
                buffer = 50 # Small buffer per item

                if check_token_limit(current_tokens + content_tokens + buffer, max_tokens):
                    content_map[res_path] = file_content # Use res:// path as key
                    current_tokens += content_tokens
                    processed_paths.add(res_path)
                    logger.debug(f"Added EXISTING target content from {res_path} ({content_tokens} tokens). Total: {current_tokens}")
                else:
                    logger.warning(f"Skipping EXISTING target content from {res_path} due to token limit.")
                    processed_paths.add(res_path)
                    break # Stop if limit reached

            except Exception as e:
                logger.error(f"Error processing existing target file {absolute_path}: {e}", exc_info=True)
                processed_paths.add(res_path)

        # Format the content map into a single string
        formatted_parts = []
        for file_path, content in content_map.items():
            lang_hint = os.path.splitext(file_path)[1].lstrip('.') or "text"
            formatted_parts.append(f"// Existing File: {file_path}\n```{lang_hint}\n{content}\n```")

        final_content = "\n\n".join(formatted_parts)
        final_tokens = count_tokens(final_content)
        logger.info(f"Assembled existing target code content for package '{package_id}' ({len(content_map)} files included, {final_tokens} tokens).")
        return final_content

    # --- Internal Helper Methods ---

    def _get_cpp_content_for_paths(self, relative_paths: list[str], max_tokens: int) -> Dict[str, str]:
        """Internal helper to retrieve C++ file content, respecting token limits."""
        content_map = {}
        current_tokens = 0
        processed_paths = set()
        logger.debug(f"Internal: Retrieving C++ content for {len(relative_paths)} paths. Max tokens: {max_tokens}")

        for rel_path in relative_paths:
            if rel_path in processed_paths: continue
            abs_path_obj = self.cpp_source_dir / rel_path
            abs_path_str = str(abs_path_obj)
            logger.debug(f"Attempting to read C++ file: base_dir='{self.cpp_source_dir}', rel_path='{rel_path}', constructed_abs_path='{abs_path_str}'")

            if not abs_path_obj.exists():
                logger.warning(f"File not found at constructed path: {abs_path_str}")
                processed_paths.add(rel_path)
                continue

            try:
                file_content = read_file_content(abs_path_str) # Uses the utility with comment removal etc.
                if file_content is None:
                    processed_paths.add(rel_path)
                    continue

                content_tokens = count_tokens(file_content)
                buffer = 50

                if check_token_limit(current_tokens + content_tokens + buffer, max_tokens):
                    content_map[rel_path] = file_content
                    current_tokens += content_tokens
                    processed_paths.add(rel_path)
                    # logger.debug(f"Internal: Added C++ content from {rel_path} ({content_tokens} tokens). Total: {current_tokens}")
                else:
                    logger.warning(f"Internal: Skipping C++ content from {rel_path} due to token limit.")
                    processed_paths.add(rel_path)
                    break
            except Exception as e:
                logger.error(f"Internal: Error processing C++ file {abs_path_str}: {e}", exc_info=True)
                processed_paths.add(rel_path)

        logger.debug(f"Internal: Finished retrieving C++ content. {len(content_map)} files added. Total tokens: {current_tokens}")
        return content_map
