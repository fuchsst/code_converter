# src/core/context_manager.py
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union
from src.logger_setup import get_logger
import src.config as config
import tiktoken
# Import formatting utils
from src.utils.formatting_utils import (
    format_structure_to_markdown,
    format_packages_summary_to_markdown,
    format_existing_files_to_markdown
)

logger = get_logger(__name__)

PACKAGES_JSON_FILENAME = "packages.json" # Define filename constant

# --- Clang Imports ---
try:
    import clang.cindex
    # Ensure libclang path is configured if necessary
    # clang.cindex.Config.set_library_path(...)
except ImportError:
    print("Clang Python bindings not found. Please install 'clang'. Context extraction will be limited.")
    clang = None # Set clang to None to handle its absence gracefully
except Exception as e:
    print(f"Error initializing libclang: {e}. Context extraction might fail.")
    clang = None

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


# --- Clang Helper ---
# Define kinds of cursors we want to extract for interfaces
INTERFACE_CURSOR_KINDS = [
    clang.cindex.CursorKind.NAMESPACE,
    clang.cindex.CursorKind.CLASS_DECL,
    clang.cindex.CursorKind.STRUCT_DECL,
    clang.cindex.CursorKind.ENUM_DECL,
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.VAR_DECL, # For global/static variables
    clang.cindex.CursorKind.TYPEDEF_DECL,
    clang.cindex.CursorKind.TYPE_ALIAS_DECL,
    clang.cindex.CursorKind.CONSTRUCTOR,
    clang.cindex.CursorKind.DESTRUCTOR,
    clang.cindex.CursorKind.FIELD_DECL, # Inside classes/structs
    clang.cindex.CursorKind.ENUM_CONSTANT_DECL, # Inside enums
    # Add others if needed, e.g., USING_DECLARATION?
]

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
            if original_len != final_len: # Log only if changes were made
                logger.debug(f"Removed comments/blank lines & condensed whitespace in {os.path.basename(file_path)} (Length: {original_len} -> {final_len})")

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

class ContextManager:
    def __init__(self, include_graph_path: str, cpp_source_dir: str, analysis_output_dir: str):
        """
        Initializes the ContextManager.

        Args:
            include_graph_path (str): Path to the JSON file containing the include graph.
            cpp_source_dir (str): Path to the root of the C++ source code.
            analysis_output_dir (str): Path to the directory for analysis output files (like packages.json).
        """
        self.include_graph_path = include_graph_path
        self.cpp_source_dir = Path(cpp_source_dir).resolve() # Store as absolute Path
        self.analysis_dir = Path(analysis_output_dir).resolve() # Store as absolute Path
        self.packages_file_path = self.analysis_dir / PACKAGES_JSON_FILENAME

        self.include_graph = self._load_include_graph()
        self.packages_data, self.package_processing_order = self._load_packages_data()
        self.clang_index = None
        self.compile_db = None

        if clang:
            try:
                self.clang_index = clang.cindex.Index.create()
                logger.info("Clang index created successfully.")
                # Try to load compile_commands.json using Path
                compile_commands_path = self.cpp_source_dir / 'compile_commands.json'
                if compile_commands_path.exists():
                    try:
                        # Pass the directory path as a string
                        self.compile_db = clang.cindex.CompilationDatabase.fromDirectory(str(self.cpp_source_dir))
                        logger.info(f"Loaded compilation database from: {self.cpp_source_dir}")
                    except clang.cindex.LibclangError as db_err:
                        logger.error(f"Failed to load compilation database from {self.cpp_source_dir}: {db_err}")
                else:
                    logger.warning(f"compile_commands.json not found in {self.cpp_source_dir}. Interface extraction may be inaccurate.")
            except Exception as clang_init_err:
                logger.error(f"Failed to initialize Clang index or database: {clang_init_err}", exc_info=True)
                self.clang_index = None # Ensure it's None on error
                self.compile_db = None
        else:
            logger.warning("Clang library not available. C++ interface extraction disabled.")


        if not self.include_graph:
             logger.warning(f"Include graph at '{include_graph_path}' failed to load or is empty.")
        else:
             # Updated log message
             logger.info(f"Context Manager initialized. Loaded include graph ({len(self.include_graph)} entries). Loaded packages data ({len(self.packages_data)} packages).")

    def _load_include_graph(self):
        """Loads the include graph JSON file."""
        if not os.path.exists(self.include_graph_path):
             logger.error(f"Include graph file not found: {self.include_graph_path}")
             return {}
        try:
            with open(self.include_graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            logger.info(f"Successfully loaded include graph from: {self.include_graph_path}")
            # Basic validation: check if it's a dictionary
            if not isinstance(graph, dict):
                logger.error(f"Include graph file is not a valid JSON dictionary: {self.include_graph_path}")
                return {}
            return graph
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from include graph file: {self.include_graph_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load include graph: {e}", exc_info=True)
            return {}

    def _load_packages_data(self) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        """
        Loads the packages data and processing order from the JSON file.
        Returns a tuple: (packages_dict, processing_order_list | None).
        """
        packages_dict = {}
        processing_order = None
        if not self.packages_file_path.exists():
            logger.info(f"{self.packages_file_path} not found. Initializing empty package data and order.")
            return packages_dict, processing_order # Return empty dict and None if file doesn't exist

        try:
            with open(self.packages_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded existing package data from {self.packages_file_path}")

            # Validate overall structure
            if not isinstance(data, dict):
                logger.warning(f"Invalid structure in {self.packages_file_path}. Expected top-level dictionary. Initializing empty.")
                return {}, None

            # Load packages dictionary
            if 'packages' in data and isinstance(data['packages'], dict):
                 packages_dict = data['packages']
                 logger.debug(f"Loaded {len(packages_dict)} packages.")
            else:
                 logger.warning(f"Missing or invalid 'packages' key in {self.packages_file_path}. Initializing empty packages.")
                 packages_dict = {}

            # Load processing order list (optional)
            if 'processing_order' in data and isinstance(data['processing_order'], list):
                 # Further validation: check if all items are strings
                 if all(isinstance(item, str) for item in data['processing_order']):
                      processing_order = data['processing_order']
                      logger.debug(f"Loaded processing order with {len(processing_order)} packages.")
                 else:
                      logger.warning(f"Invalid items in 'processing_order' list in {self.packages_file_path}. Expected list of strings. Ignoring order.")
                      processing_order = None # Reset if invalid items found
            else:
                 logger.debug(f"'processing_order' key not found or invalid in {self.packages_file_path}. No processing order loaded.")
                 processing_order = None

            return packages_dict, processing_order

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.packages_file_path}: {e}. Initializing empty.")
            return {}, None
        except Exception as e:
            logger.error(f"Error loading {self.packages_file_path}: {e}. Initializing empty.", exc_info=True)
            return {}, None

    def save_packages_data(self, packages_data_to_save: Dict[str, Any], package_order: Optional[List[str]] = None):
        """
        Saves the provided packages data and optional processing order to the JSON file.
        """
        try:
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            # Create the top-level structure
            output_data = {
                "packages": packages_data_to_save,
                "processing_order": package_order # Will be null in JSON if None
            }
            with open(self.packages_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            logger.debug(f"Saved package data ({len(packages_data_to_save)} packages) and order ({len(package_order) if package_order else 'None'}) to {self.packages_file_path}")
            # Update internal state after successful save
            self.packages_data = packages_data_to_save
            self.package_processing_order = package_order
        except Exception as e:
            logger.error(f"Failed to save package data to {self.packages_file_path}: {e}", exc_info=True)

    def get_package_order(self) -> Optional[List[str]]:
        """Returns the loaded package processing order, if available."""
        return self.package_processing_order

    def get_all_package_summaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves a summary (description, files) for all packages.
        Reloads data from file to ensure freshness.
        """
        packages_dict, _ = self._load_packages_data() # Reload data
        summaries = {}
        for pkg_id, pkg_data in packages_dict.items():
            files_with_roles = {}
            file_roles_list = pkg_data.get("file_roles", [])
            if isinstance(file_roles_list, list):
                for role_info in file_roles_list:
                    if isinstance(role_info, dict) and "file_path" in role_info and "role" in role_info:
                        files_with_roles[role_info["file_path"]] = role_info["role"]
                    else:
                        logger.warning(f"Invalid file_role format in package {pkg_id}: {role_info}")
            else:
                 logger.warning(f"Missing or invalid 'file_roles' list in package {pkg_id}. Files summary will not contain roles.")


            summaries[pkg_id] = {
                "description": pkg_data.get("description", "N/A"),
                "files": files_with_roles # Use the dictionary with roles
            }
        logger.debug(f"Retrieved summaries with file roles for {len(summaries)} packages.")
        return summaries

    def get_existing_structure(self, package_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads the existing Godot structure JSON for a specific package, if it exists.
        """
        structure_filename = f"package_{package_id}_structure.json"
        structure_path = self.analysis_dir / structure_filename
        if structure_path.exists():
            try:
                with open(structure_path, 'r', encoding='utf-8') as f:
                    structure_data = json.load(f)
                logger.debug(f"Loaded existing structure for package {package_id} from {structure_path}")
                return structure_data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load or parse existing structure file {structure_path}: {e}", exc_info=True)
                return None
            except Exception as e:
                 logger.error(f"Unexpected error loading existing structure file {structure_path}: {e}", exc_info=True)
                 return None
        else:
            logger.debug(f"No existing structure file found for package {package_id} at {structure_path}")
            return None

    def get_all_existing_godot_files(self) -> Dict[str, List[str]]:
        """
        Scans the analysis directory for all package_*.structure.json files
        and compiles a dictionary mapping package IDs to their defined Godot scenes and scripts.
        """
        all_godot_files = {}
        try:
            if not self.analysis_dir.exists():
                logger.warning(f"Analysis directory {self.analysis_dir} does not exist. Cannot scan for structure files.")
                return {}

            for item in self.analysis_dir.iterdir():
                if item.is_file() and item.name.startswith("package_") and item.name.endswith("_structure.json"):
                    match = re.match(r"package_(.+)_structure\.json", item.name)
                    if match:
                        pkg_id = match.group(1)
                        try:
                            with open(item, 'r', encoding='utf-8') as f:
                                structure_data = json.load(f)

                            package_files = []
                            # Extract scene paths
                            scenes = structure_data.get("scenes", [])
                            if isinstance(scenes, list):
                                for scene in scenes:
                                    if isinstance(scene, dict) and "path" in scene:
                                        package_files.append(scene["path"])

                            # Extract script paths
                            scripts = structure_data.get("scripts", [])
                            if isinstance(scripts, list):
                                for script in scripts:
                                    if isinstance(script, dict) and "path" in script:
                                        package_files.append(script["path"])

                            if package_files:
                                all_godot_files[pkg_id] = sorted(list(set(package_files))) # Store unique, sorted list
                                logger.debug(f"Found {len(package_files)} Godot files for package {pkg_id} in {item.name}")

                        except (json.JSONDecodeError, IOError) as e:
                            logger.error(f"Failed to load or parse structure file {item}: {e}")
                        except Exception as e:
                            logger.error(f"Unexpected error processing structure file {item}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error scanning analysis directory {self.analysis_dir} for structure files: {e}", exc_info=True)

        logger.info(f"Compiled existing Godot files from {len(all_godot_files)} packages based on structure definitions.")
        return all_godot_files

    def get_existing_godot_output_files(self, godot_project_dir: str) -> List[str]:
        """
        Scans the actual Godot project output directory for existing relevant files.

        Args:
            godot_project_dir (str): The absolute path to the Godot project directory.

        Returns:
            List[str]: A list of relative paths (from godot_project_dir) of existing
                       .gd, .tscn, .tres, .shader, .py files.
        """
        godot_dir = Path(godot_project_dir).resolve()
        if not godot_dir.is_dir():
            logger.warning(f"Godot project directory not found or not a directory: {godot_dir}. Cannot scan for existing output files.")
            return []

        logger.info(f"Scanning Godot project directory for existing output files: {godot_dir}")
        existing_files = []
        relevant_extensions = {".gd", ".tscn", ".tres", ".shader", ".py"} # Add other relevant types if needed

        try:
            for item in godot_dir.rglob('*'): # Recursive glob
                if item.is_file() and item.suffix.lower() in relevant_extensions:
                    # Calculate relative path from the godot_project_dir
                    try:
                        relative_path = item.relative_to(godot_dir).as_posix() # Use POSIX paths for consistency
                        # Prepend 'res://' to match Godot's convention
                        res_path = f"res://{relative_path}"
                        existing_files.append(res_path)
                    except ValueError:
                        # Should not happen if rglob starts within godot_dir, but handle defensively
                        logger.warning(f"Could not determine relative path for {item} within {godot_dir}")

        except Exception as e:
            logger.error(f"Error scanning Godot project directory {godot_dir}: {e}", exc_info=True)

        logger.info(f"Found {len(existing_files)} existing Godot output files in {godot_dir}.")
        return sorted(existing_files)

    def read_godot_file_content(self, godot_project_dir: str, file_path: str) -> str | None:
        """
        Reads content from a Godot file (.gd, .tscn, .tres), handling encoding issues,
        and removing comments (#) and blank lines. Delegates to the utility function.

        Args:
            file_path (str): Absolute or relative path to the Godot file.
                             If relative, it's assumed relative to the CWD or needs context.
                             It's safer to pass absolute paths or paths relative to godot_project_dir.

        Returns:
            str | None: The cleaned content or None if reading fails.
        """
        # Note: This method currently just wraps the utility function.
        return read_godot_file_content(os.path.join(godot_project_dir, file_path))


    def _get_contextual_content(self,
                                relative_paths: list[str],
                                max_tokens: int):
        """
        Retrieves full content for specified files, respecting token limits.
        Args:
            relative_paths (list[str]): Files needing full content.
            max_tokens (int): The approximate maximum tokens allowed for the combined content.

        Returns:
            dict: A dictionary mapping relative paths to their full content.
                  Stops adding files if max_tokens is exceeded.
        """
        content_map = {}
        current_tokens = 0
        processed_paths = set() # Avoid processing the same file twice

        logger.debug(f"Retrieving contextual content for {len(relative_paths)} files. Max tokens: {max_tokens}")

        # Process all files, attempting to add full content within token limits
        for rel_path in relative_paths:
            # Paths are assumed to be normalized with '/' already
            if rel_path in processed_paths: continue

            # Construct absolute path using pathlib
            abs_path_obj = self.cpp_source_dir / rel_path
            abs_path_str = str(abs_path_obj) # Convert to string for functions expecting strings

            if not abs_path_obj.exists():
                logger.warning(f"File not found: {abs_path_str}")
                processed_paths.add(rel_path) # Mark as processed even if not found
                continue

            try:
                # Always read full content now
                file_content = read_file_content(abs_path_str)
                if file_content is None:
                    processed_paths.add(rel_path) # Mark as processed even if content is None/error
                    continue

                content_tokens = count_tokens(file_content)
                buffer = 50 # Small buffer per item

                if check_token_limit(current_tokens + content_tokens + buffer, max_tokens):
                    # Use rel_path as the key consistently
                    content_map[rel_path] = file_content
                    current_tokens += content_tokens
                    processed_paths.add(rel_path)
                    logger.debug(f"Added FULL content from {rel_path} ({content_tokens} tokens). Total: {current_tokens}")
                else:
                    logger.warning(f"Skipping FULL content from {rel_path} due to token limit ({current_tokens}+{content_tokens}+{buffer} > {max_tokens}).")
                    # Mark as processed even if skipped due to limit
                    processed_paths.add(rel_path)
                    # Optimization: If we exceed the limit, stop processing further files for this context assembly.
                    # This avoids unnecessary reading/tokenizing for large packages.
                    logger.info(f"Token limit reached ({max_tokens}). Stopping further file processing for this context assembly.")
                    break # Stop iterating through relative_paths

            except Exception as e:
                logger.error(f"Error processing file {abs_path_str}: {e}", exc_info=True)
                processed_paths.add(rel_path) # Mark as processed even on error

        logger.info(f"Finished retrieving contextual content. {len(content_map)} files added. Total tokens: {current_tokens}")
        return content_map

    def _assemble_context_block(self, file_content_map: dict, other_context: dict, max_total_tokens: int) -> str:
        """
        Assembles context from file contents and other provided context items
        into a single string using Markdown formatting, respecting token limits.
        Prioritizes 'other_context' first, then file contents.

        Args:
            file_content_map (dict): Dictionary mapping relative file paths to their full content.
            other_context (dict): Dictionary of other context items (e.g., task descriptions, JSON data).
            max_total_tokens (int): The overall token limit for the final context string.

        Returns:
            str: The assembled context string formatted with Markdown.
        """
        final_context_parts = []
        current_total_tokens = 0
        # Estimate Markdown formatting tokens (```cpp\n ... \n```\n\n) per block
        markdown_overhead_per_block = count_tokens("\n\n```cpp\n\n```\n\n") + 20 # Extra buffer

        logger.debug(f"Assembling context block with max_total_tokens: {max_total_tokens}")

        # 1. Add 'instruction_context' first if provided
        instruction_context = other_context.pop('instruction_context', None) # Extract and remove from dict
        if instruction_context and isinstance(instruction_context, str):
            # Format instructions clearly
            formatted_instructions = f"**Instructions:**\n```text\n{instruction_context}\n```"
            component_tokens = count_tokens(formatted_instructions)
            separator_tokens = count_tokens("\n\n")

            if check_token_limit(current_total_tokens + component_tokens + separator_tokens, max_total_tokens):
                final_context_parts.append(formatted_instructions)
                current_total_tokens += component_tokens + separator_tokens
                logger.debug(f"Added instruction context ({component_tokens} tokens). Total tokens: {current_total_tokens}")
            else:
                logger.warning("Skipping instruction context due to token limit.")
        elif instruction_context:
             logger.warning(f"instruction_context provided but was not a string (type: {type(instruction_context)}). Skipping.")


        # 2. Add remaining 'other_context' items (e.g., task descriptions, JSON data)
        for key, content in other_context.items():
            # Skip if key was already handled or content is empty
            if not content or key == 'instruction_context':
                logger.debug(f"Skipping other context item: {key} (empty or already handled)")
                continue

            # Format based on content type (simple text vs. JSON vs. specific keys)
            title = key.replace('_', ' ').title()

            # Handle specific keys with custom formatting using utility functions
            if key == "proposed_godot_structure_md":
                 # Already formatted by utility, just add title/wrapper if needed
                 # The utility function already adds a title, so just use the content.
                 formatted_content = f"**Proposed Godot Structure:**\n{str(content)}"
            elif key == "global_packages_summary":
                 # Use the dedicated formatter
                 formatted_content = format_packages_summary_to_markdown(content)
            elif key == "existing_godot_outputs":
                 # Use the dedicated formatter, pass the title derived from the key
                 formatted_content = format_existing_files_to_markdown(content, title=title)
            elif isinstance(content, (dict, list)):
                 # Default JSON formatting for other dicts/lists
                 formatted_content = f"**{title} (JSON):**\n```json\n{json.dumps(content, indent=2)}\n```"
            elif isinstance(content, str) and content.strip().startswith(('{', '[')): # Basic check for JSON string
                 # Try to pretty-print if it looks like a JSON string
                 try:
                      parsed_json = json.loads(content)
                      formatted_content = f"**{title} (JSON String):**\n```json\n{json.dumps(parsed_json, indent=2)}\n```"
                 except json.JSONDecodeError:
                      formatted_content = f"**{title}:**\n```text\n{content}\n```" # Treat as plain text block
            else:
                 formatted_content = f"**{title}:**\n{str(content)}" # Treat as simple text

            component_tokens = count_tokens(formatted_content)
            separator_tokens = count_tokens("\n\n") # Separator between context items

            # Check limit before adding
            if check_token_limit(current_total_tokens + component_tokens + separator_tokens, max_total_tokens):
                final_context_parts.append(formatted_content)
                current_total_tokens += component_tokens + separator_tokens
                logger.debug(f"Added other context '{key}' ({component_tokens} tokens). Total tokens: {current_total_tokens}")
            else:
                logger.warning(f"Skipping other context '{key}' due to token limit.")
                # Don't break; try adding files or smaller context items

        # 3. Add file contents using Markdown code blocks
        for file_path, content in file_content_map.items():
            if not content:
                logger.debug(f"Skipping empty file content block: {file_path}")
                continue

            # Determine language hint for Markdown block
            lang_hint = os.path.splitext(file_path)[1].lstrip('.')
            if not lang_hint: lang_hint = "text" # Default if no extension

            # Format as Markdown code block
            block = f"**File:** `{file_path}`\n```{lang_hint}\n{content}\n```"
            block_tokens = count_tokens(block) # Count tokens of the formatted block

            # Check if the formatted block fits within the remaining limit
            if check_token_limit(current_total_tokens + block_tokens, max_total_tokens):
                final_context_parts.append(block)
                current_total_tokens += block_tokens + count_tokens("\n\n") # Add tokens for block and separator
                logger.debug(f"Added file content block for '{file_path}' ({block_tokens} tokens). Total tokens now: {current_total_tokens}")
            else:
                logger.warning(f"Skipping file content block for '{file_path}' due to total token limit ({current_total_tokens} + {block_tokens} > {max_total_tokens}).")
                # Stop adding more files if limit is reached
                break

        return "\n\n".join(final_context_parts)

    def _get_dependencies_for_package(self, package_relative_paths: list[str]) -> list[str]:
        """
        Finds direct dependencies for a list of package files using the include graph.

        Args:
            package_relative_paths (list[str]): List of relative file paths belonging to the package.

        Returns:
            list[str]: A list of unique relative paths of files directly included by the package files,
                       excluding the package files themselves and ensuring they are within the project.
        """
        if not self.include_graph:
            logger.warning("Include graph not loaded. Cannot determine dependencies.")
            return []

        all_dependencies = set()
        package_files_set = set(p.replace('\\', '/') for p in package_relative_paths) # Normalize paths

        for pkg_file_rel in package_files_set:
            # Ensure the file exists in the graph keys (normalize just in case)
            normalized_pkg_file = pkg_file_rel.replace('\\', '/')
            if normalized_pkg_file in self.include_graph:
                # Get the list of files included by this package file
                direct_includes = self.include_graph[normalized_pkg_file] # This is expected to be a list of dicts like {'path': '...', 'weight': ...}
                if isinstance(direct_includes, list):
                    for include_item in direct_includes:
                        # Check if the item is a dictionary with a 'path' key
                        if isinstance(include_item, dict) and 'path' in include_item:
                            included_file_path = include_item['path']
                            if not isinstance(included_file_path, str):
                                logger.warning(f"Include item for '{normalized_pkg_file}' has non-string path: {include_item}. Skipping.")
                                continue
                            normalized_include = included_file_path.replace('\\', '/')
                        elif isinstance(include_item, str):
                            # Handle potential older format where it might just be a string (less likely based on Step 2)
                            logger.debug(f"Include item for '{normalized_pkg_file}' is a string (unexpected format): {include_item}. Processing anyway.")
                            normalized_include = include_item.replace('\\', '/')
                        else:
                            logger.warning(f"Unexpected include item format for '{normalized_pkg_file}': {include_item}. Skipping.")
                            continue

                        # Add to dependencies only if it's not part of the original package
                        if normalized_include not in package_files_set:
                            # We assume the include graph already filtered for project files
                            all_dependencies.add(normalized_include)
                else:
                    logger.warning(f"Include graph entry for '{normalized_pkg_file}' is not a list: {direct_includes}")
            # else:
                # logger.debug(f"Package file '{normalized_pkg_file}' not found as a key in the include graph.")

        logger.debug(f"Found {len(all_dependencies)} unique dependencies for package files: {package_relative_paths}")
        return sorted(list(all_dependencies))

    # --- Method for assembling context for specific steps ---

    def get_context_for_step(self, step_name: str, **kwargs) -> str:
        """
        Assembles context for a given workflow step, retrieving full content for all specified files.

        Args:
            step_name (str): Identifier for the workflow step (e.g., "STRUCTURE_DEFINITION").
            primary_relative_paths (list[str]): List of primary file paths for the step.
            dependency_relative_paths (list[str]): List of dependency file paths for the step.
            **kwargs: Additional context items (e.g., 'task_description', 'work_package_info').

        Returns:
            str: The assembled context string formatted with Markdown, or empty string on error.
        """
        # Extract file path lists from kwargs, providing empty lists as defaults
        primary_relative_paths = kwargs.get('primary_relative_paths', [])
        dependency_relative_paths = kwargs.get('dependency_relative_paths', [])

        # Combine primary and dependency files into a single list of unique paths
        all_relative_paths = sorted(list(set(primary_relative_paths + dependency_relative_paths)))

        logger.info(f"Assembling context for STEP '{step_name}' ({len(all_relative_paths)} total files)")

        # --- Configuration ---
        # Use MAX_CONTEXT_TOKENS from config
        # Subtract a buffer for the prompt itself and potential LLM response overhead
        prompt_buffer = config.PROMPT_TOKEN_BUFFER
        max_total_tokens = config.MAX_CONTEXT_TOKENS
        max_context_assembly_tokens = max_total_tokens - prompt_buffer
        if max_context_assembly_tokens <= 0:
             logger.error(f"MAX_CONTEXT_TOKENS ({max_total_tokens}) is too small with buffer ({prompt_buffer}). Cannot assemble context.")
             return ""

        # --- Separate other context from file path args ---
        other_context = {k: v for k, v in kwargs.items() if k not in ['primary_relative_paths', 'dependency_relative_paths']}

        # --- Retrieve Contextual File Content ---
        # Allocate budget for file content (now always full content)
        # Use CONTEXT_FILE_BUDGET_RATIO from config
        file_content_budget = int(max_context_assembly_tokens * config.CONTEXT_FILE_BUDGET_RATIO)
        file_content_map = self._get_contextual_content(
            relative_paths=all_relative_paths,
            max_tokens=file_content_budget
        )

        if not file_content_map and all_relative_paths:
             logger.warning(f"Failed to retrieve content for any specified files for step '{step_name}'. Context might be incomplete.")
             # Proceed with only other_context if available, but log warning

        # --- Assemble Final Context Block ---
        # The assembly limit here applies to the final formatted string
        final_context = self._assemble_context_block(
            file_content_map=file_content_map,
            other_context=other_context,
            max_total_tokens=max_context_assembly_tokens
        )

        final_token_count = count_tokens(final_context)
        logger.info(f"Assembled context for STEP '{step_name}' ({final_token_count} tokens / {max_context_assembly_tokens} assembly limit).")

        if final_token_count >= max_context_assembly_tokens:
             logger.warning(f"Final assembled context for {step_name} might be exceeding token limit!")
        elif not final_context:
             logger.warning(f"Assembled context for {step_name} is empty.")


        return final_context
