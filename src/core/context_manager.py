# src/core/context_manager.py
import os
import json
import re
from logger_setup import get_logger
import config
import tiktoken # Added for token counting

logger = get_logger(__name__)

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
            # Remove blank lines (lines containing only whitespace)
            content = "\n".join(line for line in content.splitlines() if line.strip())
            final_len = len(content)
            if original_len != final_len: # Log only if changes were made
                logger.debug(f"Removed comments/blank lines from {os.path.basename(file_path)} (Length: {original_len} -> {final_len})")

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
            # TODO: Map model_name to specific tokenizer if needed in the future
            # For now, using the default initialized tokenizer
            tokens = tokenizer.encode(text)
            token_count = len(tokens)
            # logger.debug(f"Counted tokens using tiktoken ({model_name} -> cl100k_base): {token_count}")
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

# --- ContextManager Class ---

class ContextManager:
    def __init__(self, include_graph_path: str, cpp_source_dir: str):
        """
        Initializes the ContextManager.

        Args:
            include_graph_path (str): Path to the JSON file containing the include graph.
            cpp_source_dir (str): Path to the root of the C++ source code.
        """
        self.include_graph_path = include_graph_path
        self.cpp_source_dir = os.path.abspath(cpp_source_dir)
        self.include_graph = self._load_include_graph()
        self.clang_index = None
        self.compile_db = None

        if clang:
            try:
                self.clang_index = clang.cindex.Index.create()
                logger.info("Clang index created successfully.")
                # Try to load compile_commands.json
                compile_commands_dir = self.cpp_source_dir
                cc_path = os.path.join(compile_commands_dir, 'compile_commands.json')
                if os.path.exists(cc_path):
                    try:
                        self.compile_db = clang.cindex.CompilationDatabase.fromDirectory(compile_commands_dir)
                        logger.info(f"Loaded compilation database from: {compile_commands_dir}")
                    except clang.cindex.LibclangError as db_err:
                        logger.error(f"Failed to load compilation database from {compile_commands_dir}: {db_err}")
                else:
                    logger.warning(f"compile_commands.json not found in {compile_commands_dir}. Interface extraction may be inaccurate.")
            except Exception as clang_init_err:
                logger.error(f"Failed to initialize Clang index or database: {clang_init_err}", exc_info=True)
                self.clang_index = None # Ensure it's None on error
                self.compile_db = None
        else:
            logger.warning("Clang library not available. C++ interface extraction disabled.")


        if not self.include_graph:
             logger.warning(f"Include graph at '{include_graph_path}' failed to load or is empty.")
        else:
             logger.info(f"Context Manager initialized. Loaded include graph with {len(self.include_graph)} entries.")

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

    def _extract_cpp_interface(self, file_path_abs: str) -> str | None:
        """
        Parses a C++ file using libclang and extracts declarations/signatures.
        Returns the extracted interface as a string, or None on failure.
        """
        if not self.clang_index or not clang:
            logger.warning(f"Clang not available or not initialized. Cannot extract interface for {file_path_abs}.")
            return None

        logger.debug(f"Attempting to extract C++ interface from: {file_path_abs}")
        interface_parts = []
        try:
            args = []
            if self.compile_db:
                commands = self.compile_db.getCompileCommands(file_path_abs)
                if commands:
                    # Use the first command's arguments, excluding compiler and output flags
                    cmd_args = list(commands[0].arguments)[1:]
                    try:
                        o_idx = cmd_args.index('-o')
                        cmd_args.pop(o_idx) # Remove -o
                        cmd_args.pop(o_idx) # Remove output file path
                    except ValueError:
                        pass # -o not found
                    args.extend(cmd_args)
                    logger.debug(f"Using compile args for {os.path.basename(file_path_abs)}: {' '.join(args[:5])}...") # Log first few args
                else:
                    logger.debug(f"No specific compile commands found for {file_path_abs}. Parsing with default flags.")
            else:
                 logger.debug(f"No compile database loaded. Parsing {file_path_abs} with default flags.")


            # Parse the file, skipping function bodies for efficiency
            # Consider adding PARSE_DETAILED_PROCESSING_RECORD if macro handling is needed
            tu = self.clang_index.parse(
                file_path_abs,
                args=args,
                options=clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES |
                        clang.cindex.TranslationUnit.PARSE_INCOMPLETE # Try to parse even with errors
            )

            if not tu:
                logger.error(f"Clang failed to create translation unit for: {file_path_abs}")
                return None

            # Check for fatal parsing errors
            has_fatal_errors = any(d.severity >= clang.cindex.Diagnostic.Error for d in tu.diagnostics)
            if has_fatal_errors:
                logger.warning(f"Clang encountered errors parsing {file_path_abs}. Interface extraction might be incomplete.")
                # Log first few errors for debugging
                for i, diag in enumerate(tu.diagnostics):
                     if diag.severity >= clang.cindex.Diagnostic.Error:
                          logger.debug(f"  Error {i+1}: {diag.spelling} at {diag.location}")
                     if i > 2: break # Limit logged errors

            # Read the source file content once for extent extraction
            try:
                with open(file_path_abs, 'rb') as f: # Read as bytes for extent offsets
                    source_bytes = f.read()
            except Exception as read_err:
                 logger.error(f"Could not read source file {file_path_abs} for extent extraction: {read_err}")
                 return None # Cannot extract without source

            # Traverse the AST
            for cursor in tu.cursor.walk_preorder():
                # Only process cursors from the main file, not included headers
                if cursor.location.file and cursor.location.file.name == file_path_abs:
                    if cursor.kind in INTERFACE_CURSOR_KINDS:
                        # Get the source range (extent) of the cursor
                        extent = cursor.extent
                        start_offset = extent.start.offset
                        end_offset = extent.end.offset

                        # Extract the source code snippet using byte offsets
                        snippet_bytes = source_bytes[start_offset:end_offset]
                        try:
                            snippet = snippet_bytes.decode('utf-8', errors='ignore')
                            # Basic formatting: Add semicolon if it's likely missing (simple heuristic)
                            trimmed_snippet = snippet.strip()
                            if cursor.kind in [clang.cindex.CursorKind.FUNCTION_DECL,
                                               clang.cindex.CursorKind.CXX_METHOD,
                                               clang.cindex.CursorKind.CONSTRUCTOR,
                                               clang.cindex.CursorKind.DESTRUCTOR] and \
                               '{' not in trimmed_snippet and \
                               not trimmed_snippet.endswith(';'):
                                snippet += ';'

                            # Add newline for separation, handle nested elements later if needed
                            interface_parts.append(snippet + "\n")
                        except UnicodeDecodeError:
                             logger.warning(f"Could not decode snippet from {file_path_abs} at offset {start_offset}:{end_offset}")


            if not interface_parts:
                 logger.warning(f"No interface elements extracted from {file_path_abs}. File might be empty or only contain implementations.")
                 # Return empty string instead of None if parsing succeeded but found nothing
                 return ""

            # Join parts and add a header comment
            final_interface = f"// Extracted interface from: {os.path.basename(file_path_abs)}\n\n" + "".join(interface_parts)
            logger.debug(f"Successfully extracted interface from {file_path_abs} ({len(final_interface)} chars)")
            return final_interface

        except clang.cindex.LibclangError as e:
            logger.error(f"Libclang error extracting interface from {file_path_abs}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting interface from {file_path_abs}: {e}", exc_info=True)
            return None


    def _get_contextual_content(self,
                                primary_relative_paths: list[str],
                                dependency_relative_paths: list[str],
                                max_tokens: int):
        """
        Retrieves content for specified files, using full content for primary files
        and extracted interfaces for dependency files, respecting token limits.
        Args:
            primary_relative_paths (list[str]): Files needing full content.
            dependency_relative_paths (list[str]): Files needing only interface extraction.
            max_tokens (int): The approximate maximum tokens allowed for the combined content.

        Returns:
            dict: A dictionary mapping relative paths to their content (full or interface).
                  Stops adding files if max_tokens is exceeded.
        """
        content_map = {}
        current_tokens = 0
        processed_paths = set() # Avoid processing the same file twice

        logger.debug(f"Retrieving contextual content: {len(primary_relative_paths)} primary, {len(dependency_relative_paths)} dependency files. Max tokens: {max_tokens}")

        # 1. Process Primary Files (Full Content)
        for rel_path in primary_relative_paths:
            normalized_rel_path = rel_path.replace('\\', '/')
            if normalized_rel_path in processed_paths: continue

            abs_path = os.path.join(self.cpp_source_dir, normalized_rel_path)
            if not os.path.exists(abs_path):
                logger.warning(f"[Primary] File not found: {abs_path}")
                continue

            try:
                file_content = read_file_content(abs_path)
                if file_content is None: continue

                content_tokens = count_tokens(file_content)
                buffer = 50 # Small buffer per item

                if check_token_limit(current_tokens + content_tokens + buffer, max_tokens):
                    content_map[normalized_rel_path] = file_content
                    current_tokens += content_tokens
                    processed_paths.add(normalized_rel_path)
                    logger.debug(f"Added FULL content from {normalized_rel_path} ({content_tokens} tokens). Total: {current_tokens}")
                else:
                    logger.warning(f"Skipping FULL content from {normalized_rel_path} due to token limit ({current_tokens}+{content_tokens}+{buffer} > {max_tokens}).")
                    break # Stop adding primary files

            except Exception as e:
                logger.error(f"Error processing primary file {abs_path}: {e}", exc_info=True)

        # 2. Process Dependency Files (Interface Extraction)
        # Only proceed if token limit wasn't hit by primary files
        if current_tokens < max_tokens:
            for rel_path in dependency_relative_paths:
                normalized_rel_path = rel_path.replace('\\', '/')
                if normalized_rel_path in processed_paths: continue # Skip if already added as primary

                abs_path = os.path.join(self.cpp_source_dir, normalized_rel_path)
                if not os.path.exists(abs_path):
                    logger.warning(f"[Dependency] File not found: {abs_path}")
                    continue

                try:
                    # Extract interface (returns None on clang error, "" if parse ok but no elements)
                    interface_content = self._extract_cpp_interface(abs_path)

                    if interface_content is not None: # Check for None (parse error)
                        if not interface_content: # Handle empty interface string
                             logger.debug(f"Extracted empty interface for {normalized_rel_path}. Skipping addition.")
                             processed_paths.add(normalized_rel_path) # Mark as processed
                             continue

                        content_tokens = count_tokens(interface_content)
                        buffer = 50 # Small buffer per item

                        if check_token_limit(current_tokens + content_tokens + buffer, max_tokens):
                            content_map[normalized_rel_path] = interface_content # Add the extracted interface
                            current_tokens += content_tokens
                            processed_paths.add(normalized_rel_path)
                            logger.debug(f"Added INTERFACE content from {normalized_rel_path} ({content_tokens} tokens). Total: {current_tokens}")
                        else:
                            logger.warning(f"Skipping INTERFACE content from {normalized_rel_path} due to token limit ({current_tokens}+{content_tokens}+{buffer} > {max_tokens}).")
                            break # Stop adding dependency files
                    else:
                         # Extraction failed (logged in _extract_cpp_interface)
                         logger.warning(f"Failed to extract interface for dependency {normalized_rel_path}. Skipping.")
                         processed_paths.add(normalized_rel_path) # Mark as processed even on failure

                except Exception as e:
                    logger.error(f"Error processing dependency file {abs_path}: {e}", exc_info=True)

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

        # 1. Add 'other_context' items first (e.g., task description, JSON data)
        for key, content in other_context.items():
            if not content:
                logger.debug(f"Skipping empty other context item: {key}")
                continue

            # Format based on content type (simple text vs JSON)
            if isinstance(content, (dict, list)):
                 formatted_content = f"**{key.replace('_', ' ').title()} (JSON):**\n```json\n{json.dumps(content, indent=2)}\n```"
            else:
                 formatted_content = f"**{key.replace('_', ' ').title()}:**\n{str(content)}"

            component_tokens = count_tokens(formatted_content)
            separator_tokens = count_tokens("\n\n") # Separator between other context items

            if check_token_limit(current_total_tokens + component_tokens + separator_tokens, max_total_tokens):
                final_context_parts.append(formatted_content)
                current_total_tokens += component_tokens + separator_tokens
                logger.debug(f"Added other context '{key}' ({component_tokens} tokens). Total tokens: {current_total_tokens}")
            else:
                logger.warning(f"Skipping other context '{key}' due to token limit.")
                # Don't break here, try adding files still

        # 2. Add file contents using Markdown code blocks
        for file_path, content in file_content_map.items():
            if not content:
                logger.debug(f"Skipping empty file content: {file_path}")
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
                direct_includes = self.include_graph[normalized_pkg_file]
                if isinstance(direct_includes, list):
                    for included_file_rel in direct_includes:
                        normalized_include = included_file_rel.replace('\\', '/')
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
        Assembles context for a given workflow step based on provided arguments.
        Assembles context for a given workflow step, distinguishing between primary files
        (requiring full content) and dependency files (requiring extracted interfaces).

        Args:
            step_name (str): Identifier for the workflow step (e.g., "STRUCTURE_DEFINITION").
            primary_relative_paths (list[str]): List of file paths needing full content.
            dependency_relative_paths (list[str]): List of file paths needing interface extraction.
            **kwargs: Additional context items (e.g., 'task_description', 'work_package_info').

        Returns:
            str: The assembled context string formatted with Markdown, or empty string on error.
        """
        # Extract file path lists from kwargs, providing empty lists as defaults
        primary_relative_paths = kwargs.get('primary_relative_paths', [])
        dependency_relative_paths = kwargs.get('dependency_relative_paths', [])

        logger.info(f"Assembling context for STEP '{step_name}' ({len(primary_relative_paths)} primary, {len(dependency_relative_paths)} dependencies)")

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
        # Allocate budget for file content (full + interfaces)
        # Use CONTEXT_FILE_BUDGET_RATIO from config
        file_content_budget = int(max_context_assembly_tokens * config.CONTEXT_FILE_BUDGET_RATIO)
        file_content_map = self._get_contextual_content(
            primary_relative_paths=primary_relative_paths,
            dependency_relative_paths=dependency_relative_paths,
            max_tokens=file_content_budget
        )

        if not file_content_map and (primary_relative_paths or dependency_relative_paths):
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
