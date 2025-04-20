# src/utils/dependency_analyzer_regex.py
import os
import re
import json
from pathlib import Path # Use pathlib for easier path manipulation
from typing import List, Optional
from src.logger_setup import get_logger

logger = get_logger(__name__)

# Regex to find #include "..." directives
# - Matches '#include'
# - Followed by one or more whitespace characters (\s+)
# - Followed by a double quote (")
# - Captures one or more characters that are not a double quote ([^"]+)
# - Followed by a closing double quote (")
INCLUDE_REGEX = re.compile(r'#include\s+"([^"]+)"')

# Typical C++ file extensions to scan
CPP_EXTENSIONS = {'.cpp', '.h', '.hpp', '.c', '.cc', '.hh'}


def calculate_path_distance(from_path_rel: str, to_path_rel: str) -> int:
    """
    Calculates a simple directory traversal distance between two relative file paths.
    Distance = (levels up from 'from_path' dir to common ancestor) + (levels down from common ancestor to 'to_path' dir).
    Assumes paths are relative to the same root and use '/' separators.
    Returns 1 if files are in the same directory.
    """
    logger.debug(f"Calculating distance from '{from_path_rel}' to '{to_path_rel}'")
    # Use pathlib for robust path handling
    from_dir = Path(from_path_rel).parent
    to_dir = Path(to_path_rel).parent
    logger.debug(f"  From dir: {from_dir}, To dir: {to_dir}")

    # Handle edge case: if paths are identical (e.g., self-include, though unlikely)
    if from_path_rel == to_path_rel:
        logger.debug("  Paths are identical, distance is 0.")
        return 0

    # Handle case: files in the same directory
    if from_dir == to_dir:
        logger.debug("  Directories are the same, distance is 0.")
        return 0 # Changed from 1 to 0 for same directory

    # Find common ancestor using pathlib parts
    from_parts = from_dir.parts
    to_parts = to_dir.parts
    logger.debug(f"  From parts: {from_parts}, To parts: {to_parts}")

    common_len = 0
    for i in range(min(len(from_parts), len(to_parts))):
        if from_parts[i] == to_parts[i]:
            common_len += 1
        else:
            break
    logger.debug(f"  Common path length: {common_len}")

    levels_up = len(from_parts) - common_len
    levels_down = len(to_parts) - common_len
    logger.debug(f"  Levels up: {levels_up}, Levels down: {levels_down}")

    distance = levels_up + levels_down
    logger.debug(f"  Calculated distance: {distance}")
    return distance


def generate_include_graph_regex(source_dir: str,
                                 output_path: str,
                                 exclude_folders: Optional[List[str]] = None) -> bool:
    """
    Generates an include graph for C++ files using regex to find project includes,
    calculating path distance weights and supporting folder exclusion.

    Args:
        source_dir (str): The absolute path to the root directory of the C++ source code.
        output_path (str): The absolute path where the output JSON graph should be saved.
        exclude_folders (Optional[List[str]]): A list of folder paths relative to source_dir
                                                to exclude from analysis.

    Returns:
        bool: True if the graph generation and saving were successful, False otherwise.
    """
    logger.info(f"Starting Regex include graph generation for directory: {source_dir}")
    abs_source_dir = os.path.abspath(source_dir)
    include_graph = {}
    processed_files = 0
    skipped_files = 0
    skipped_dirs = 0 # Track skipped directories

    # Normalize exclude_folders relative to source_dir
    normalized_excludes = set()
    if exclude_folders:
        for folder in exclude_folders:
            # Make sure exclude paths are relative to source_dir and use forward slashes
            try:
                # Ensure the path exists relative to source_dir before getting absolute path
                abs_folder_path = os.path.abspath(os.path.join(source_dir, folder))
                if os.path.commonpath([abs_folder_path, abs_source_dir]) != abs_source_dir:
                     logger.warning(f"Exclude folder '{folder}' is outside source directory '{source_dir}'. Skipping.")
                     continue

                rel_exclude = os.path.relpath(abs_folder_path, abs_source_dir).replace('\\', '/')
                # Handle cases like '.' or '' which mean exclude the root itself (disallow this)
                if rel_exclude and rel_exclude != '.':
                    normalized_excludes.add(rel_exclude)
                elif rel_exclude == '.':
                    logger.warning("Excluding the root directory '.' is not allowed. Skipping this exclusion.")
            except ValueError as e:
                 logger.warning(f"Could not normalize exclude folder '{folder}': {e}. Skipping.")

        if normalized_excludes:
            logger.info(f"Excluding folders relative to source: {sorted(list(normalized_excludes))}")

    if not os.path.isdir(abs_source_dir):
         logger.error(f"Source directory not found: {abs_source_dir}")
         return False

    for root, dirs, files in os.walk(abs_source_dir, topdown=True): # Use topdown=True to modify dirs list
        # --- Exclusion Logic ---
        current_rel_dir = os.path.relpath(root, abs_source_dir).replace('\\', '/')
        if current_rel_dir == '.': # Root directory case
            current_rel_dir = ""

        # Check if the current directory itself or a parent is excluded
        is_excluded = False
        temp_dir = current_rel_dir
        while True:
            if temp_dir in normalized_excludes:
                is_excluded = True
                break
            parent = os.path.dirname(temp_dir)
            if parent == temp_dir: # Reached root or empty string
                break
            temp_dir = parent

        if is_excluded:
            logger.debug(f"Skipping excluded directory (or subdirectory of excluded): {current_rel_dir}")
            # Remove all subdirectories from the list so os.walk doesn't visit them
            skipped_dirs += len(dirs)
            dirs[:] = []
            continue # Skip processing files in this directory

        # Filter subdirectories to prevent walking into explicitly excluded ones later
        # (This might be slightly redundant with the check above but ensures exact matches are pruned)
        original_dirs_count = len(dirs)
        dirs[:] = [d for d in dirs if os.path.join(current_rel_dir, d).replace('\\', '/') not in normalized_excludes]
        skipped_dirs += original_dirs_count - len(dirs)
        # --- End Exclusion Logic ---

        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in CPP_EXTENSIONS:
                file_path_abs = os.path.join(root, filename)
                # Use forward slashes for consistency in keys and paths
                relative_file_path = os.path.relpath(file_path_abs, abs_source_dir).replace('\\', '/')
                logger.debug(f"Processing file: {relative_file_path}")
                processed_files += 1
                includes_with_weights = [] # Store dicts: {"path": str, "weight": int}
                try:
                    with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = INCLUDE_REGEX.findall(content)
                        for include_path_str in matches:
                            # Use pathlib for robust path handling
                            current_file_path_obj = Path(file_path_abs)
                            current_dir_obj = current_file_path_obj.parent
                            source_dir_obj = Path(abs_source_dir) # Ensure source_dir is Path object

                            # Attempt 1: Resolve relative to the current file's directory
                            resolved_path_rel_current = (current_dir_obj / include_path_str).resolve()

                            # Attempt 2: Resolve relative to the source directory root
                            resolved_path_rel_source = (source_dir_obj / include_path_str).resolve()

                            final_abs_path = None
                            # Check if path relative to current file exists and is within source_dir
                            if resolved_path_rel_current.exists() and resolved_path_rel_current.is_relative_to(source_dir_obj):
                                final_abs_path = resolved_path_rel_current
                                logger.debug(f"  Resolved '{include_path_str}' relative to current file: {final_abs_path}")
                            # Else, check if path relative to source root exists and is within source_dir
                            elif resolved_path_rel_source.exists() and resolved_path_rel_source.is_relative_to(source_dir_obj):
                                final_abs_path = resolved_path_rel_source
                                logger.debug(f"  Resolved '{include_path_str}' relative to source root: {final_abs_path}")
                            # else:
                                # logger.debug(f"  Could not resolve '{include_path_str}' within project from {relative_file_path}")


                            # If we found a valid path within the project source directory
                            if final_abs_path:
                                # Convert to relative path based on the main source_dir for the graph key
                                rel_path_from_source = final_abs_path.relative_to(source_dir_obj).as_posix() # Use as_posix() for '/' separators

                                # --- Check if the *included file* is in an excluded directory ---
                                included_dir_rel = os.path.dirname(rel_path_from_source).replace('\\', '/')
                                if included_dir_rel == '.': included_dir_rel = "" # Handle root includes

                                is_include_excluded = False
                                temp_dir = included_dir_rel
                                while True:
                                    if temp_dir in normalized_excludes:
                                        is_include_excluded = True
                                        break
                                    parent = os.path.dirname(temp_dir)
                                    if parent == temp_dir: # Reached root or empty string
                                        break
                                    temp_dir = parent

                                if is_include_excluded:
                                    logger.debug(f"  Skipping include pointing to excluded directory: '{include_path_str}' -> {rel_path_from_source}")
                                    continue
                                # --- End include exclusion check ---

                                # Calculate distance/weight
                                distance = calculate_path_distance(relative_file_path, rel_path_from_source)
                                includes_with_weights.append({"path": rel_path_from_source, "weight": distance})
                                logger.debug(f"  Found project include: {rel_path_from_source} (Weight: {distance})")
                            else:
                                logger.debug(f"  Skipping non-project or external include: '{include_path_str}' (Tried: {resolved_path_rel_current}, {resolved_path_rel_source})")

                except FileNotFoundError:
                    logger.warning(f"File not found during processing (should not happen with os.walk): {file_path_abs}")
                    skipped_files += 1
                    continue
                except Exception as e:
                    logger.error(f"Error reading or processing file {relative_file_path}: {e}", exc_info=True)
                    skipped_files += 1
                    continue # Skip this file on error

                # Sort includes by path for consistent output
                includes_with_weights.sort(key=lambda x: x['path'])
                # Add even if includes is empty, to represent files with no project includes
                include_graph[relative_file_path] = includes_with_weights

    logger.info(f"Generated include graph with {len(include_graph)} entries using Regex ({processed_files} processed, {skipped_files} skipped/failed, {skipped_dirs} dirs skipped).")

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir: # Only create if path includes a directory
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(include_graph, f, indent=4)
        logger.info(f"Include graph saved successfully to: {output_path}")
        return True
    except IOError as e:
        logger.error(f"Failed to save include graph to {output_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the graph: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Example usage: Run this script directly
    # Assumes config.py might exist in parent dirs or PYTHONPATH is set
    # Also add exclude_folders example
    try:
        import src.config as config
        # Default values if not in config
        source_directory = getattr(config, 'CPP_SOURCE_DIR', './input_code') # Example default
        output_file = getattr(config, 'INCLUDE_GRAPH_PATH', './analysis_results/includes_regex.json') # Example default
        exclude_folders_example = getattr(config, 'EXCLUDE_FOLDERS', ['build', 'tests']) # Example default
    except ImportError:
        logger.warning("config.py not found, using hardcoded defaults for standalone run.")
        source_directory = './input_code' # Example default
        output_file = './analysis_results/includes_regex.json' # Example default
        exclude_folders_example = ['build', 'tests'] # Example default
    except AttributeError as e:
         logger.error(f"Configuration error: Potentially missing variable in config.py ({e}). Using defaults.")
         source_directory = './input_code'
         output_file = './analysis_results/includes_regex.json'
         exclude_folders_example = ['build', 'tests']

    logger.info(f"Running standalone regex analysis. Source: '{source_directory}', Output: '{output_file}', Exclude: {exclude_folders_example}")
    if not os.path.isdir(source_directory):
         logger.error(f"Source directory not found: {source_directory}")
    else:
        generate_include_graph_regex(source_directory, output_file, exclude_folders=exclude_folders_example)
