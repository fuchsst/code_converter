# src/utils/dependency_analyzer_regex.py
import os
import re
import json
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

def generate_include_graph_regex(source_dir, output_path):
    """
    Generates an include graph for C++ files using regex to find project includes.

    Args:
        source_dir (str): The absolute path to the root directory of the C++ source code.
        output_path (str): The absolute path where the output JSON graph should be saved.

    Returns:
        bool: True if the graph generation and saving were successful, False otherwise.
    """
    logger.info(f"Starting Regex include graph generation for directory: {source_dir}")
    abs_source_dir = os.path.abspath(source_dir)
    include_graph = {}
    processed_files = 0
    skipped_files = 0

    if not os.path.isdir(abs_source_dir):
        logger.error(f"Source directory not found: {abs_source_dir}")
        return False

    for root, _, files in os.walk(abs_source_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in CPP_EXTENSIONS:
                file_path_abs = os.path.join(root, filename)
                relative_file_path = os.path.relpath(file_path_abs, abs_source_dir).replace('\\', '/')
                logger.debug(f"Processing file: {relative_file_path}")
                processed_files += 1
                includes = set()
                try:
                    with open(file_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        matches = INCLUDE_REGEX.findall(content)
                        for included_file in matches:
                            # Normalize the included path relative to the *current file's* directory first
                            current_dir = os.path.dirname(file_path_abs)
                            abs_included_path = os.path.abspath(os.path.join(current_dir, included_file))

                            # Check if the resolved absolute path is within the source directory
                            if os.path.commonpath([abs_included_path, abs_source_dir]) == abs_source_dir:
                                # Convert to relative path based on the main source_dir for the graph key
                                rel_path_from_source = os.path.relpath(abs_included_path, abs_source_dir).replace('\\', '/')
                                includes.add(rel_path_from_source)
                                logger.debug(f"  Found project include: {rel_path_from_source}")
                            else:
                                logger.debug(f"  Skipping non-project or external include: {included_file} -> {abs_included_path}")

                except FileNotFoundError:
                    logger.warning(f"File not found during processing (should not happen with os.walk): {file_path_abs}")
                    skipped_files += 1
                    continue
                except Exception as e:
                    logger.error(f"Error reading or processing file {relative_file_path}: {e}", exc_info=True)
                    skipped_files += 1
                    continue # Skip this file on error

                # Add even if includes is empty, to represent files with no project includes
                include_graph[relative_file_path] = sorted(list(includes))

    logger.info(f"Generated include graph with {len(include_graph)} entries using Regex ({processed_files} processed, {skipped_files} skipped/failed).")

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
    try:
        import src.config as config
        # Default values if not in config
        source_directory = getattr(config, 'CPP_SOURCE_DIR', './input_code') # Example default
        output_file = getattr(config, 'INCLUDE_GRAPH_PATH', './analysis_results/includes_regex.json') # Example default
    except ImportError:
        logger.warning("config.py not found, using hardcoded defaults for standalone run.")
        source_directory = './input_code' # Example default
        output_file = './analysis_results/includes_regex.json' # Example default
    except AttributeError as e:
         logger.error(f"Configuration error: Potentially missing variable in config.py ({e}). Using defaults.")
         source_directory = './input_code'
         output_file = './analysis_results/includes_regex.json'

    logger.info(f"Running standalone regex analysis. Source: '{source_directory}', Output: '{output_file}'")
    if not os.path.isdir(source_directory):
         logger.error(f"Source directory not found: {source_directory}")
    else:
        generate_include_graph_regex(source_directory, output_file)
