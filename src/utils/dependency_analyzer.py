# src/utils/dependency_analyzer.py
import os
import json
import sys
from logger_setup import get_logger
import config # Assuming config.py defines CPP_SOURCE_DIR and INCLUDE_GRAPH_PATH

# --- Clang Imports ---
try:
    import clang.cindex
    # Optional: Set path to libclang if not found automatically
    # clang.cindex.Config.set_library_file('/path/to/libclang.so') # Example for Linux
    # clang.cindex.Config.set_library_path('/path/to/llvm/bin') # Example for Windows
except ImportError:
    print("Clang Python bindings not found. Please install 'clang' (e.g., pip install clang)")
    sys.exit(1)
except Exception as e:
     print(f"Error initializing libclang: {e}. Ensure libclang is installed and accessible.")
     print("You might need to set the library path using clang.cindex.Config.set_library_path()")
     sys.exit(1)

logger = get_logger(__name__)

# --- Helper Functions ---

def find_compile_commands(project_dir):
    """Finds the compile_commands.json file."""
    cc_path = os.path.join(project_dir, 'compile_commands.json')
    if os.path.exists(cc_path):
        logger.info(f"Found compile_commands.json at: {cc_path}")
        return cc_path
    else:
        logger.warning(f"compile_commands.json not found in {project_dir}. Clang analysis might be inaccurate.")
        return None

# --- Clang-based Include Extraction ---

def extract_includes_with_clang(file_path, compile_db, index, source_dir_abs):
    """
    Parses a C++ file using libclang to extract direct #include directives.
    Requires compile_commands.json for accurate parsing.
    """
    includes = set()
    try:
        # Get compile commands for the specific file
        commands = compile_db.getCompileCommands(file_path)
        if not commands:
            logger.debug(f"No compile commands found for {file_path} in compile_commands.json. Skipping clang parse.")
            # Optionally fall back to regex or skip? For now, skip.
            return None # Indicate parse failure due to missing commands

        # Use the first command (usually only one per file)
        # Extract arguments, removing the compiler itself and the output flag
        args = list(commands[0].arguments)[1:]
        try:
            args.remove('-o') # Remove output flag and its argument if present
            args.pop(args.index('-o') + 1)
        except ValueError:
            pass # -o not found

        # Parse the file
        # Consider adding options like clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        # or clang.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES for performance if needed.
        tu = index.parse(file_path, args=args)

        if not tu:
            logger.error(f"Clang failed to parse file: {file_path}")
            return None # Indicate parse failure

        # Check for parsing errors
        has_errors = False
        for diag in tu.diagnostics:
            if diag.severity >= clang.cindex.Diagnostic.Error:
                logger.warning(f"Clang parse error in {file_path}: {diag.spelling} at {diag.location}")
                has_errors = True
        # Decide if errors should prevent include extraction. For now, proceed cautiously.
        # if has_errors: return None

        # Traverse the AST for include directives
        for cursor in tu.cursor.walk_preorder():
            if cursor.kind == clang.cindex.CursorKind.INCLUDE_DIRECTIVE:
                included_file_node = cursor.get_included_file()
                if included_file_node:
                    included_file_path = included_file_node.name
                    # Normalize path
                    abs_included_path = os.path.abspath(included_file_path)

                    # Check if the included file is within the project source directory
                    if os.path.commonpath([abs_included_path, source_dir_abs]) == source_dir_abs:
                        # Convert to relative path for the graph
                        rel_path = os.path.relpath(abs_included_path, source_dir_abs).replace('\\', '/')
                        includes.add(rel_path)
                        # logger.debug(f"Found project include: {rel_path} in {os.path.relpath(file_path, source_dir_abs)}")
                    # else:
                        # logger.debug(f"Ignoring non-project include: {abs_included_path} in {file_path}")

    except clang.cindex.LibclangError as e:
        logger.error(f"Libclang error processing file {file_path}: {e}", exc_info=True)
        return None # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error processing file {file_path} with clang: {e}", exc_info=True)
        return None # Indicate failure

    return sorted(list(includes))


def generate_include_graph(source_dir, output_path, compile_commands_path=None):
    """
    Generates an include graph for C++ files using libclang and compile_commands.json.
    """
    logger.info(f"Starting Clang include graph generation for directory: {source_dir}")
    abs_source_dir = os.path.abspath(source_dir)

    if compile_commands_path is None:
        compile_commands_path = find_compile_commands(abs_source_dir)

    if not compile_commands_path or not os.path.exists(compile_commands_path):
        logger.error(f"compile_commands.json not found or specified. Cannot perform accurate Clang analysis.")
        return False

    try:
        compile_db = clang.cindex.CompilationDatabase.fromDirectory(os.path.dirname(compile_commands_path))
        index = clang.cindex.Index.create()
    except clang.cindex.LibclangError as e:
         logger.error(f"Failed to load compilation database or create Clang index: {e}")
         return False
    except Exception as e:
         logger.error(f"Unexpected error initializing Clang: {e}", exc_info=True)
         return False


    # Get list of files from compile_commands.json AND potentially scan directory
    # Files in compile_commands are usually the primary compilation units (.cpp)
    files_in_db = {os.path.abspath(cmd.filename) for cmd in compile_db.getAllCompileCommands()}
    logger.info(f"Found {len(files_in_db)} files listed in {compile_commands_path}")

    files_to_process = files_in_db
    logger.warning("Processing only files listed in compile_commands.json. Header files not included by these units might be missed in the dependency graph.")

    if not files_to_process:
        logger.warning(f"No files found in compile_commands.json within {source_dir}. Cannot generate graph.")
        return False

    include_graph = {}
    processed_count = 0
    skipped_count = 0

    for file_path_abs in files_to_process:
         # Ensure file is within the target source directory
         if os.path.commonpath([file_path_abs, abs_source_dir]) != abs_source_dir:
             logger.debug(f"Skipping file outside source directory: {file_path_abs}")
             continue

         # Make the key relative to the source_dir for consistency
         relative_file_path = os.path.relpath(file_path_abs, abs_source_dir).replace('\\', '/')
         logger.debug(f"Processing file: {relative_file_path}")

         includes = extract_includes_with_clang(file_path_abs, compile_db, index, abs_source_dir)

         if includes is not None:
             # Only add entry if includes were successfully processed (even if empty)
             include_graph[relative_file_path] = includes
             processed_count += 1
         else:
             logger.warning(f"Skipping entry for {relative_file_path} due to processing error or missing compile commands.")
             skipped_count += 1

    logger.info(f"Generated include graph with {len(include_graph)} entries using Clang ({processed_count} processed, {skipped_count} skipped/failed).")

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
    # Make sure config.py has CPP_SOURCE_DIR and INCLUDE_GRAPH_PATH defined
    try:
        # Default values if not in config
        source_directory = getattr(config, 'CPP_SOURCE_DIR', './input_code') # Example default
        output_file = getattr(config, 'INCLUDE_GRAPH_PATH', './analysis_results/includes.json') # Example default

        if not os.path.isdir(source_directory):
             logger.error(f"Source directory not found: {source_directory}")
        else:
            logger.info(f"Running standalone analysis. Source: '{source_directory}', Output: '{output_file}'")
            generate_include_graph(source_directory, output_file)

    except AttributeError as e:
         logger.error(f"Configuration error: Potentially missing variable in config.py ({e}). Using defaults if possible.")
         # Attempt to run with defaults if specific config vars are missing
         source_directory = './input_code'
         output_file = './analysis_results/includes.json'
         logger.info(f"Running standalone analysis with defaults. Source: '{source_directory}', Output: '{output_file}'")
         if not os.path.isdir(source_directory):
             logger.error(f"Default source directory not found: {source_directory}")
         else:
             generate_include_graph(source_directory, output_file)

    except Exception as e:
         logger.error(f"An error occurred during script execution: {e}", exc_info=True)
