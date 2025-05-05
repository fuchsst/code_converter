# tools/godot_validator_tool.py
import subprocess
import os
import tempfile
from src.logger_setup import get_logger
import src.config as config


logger = get_logger(__name__)


def validate_godot_project(godot_project_path: str, godot_exe_path: str = config.GODOT_EXECUTABLE_PATH) -> dict:
    """
    Validates the entire Godot project by running the editor headlessly.
    Checks for script parsing errors across the project.

    Args:
        godot_project_path (str): The absolute path to the Godot project directory (containing project.godot).
        godot_exe_path (str): Optional path to the Godot executable. Defaults to path in config.

    Returns:
        dict: A dictionary with keys 'status' ('success' or 'failure') and 'errors'
              (string containing error messages from Godot stderr if validation fails,
              or None if successful).
    """
    logger.info(f"Validating Godot project at: {godot_project_path}")
    status = "failure"
    errors = "Unknown validation error."

    if not os.path.isdir(godot_project_path):
        errors = f"Godot project path not found or is not a directory: {godot_project_path}"
        logger.error(errors)
        return {"status": status, "errors": errors}

    if not os.path.exists(os.path.join(godot_project_path, "project.godot")):
        errors = f"'project.godot' file not found in: {godot_project_path}"
        logger.error(errors)
        return {"status": status, "errors": errors}

    try:
        # Prepare the command for project validation
        command = [godot_exe_path, "--headless", "--path", godot_project_path]

        # Execute the command
        logger.debug(f"Executing project validation command: {' '.join(command)}")
        # Increased timeout might be needed for larger projects
        timeout_seconds = 60
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False, timeout=timeout_seconds)

        if result.returncode == 0:
            # Even with exit code 0, stderr might contain warnings or non-fatal errors.
            # We consider exit code 0 as success for basic syntax/parsing.
            status = "success"
            errors = None # No fatal errors reported via exit code
            logger.info(f"Godot project validation successful (Exit Code 0). Stderr (if any):\n{result.stderr or 'None'}")
        else:
            status = "failure"
            # Errors are typically printed to stderr when exit code is non-zero
            errors = f"Godot project validation failed (Exit Code {result.returncode}):\n{result.stderr or 'No specific error message on stderr.'}"
            logger.warning(f"Godot project validation failed:\n{errors}")

    except FileNotFoundError:
        errors = f"Godot executable not found at '{godot_exe_path}'. Please ensure it's in PATH or configured correctly."
        logger.error(errors)
    except subprocess.TimeoutExpired:
        errors = f"Godot project validation timed out after {timeout_seconds} seconds."
        logger.error(errors)
    except Exception as e:
        errors = f"An unexpected error occurred during project validation: {e}"
        logger.error(errors, exc_info=True)
    # No temporary file cleanup needed anymore

    return {"status": status, "errors": errors}
