# tools/godot_validator_tool.py
import subprocess
import os
import tempfile
from logger_setup import get_logger
import config
from crewai_tools import tool # Import the tool decorator

logger = get_logger(__name__)

@tool("GDScript Syntax Validator")
def validate_gdscript_syntax(script_content: str, godot_exe_path: str = config.GODOT_EXECUTABLE_PATH) -> dict:
    """
    Validates the syntax of a given GDScript code string using the Godot executable's
    `--check-only` command-line option. This tool is useful for quickly checking if
    generated or modified GDScript code is syntactically correct before saving or
    further processing.

    Args:
        script_content (str): The string containing the GDScript code to validate.
        godot_exe_path (str): Optional path to the Godot executable. Defaults to path in config.

    Returns:
        dict: A dictionary with keys 'status' ('success' or 'failure') and 'errors'
              (string containing error messages from Godot stderr/stdout if validation fails,
              or None if successful).
    """
    logger.info("Validating GDScript syntax...")
    status = "failure"
    errors = "Unknown validation error."
    temp_script_path = None # Initialize path variable

    try:
        # Create a temporary file to hold the script content
        with tempfile.NamedTemporaryFile(mode='w', suffix=".gd", delete=False, encoding='utf-8') as temp_script:
            temp_script_path = temp_script.name
            temp_script.write(script_content)
            temp_script.flush() # Ensure content is written to disk
            logger.debug(f"Created temporary GDScript file: {temp_script_path}")

        # Prepare the command for GDScript validation
        command = [godot_exe_path, "--headless", "--check-only", "--script", temp_script_path]
        # Note: Running from *outside* a project context might limit checks (e.g., class_name usage).
        # Added --headless as it's often needed for CLI operations without opening the editor.

        # Execute the command
        logger.debug(f"Executing validation command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False) # Use check=False to handle non-zero exit codes manually

        if result.returncode == 0:
            # Check stderr for potential warnings even on success code 0
            if result.stderr:
                 logger.warning(f"GDScript validation successful (code 0), but stderr contained warnings:\n{result.stderr}")
                 # Decide if warnings should constitute failure or just be logged
                 # For now, treat code 0 as success regardless of stderr warnings
            status = "success"
            errors = None
            logger.info("GDScript syntax validation successful.")
        else:
            status = "failure"
            # Combine stdout and stderr for more complete error context
            errors = f"Godot validation failed (Exit Code {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            logger.warning(f"GDScript syntax validation failed:\n{errors}")

    except FileNotFoundError:
        errors = f"Godot executable not found at '{godot_exe_path}'. Please ensure it's in PATH or configured correctly."
        logger.error(errors)
    except Exception as e:
        errors = f"An unexpected error occurred during validation: {e}"
        logger.error(errors, exc_info=True)
    finally:
        # Clean up the temporary file
        if 'temp_script_path' in locals() and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
                logger.debug(f"Removed temporary script file: {temp_script_path}")
            except OSError as e:
                logger.error(f"Failed to remove temporary file {temp_script_path}: {e}")

    return {"status": status, "errors": errors}
