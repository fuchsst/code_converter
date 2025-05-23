# tools/godot_validator_tool.py
import subprocess
import os
from src.logger_setup import get_logger
import src.config as config


logger = get_logger(__name__)

# Keywords to identify script errors in Godot's output
GODOT_ERROR_SIGNATURES = [
    "SCRIPT ERROR:",
    "ERROR: Failed to load script", # This is a more general error but often follows script errors
    "Parse Error:" # Often part of a SCRIPT ERROR line or other error messages
]

def validate_godot_project(
    godot_project_path: str,
    godot_exe_path: str = config.GODOT_EXECUTABLE_PATH,
    target_file_res_path: str = None
) -> dict:
    """
    Validates the Godot project, focusing on errors related to a specific target file if provided.

    Args:
        godot_project_path (str): Absolute path to the Godot project directory.
        godot_exe_path (str): Optional path to the Godot executable.
        target_file_res_path (str, optional): The 'res://' path of the modified file.
                                              If provided, validation focuses on errors in this file.

    Returns:
        dict: {'status': 'success'|'failure', 'errors': str|None}
              'errors' contains relevant error messages if status is 'failure'.
    """
    if target_file_res_path:
        logger.info(f"Validating Godot project at: {godot_project_path}, focusing on file: {target_file_res_path}")
    else:
        logger.info(f"Validating Godot project at: {godot_project_path} (no specific target file)")

    status = "failure" # Default to failure
    errors_output = "Unknown validation error." # Default error message

    if not os.path.isdir(godot_project_path):
        errors_output = f"Godot project path not found or is not a directory: {godot_project_path}"
        logger.error(errors_output)
        return {"status": status, "errors": errors_output}

    if not os.path.exists(os.path.join(godot_project_path, "project.godot")):
        errors_output = f"'project.godot' file not found in: {godot_project_path}"
        logger.error(errors_output)
        return {"status": status, "errors": errors_output}

    try:
        # Prepare the command for project validation
        command = [godot_exe_path, "--headless", "--path", godot_project_path, "-e", "--quit"]

        # Execute the command
        logger.debug(f"Executing project validation command: {' '.join(command)}")
        # Increased timeout might be needed for larger projects
        timeout_seconds = 60
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False, timeout=timeout_seconds)

        stderr_content = result.stderr or ""
        relevant_error_lines = []
        other_project_error_lines = [] # For errors not related to the target file but still are script errors

        if stderr_content:
            lines = stderr_content.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                is_script_error_line = any(sig in line for sig in GODOT_ERROR_SIGNATURES)

                if is_script_error_line:
                    # This line indicates a script error. Collect the block.
                    current_error_block = [line]
                    # Look ahead for subsequent lines that are part of this error (e.g., '   at: ...')
                    j = i + 1
                    # Continue if line starts with spaces (common for 'at:' lines) or doesn't start a new error
                    while j < len(lines) and \
                          (lines[j].startswith("   ") or not any(sig in lines[j] for sig in GODOT_ERROR_SIGNATURES)):
                        current_error_block.append(lines[j])
                        j += 1
                    
                    # Check if this error block is relevant to the target_file_res_path
                    block_is_relevant = False
                    if target_file_res_path:
                        for err_line in current_error_block:
                            if target_file_res_path in err_line:
                                block_is_relevant = True
                                break
                    
                    if block_is_relevant:
                        relevant_error_lines.extend(current_error_block)
                    else:
                        other_project_error_lines.extend(current_error_block)
                    i = j # Move past the processed block
                else:
                    # If not a script error line, it might be other stderr noise.
                    # We are primarily interested in script errors for failing validation.
                    i += 1
        
        # Determine status based on return code and relevant errors
        if result.returncode == 0:
            if relevant_error_lines:
                status = "failure"
                errors_output = "\n".join(relevant_error_lines)
                logger.warning(
                    f"Godot project validation failed for '{target_file_res_path}' (Exit Code 0) "
                    f"due to relevant script errors:\n{errors_output}"
                )
                if other_project_error_lines:
                    logger.info("Other script errors found in project (not related to target file):\n" + "\n".join(other_project_error_lines))
            else:
                status = "success"
                errors_output = None
                log_message = f"Godot project validation successful (Exit Code 0)."
                if target_file_res_path:
                    log_message += f" No script errors found for '{target_file_res_path}'."
                
                if other_project_error_lines:
                    logger.info(f"{log_message} However, other script errors were found in the project:\n" + "\n".join(other_project_error_lines))
                elif stderr_content: # Log generic stderr if it exists and no specific errors were categorized
                     logger.info(f"{log_message} Full stderr (no relevant script errors detected for target file):\n{stderr_content}")
                else:
                     logger.info(f"{log_message} No significant stderr output.")
        else: # Non-zero exit code always means failure
            status = "failure"
            # Prioritize relevant errors if found
            if relevant_error_lines:
                errors_output = (
                    f"Godot project validation failed (Exit Code {result.returncode}) "
                    f"with script errors relevant to '{target_file_res_path}':\n" + 
                    "\n".join(relevant_error_lines)
                )
                if other_project_error_lines:
                     errors_output += "\n\nAdditional project script errors (unrelated to target file):\n" + "\n".join(other_project_error_lines)
            elif other_project_error_lines: # Non-zero exit, but errors are not for target file
                 errors_output = (
                    f"Godot project validation failed (Exit Code {result.returncode}) "
                    f"due to script errors in other files:\n" +
                    "\n".join(other_project_error_lines)
                 )
            elif stderr_content: # Non-zero exit code, but no specific script errors parsed (e.g. engine crash)
                 errors_output = f"Godot project validation failed (Exit Code {result.returncode}). Full stderr:\n{stderr_content}"
            else: # Non-zero exit code, no stderr
                 errors_output = f"Godot project validation failed (Exit Code {result.returncode}) with no specific error message on stderr."
            logger.warning(errors_output)

    except FileNotFoundError:
        errors_output = f"Godot executable not found at '{godot_exe_path}'. Please ensure it's in PATH or configured correctly."
        logger.error(errors_output)
    except subprocess.TimeoutExpired:
        errors_output = f"Godot project validation timed out after {timeout_seconds} seconds."
        logger.error(errors_output)
    except Exception as e:
        errors_output = f"An unexpected error occurred during project validation: {e}"
        logger.error(errors_output, exc_info=True)

    return {"status": status, "errors": errors_output}
