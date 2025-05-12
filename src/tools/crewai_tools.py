# src/tools/crewai_tools.py
from crewai.tools import BaseTool
from typing import Type, Any, List, Dict
from pydantic import BaseModel, Field
import src.config as config
import os

from src.tools.godot_validator_tool import validate_godot_project
from src.tools.framework_tools_wrapper import CrewAIFileWriter, CrewAIFileReader, CustomFileReplacer
from src.tools.remapping_logic import RemappingLogic
from src.core.tool_interfaces import IFileWriter, IFileReader, IFileReplacer
from src.logger_setup import get_logger
from src.tasks.step5.process_code import RemappingAdvice

logger = get_logger(__name__)

# --- Tool Input Schemas (using Pydantic) ---

class WriteFileInput(BaseModel):
    """Input schema for FileWriterTool."""
    file_path: str = Field(..., description="The absolute path of the file to write to.")
    content: str = Field(..., description="The content to write to the file.")

class ReplaceFileInput(BaseModel):
    """Input schema for FileReplacerTool."""
    file_path: str = Field(..., description="The absolute path of the file to modify.")
    diff: str = Field(..., description="The diff string in SEARCH/REPLACE format.")
    # Example diff format:
    # <<<<<<< SEARCH
    # [exact content to find]
    # =======
    # [new content to replace with]
    # >>>>>>> REPLACE

class ValidateSyntaxInput(BaseModel):
    """Input schema for GodotSyntaxValidatorTool."""
    script_content: str = Field(..., description="The GDScript content to validate.") # This will be removed

class ProjectValidationInput(BaseModel):
    """Input schema for GodotProjectValidatorTool."""
    godot_project_path: str = Field(..., description="The absolute path to the Godot project directory (containing project.godot).")
    target_file_path: str = Field(..., description="The res:// path of the file that was modified and should be checked for related errors.")

class RemappingCheckInput(BaseModel):
    """Input schema for RemappingLogicTool."""
    # Expecting a list of dictionaries, but Pydantic needs a more concrete type for validation.
    # Using List[Dict[str, Any]] is flexible but less strict.
    # If the structure of failed_tasks is stable, define a FailedTask Pydantic model here.
    failed_tasks: List[Dict[str, Any]] = Field(..., description="A list of dictionaries, where each dictionary represents a failed task report from Step 5 processing.")

class ReadFileInput(BaseModel):
    """Input schema for FileReaderTool."""
    file_path: str = Field(..., description="The absolute path of the file to read.")


# --- CrewAI Tool Definitions ---

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = ("Writes the given content to the specified file path. "
                        "Overwrites the file if it exists, creates it if it doesn't. "
                        "Requires 'file_path' and 'content'.")
    args_schema: Type[BaseModel] = WriteFileInput
    writer: IFileWriter = CrewAIFileWriter() # Use the existing wrapper

    def _run(self, file_path: str, content: str) -> str:
        logger.debug(f"FileWriterTool executing: received path='{file_path}'")
        absolute_path = file_path # Assume it might be absolute already

        # Check for res:// path and convert if necessary
        if file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if godot_project_dir:
                relative_path = file_path[len("res://"):]
                absolute_path = os.path.abspath(os.path.join(godot_project_dir, relative_path))
                logger.debug(f"FileWriterTool: Converted 'res://' path to absolute path: '{absolute_path}'")
            else:
                logger.error("FileWriterTool: Cannot convert 'res://' path because GODOT_PROJECT_DIR is not configured.")
                return "Error: GODOT_PROJECT_DIR not configured, cannot resolve res:// path."

        # Use the potentially converted absolute path
        result = self.writer.write(path=absolute_path, content=content)
        logger.info(f"FileWriterTool wrote '{absolute_path}' with content length {len(content)}.")
        # Return a descriptive string based on the result dict
        status = result.get('status', 'failure')
        message = result.get('message', 'No message provided.')
        return f"File Write Status: {status}. Message: {message}"

class FileReplacerTool(BaseTool):
    name: str = "File Content Replacer"
    description: str = ("Replaces a specific block of text within an existing file using a SEARCH/REPLACE diff format. "
                        "Requires 'file_path' and 'diff'.")
    args_schema: Type[BaseModel] = ReplaceFileInput
    replacer: IFileReplacer = CustomFileReplacer() # Use the existing wrapper

    def _run(self, file_path: str, diff: str) -> str:
        logger.debug(f"FileReplacerTool executing: received path='{file_path}'")
        absolute_path = file_path # Assume it might be absolute already

        # Check for res:// path and convert if necessary
        if file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if godot_project_dir:
                relative_path = file_path[len("res://"):]
                absolute_path = os.path.abspath(os.path.join(godot_project_dir, relative_path))
                logger.debug(f"FileReplacerTool: Converted 'res://' path to absolute path: '{absolute_path}'")
            else:
                logger.error("FileReplacerTool: Cannot convert 'res://' path because GODOT_PROJECT_DIR is not configured.")
                return "Error: GODOT_PROJECT_DIR not configured, cannot resolve res:// path."

        # Use the potentially converted absolute path
        result = self.replacer.replace(path=absolute_path, diff=diff)
        logger.info(f"FileReplacerTool wrote '{absolute_path}'.")
        status = result.get('status', 'failure')
        message = result.get('message', 'No message provided.')
        return f"File Replace Status: {status}. Message: {message}"

class GodotProjectValidatorTool(BaseTool):
    name: str = "Godot Project Validator"
    description: str = ("Validates the entire Godot project for script parsing errors by running the editor headlessly. "
                        "Filters the output to show only errors related to the specified 'target_file_path'. "
                        "Requires 'godot_project_path' and 'target_file_path' (e.g., 'res://path/to/file.gd'). "
                        "Primarily checks script integrity; scene/resource load errors might also be caught.")
    args_schema: Type[BaseModel] = ProjectValidationInput

    def _run(self, godot_project_path: str, target_file_path: str) -> str:
        logger.debug(f"GodotProjectValidatorTool executing for project: {godot_project_path}, target_file: {target_file_path}")
        try:
            # Call as a static/module function, not a method of self
            result = validate_godot_project(godot_project_path=godot_project_path)
            status = result.get('status', 'failure')
            errors = result.get('errors') # This is the full stderr output on failure

            if status == 'success':
                return "Project validation successful (Exit Code 0)."
            else:
                if not errors:
                    return "Project validation failed (Non-zero exit code), but no specific errors found on stderr."

                # Filter errors related to the target file
                relevant_errors = []
                # Normalize target path for comparison (e.g., 'res://scripts/player.gd' -> 'scripts/player.gd')
                normalized_target_path = target_file_path.replace("res://", "").replace("\\", "/")
                try:
                    for line in errors.splitlines():
                        # Check if the line contains the normalized target path or the original res:// path
                        if normalized_target_path in line.replace("\\", "/") or target_file_path in line:
                            relevant_errors.append(line)
                except Exception as filter_err:
                     logger.error(f"Error filtering validation output: {filter_err}", exc_info=True)
                     # Return unfiltered errors if filtering fails
                     return f"Project validation failed. Could not filter errors. Full Errors:\n{errors}"


                if relevant_errors:
                    filtered_error_string = "\n".join(relevant_errors)
                    logger.warning(f"Project validation failed with errors relevant to {target_file_path}:\n{filtered_error_string}")
                    return f"Project validation failed. Errors related to {target_file_path}:\n{filtered_error_string}"
                else:
                    logger.info(f"Project validation failed (Exit Code {result.get('returncode', 'N/A')}), but no errors found related to {target_file_path}. Treating as success for this file.")
                    # Optionally log the full errors for context
                    logger.debug(f"Full validation errors (unrelated):\n{errors}")
                    return f"Project validation successful (Exit Code 0 for {target_file_path}, though other project errors exist)."

        except Exception as e:
            logger.error(f"Error during GodotProjectValidatorTool execution: {e}", exc_info=True)
            return f"Project validation failed: Unexpected tool error - {e}"

class RemappingLogicTool(BaseTool):
    name: str = "Remapping Logic Analyzer"
    description: str = ("Analyzes a list of failed task reports for a work package to determine if remapping (re-running Step 4) is advisable. "
                        "Requires 'failed_tasks' (a list of dictionaries).")
    args_schema: Type[BaseModel] = RemappingCheckInput
    # remapping_logic class attribute will be used if we need to instantiate RemappingLogic
    # For now, we are calling static methods, so it's not strictly used but good for consistency.
    remapping_logic_class: Type[RemappingLogic] = RemappingLogic 

    def _run(self, failed_tasks: List[Dict[str, Any]]) -> str:
        logger.debug(f"RemappingLogicTool executing with {len(failed_tasks)} failed tasks.")
        try:
            should_remap_bool = self.remapping_logic_class.should_remap_package(failed_tasks=failed_tasks)
            
            reason_str = ""
            feedback_str = ""

            if should_remap_bool:
                feedback_str = self.remapping_logic_class.generate_mapping_feedback(failed_tasks)
                # Determine a more specific reason based on feedback content if possible
                if "High search block failure rate" in feedback_str:
                    reason_str = "High rate of search block related failures."
                elif "High validation failure rate" in feedback_str:
                    reason_str = "High rate of code validation failures."
                elif "CodeGen returned empty or non-string result" in feedback_str or "S1:" in feedback_str: # Check for S1 type errors
                    reason_str = "Systemic code generation failures detected."
                else:
                    reason_str = "Analysis of failure patterns suggests mapping issues."
                logger.info(f"RemappingLogicTool determined remapping IS recommended. Reason: {reason_str}")
            else:
                reason_str = "Failure patterns do not strongly indicate a systemic mapping issue requiring remapping."
                feedback_str = "No specific feedback for remapping; individual errors should be addressed if possible, or the issues are not related to mapping."
                logger.info("RemappingLogicTool determined remapping is NOT recommended.")

            # Import RemappingAdvice here or ensure it's available in the scope
            # For simplicity, assuming it's imported where this tool is defined (e.g., at the top of crewai_tools.py)
            # from src.tasks.step5.process_code import RemappingAdvice # Ensure this import exists at module level

            advice = RemappingAdvice(
                recommend_remapping=should_remap_bool,
                reason=reason_str,
                feedback=feedback_str
            )
            return advice.model_dump_json()
        except Exception as e:
            logger.error(f"Error during RemappingLogicTool execution: {e}", exc_info=True)
            # Return a JSON string indicating error, conforming to RemappingAdvice structure if possible
            error_advice = RemappingAdvice(
                recommend_remapping=False, # Default to false on error
                reason=f"Error during remapping analysis: {e}",
                feedback="An internal error occurred while trying to determine remapping advice."
            )
            return error_advice.model_dump_json()

# Example of how to instantiate (likely done in Step5Executor)
# file_writer_tool = FileWriterTool()
# file_replacer_tool = FileReplacerTool()
# syntax_validator_tool = GodotSyntaxValidatorTool()
class FileReaderTool(BaseTool):
    name: str = "File Reader"
    description: str = ("Reads the entire content of the specified file path. "
                        "Requires 'file_path'. Returns the file content or an error message if reading fails.")
    args_schema: Type[BaseModel] = ReadFileInput
    reader: IFileReader = CrewAIFileReader() # Use the existing wrapper

    def _run(self, file_path: str) -> str:
        logger.debug(f"FileReaderTool executing: received path='{file_path}'")
        absolute_path = file_path # Assume it might be absolute already

        # Check for res:// path and convert if necessary
        if file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if godot_project_dir:
                relative_path = file_path[len("res://"):]
                absolute_path = os.path.abspath(os.path.join(godot_project_dir, relative_path))
                logger.debug(f"Converted 'res://' path to absolute path: '{absolute_path}'")
            else:
                logger.error("FileReaderTool: Cannot convert 'res://' path because GODOT_PROJECT_DIR is not configured.")
                return "Error: GODOT_PROJECT_DIR not configured, cannot resolve res:// path."

        # Use the potentially converted absolute path
        result = self.reader.read(path=absolute_path)
        status = result.get('status', 'failure')
        if status == 'success':
            # Return the content directly on success
            return result.get('content', '')
        else:
            # Return a clear error message on failure
            message = result.get('message', 'Failed to read file.')
            logger.warning(f"FileReaderTool failed for path '{absolute_path}' (original: '{file_path}'): {message}")
            # Prepend "Error:" to make it clear to the agent that reading failed
            return f"Error: {message}"

# Example of how to instantiate (likely done in Step5Executor)
# file_writer_tool = FileWriterTool()
# file_replacer_tool = FileReplacerTool()
# syntax_validator_tool = GodotSyntaxValidatorTool()
# remapping_tool = RemappingLogicTool()
# file_reader_tool = FileReaderTool()
