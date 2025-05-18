# src/tools/crewai_tools.py
from crewai.tools import BaseTool
from typing import Type, Any, List, Dict
from pydantic import BaseModel, Field
import src.config as config
import os

from src.tools.godot_validator_tool import validate_godot_project
from src.tools.framework_tools_wrapper import CrewAIFileWriter, CustomFileReplacer # Removed CrewAIFileReader
from src.tools.remapping_logic import RemappingLogic
from src.core.tool_interfaces import IFileWriter, IFileReplacer # Removed IFileReader from here as it's used by the new tools
from src.logger_setup import get_logger
from src.tasks.step5.process_code import RemappingAdvice
# Import new reader tools and their schemas
from src.tools.cpp_file_reader_tool import CppFileReaderTool, CppReadFileInput
from src.tools.godot_file_reader_tool import GodotFileReaderTool, GodotReadFileInput


logger = get_logger(__name__)

# --- Tool Input Schemas (using Pydantic) ---

class WriteFileInput(BaseModel):
    """Input schema for FileWriterTool."""
    file_path: str = Field(..., description="The file path. Can be res:// in GODOT_PROJECT_DIR, or relative to CPP_PROJECT_DIR.")
    content: str = Field(..., description="The content to write to the file.")

class ReplaceFileInput(BaseModel):
    """Input schema for FileReplacerTool."""
    file_path: str = Field(..., description="The file path. Can be res:// in GODOT_PROJECT_DIR, or relative to CPP_PROJECT_DIR.")
    diff: str = Field(..., description="The diff string in SEARCH/REPLACE format.")


class ProjectValidationInput(BaseModel):
    """Input schema for GodotProjectValidatorTool."""
    godot_project_path: str = Field(..., description="The absolute path to the Godot project directory (containing project.godot).")
    target_file_path: str = Field(..., description="The res:// path of the file that was modified and should be checked for related errors.")

class RemappingCheckInput(BaseModel):
    """Input schema for RemappingLogicTool."""
    failed_tasks: List[Dict[str, Any]] = Field(..., description="A list of dictionaries, where each dictionary represents a failed task report from Step 5 processing.")


# --- CrewAI Tool Definitions ---

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = ("Writes the given content to the specified file path. "
                        "Interprets res:// paths relative to GODOT_PROJECT_DIR, other relative paths to CPP_PROJECT_DIR. "
                        "Overwrites the file if it exists, creates it if it doesn't. "
                        "Requires 'file_path' and 'content'.")
    args_schema: Type[BaseModel] = WriteFileInput
    writer: IFileWriter = CrewAIFileWriter()

    def _run(self, file_path: str, content: str) -> str:
        logger.debug(f"FileWriterTool executing: received path='{file_path}'")
        resolved_path_str: str
        
        if os.path.isabs(file_path):
            resolved_path_str = file_path
            logger.debug(f"FileWriterTool: Path '{file_path}' is absolute.")
        elif file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if godot_project_dir:
                relative_path = file_path[len("res://"):]
                resolved_path_str = os.path.abspath(os.path.join(godot_project_dir, relative_path))
                logger.debug(f"FileWriterTool: Converted 'res://' path '{file_path}' to absolute path: '{resolved_path_str}'")
            else:
                logger.error("FileWriterTool: GODOT_PROJECT_DIR not configured, cannot resolve res:// path.")
                return "Error: GODOT_PROJECT_DIR not configured for res:// path."
        else: # Relative path, not starting with res://, assume relative to CPP_PROJECT_DIR
            cpp_project_dir = config.CPP_PROJECT_DIR
            if cpp_project_dir:
                resolved_path_str = os.path.abspath(os.path.join(cpp_project_dir, file_path))
                logger.debug(f"FileWriterTool: Resolved relative path '{file_path}' to absolute path: '{resolved_path_str}' using CPP_PROJECT_DIR.")
            else:
                logger.error("FileWriterTool: CPP_PROJECT_DIR not configured, cannot resolve relative path.")
                return "Error: CPP_PROJECT_DIR not configured for relative path."

        result = self.writer.write(path=resolved_path_str, content=content)
        logger.info(f"FileWriterTool wrote '{resolved_path_str}' with content length {len(content)}.")
        status = result.get('status', 'failure')
        message = result.get('message', 'No message provided.')
        return f"File Write Status: {status}. Message: {message}"

class FileReplacerTool(BaseTool):
    name: str = "File Content Replacer"
    description: str = ("Replaces a specific block of text within an existing file using a SEARCH/REPLACE diff format. "
                        "Interprets res:// paths relative to GODOT_PROJECT_DIR, other relative paths to CPP_PROJECT_DIR. "
                        "Requires 'file_path' and 'diff'.")
    args_schema: Type[BaseModel] = ReplaceFileInput
    replacer: IFileReplacer = CustomFileReplacer()

    def _run(self, file_path: str, diff: str) -> str:
        logger.debug(f"FileReplacerTool executing: received path='{file_path}'")
        resolved_path_str: str

        if os.path.isabs(file_path):
            resolved_path_str = file_path
            logger.debug(f"FileReplacerTool: Path '{file_path}' is absolute.")
        elif file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if godot_project_dir:
                relative_path = file_path[len("res://"):]
                resolved_path_str = os.path.abspath(os.path.join(godot_project_dir, relative_path))
                logger.debug(f"FileReplacerTool: Converted 'res://' path '{file_path}' to absolute path: '{resolved_path_str}'")
            else:
                logger.error("FileReplacerTool: GODOT_PROJECT_DIR not configured, cannot resolve res:// path.")
                return "Error: GODOT_PROJECT_DIR not configured for res:// path."
        else: # Relative path, not starting with res://, assume relative to CPP_PROJECT_DIR
            cpp_project_dir = config.CPP_PROJECT_DIR
            if cpp_project_dir:
                resolved_path_str = os.path.abspath(os.path.join(cpp_project_dir, file_path))
                logger.debug(f"FileReplacerTool: Resolved relative path '{file_path}' to absolute path: '{resolved_path_str}' using CPP_PROJECT_DIR.")
            else:
                logger.error("FileReplacerTool: CPP_PROJECT_DIR not configured, cannot resolve relative path.")
                return "Error: CPP_PROJECT_DIR not configured for relative path."

        result = self.replacer.replace(path=resolved_path_str, diff=diff)
        logger.info(f"FileReplacerTool processed '{resolved_path_str}'.")
        status = result.get('status', 'failure')
        message = result.get('message', 'No message provided.')
        return f"File Replace Status: {status}. Message: {message}"


class GodotProjectValidatorTool(BaseTool):
    name: str = "Godot Project Validator"
    description: str = ("Validates the entire Godot project for script parsing errors by running the editor headlessly. "
                        "Filters the output to show only errors related to the specified 'target_file_path'. "
                        "Requires 'godot_project_path' (absolute path) and 'target_file_path' (e.g., 'res://path/to/file.gd'). "
                        "Primarily checks script integrity; scene/resource load errors might also be caught.")
    args_schema: Type[BaseModel] = ProjectValidationInput

    def _run(self, godot_project_path: str, target_file_path: str) -> str:
        logger.debug(f"GodotProjectValidatorTool executing for project: {godot_project_path}, target_file: {target_file_path}")
        try:
            result = validate_godot_project(godot_project_path=godot_project_path)
            status = result.get('status', 'failure')
            errors = result.get('errors') 

            if status == 'success':
                return "Project validation successful (Exit Code 0)."
            else:
                if not errors:
                    return "Project validation failed (Non-zero exit code), but no specific errors found on stderr."

                relevant_errors = []
                normalized_target_path = target_file_path.replace("res://", "").replace("\\", "/")
                try:
                    for line in errors.splitlines():
                        if normalized_target_path in line.replace("\\", "/") or target_file_path in line:
                            relevant_errors.append(line)
                except Exception as filter_err:
                     logger.error(f"Error filtering validation output: {filter_err}", exc_info=True)
                     return f"Project validation failed. Could not filter errors. Full Errors:\n{errors}"

                if relevant_errors:
                    filtered_error_string = "\n".join(relevant_errors)
                    logger.warning(f"Project validation failed with errors relevant to {target_file_path}:\n{filtered_error_string}")
                    return f"Project validation failed. Errors related to {target_file_path}:\n{filtered_error_string}"
                else:
                    logger.info(f"Project validation failed (Exit Code {result.get('returncode', 'N/A')}), but no errors found related to {target_file_path}. Treating as success for this file.")
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
    remapping_logic_class: Type[RemappingLogic] = RemappingLogic 

    def _run(self, failed_tasks: List[Dict[str, Any]]) -> str:
        logger.debug(f"RemappingLogicTool executing with {len(failed_tasks)} failed tasks.")
        try:
            should_remap_bool = self.remapping_logic_class.should_remap_package(failed_tasks=failed_tasks)
            reason_str = ""
            feedback_str = ""

            if should_remap_bool:
                feedback_str = self.remapping_logic_class.generate_mapping_feedback(failed_tasks)
                if "High search block failure rate" in feedback_str:
                    reason_str = "High rate of search block related failures."
                elif "High validation failure rate" in feedback_str:
                    reason_str = "High rate of code validation failures."
                elif "CodeGen returned empty or non-string result" in feedback_str or "S1:" in feedback_str:
                    reason_str = "Systemic code generation failures detected."
                else:
                    reason_str = "Analysis of failure patterns suggests mapping issues."
                logger.info(f"RemappingLogicTool determined remapping IS recommended. Reason: {reason_str}")
            else:
                reason_str = "Failure patterns do not strongly indicate a systemic mapping issue requiring remapping."
                feedback_str = "No specific feedback for remapping; individual errors should be addressed if possible, or the issues are not related to mapping."
                logger.info("RemappingLogicTool determined remapping is NOT recommended.")

            advice = RemappingAdvice(
                recommend_remapping=should_remap_bool,
                reason=reason_str,
                feedback=feedback_str
            )
            return advice.model_dump_json()
        except Exception as e:
            logger.error(f"Error during RemappingLogicTool execution: {e}", exc_info=True)
            error_advice = RemappingAdvice(
                recommend_remapping=False,
                reason=f"Error during remapping analysis: {e}",
                feedback="An internal error occurred while trying to determine remapping advice."
            )
            return error_advice.model_dump_json()
