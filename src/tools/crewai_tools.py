# src/tools/crewai_tools.py
from crewai.tools import BaseTool
from typing import Type, Any, List, Dict
from pydantic import BaseModel, Field

# Import the existing wrappers/logic these tools will use
from .framework_tools_wrapper import CrewAIFileWriter, CrewAIFileReader, CustomFileReplacer, GodotSyntaxValidator
from ..core.remapping_logic import RemappingLogic
from ..core.tool_interfaces import IFileWriter, IFileReader, IFileReplacer, ISyntaxValidator # For type hints
from src.logger_setup import get_logger

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
    script_content: str = Field(..., description="The GDScript content to validate.")

class RemappingCheckInput(BaseModel):
    """Input schema for RemappingLogicTool."""
    # Expecting a list of dictionaries, but Pydantic needs a more concrete type for validation.
    # Using List[Dict[str, Any]] is flexible but less strict.
    # If the structure of failed_tasks is stable, define a FailedTask Pydantic model here.
    failed_tasks: List[Dict[str, Any]] = Field(..., description="A list of dictionaries, where each dictionary represents a failed task report from Step 5 processing.")


# --- CrewAI Tool Definitions ---

class FileWriterTool(BaseTool):
    name: str = "File Writer"
    description: str = ("Writes the given content to the specified file path. "
                        "Overwrites the file if it exists, creates it if it doesn't. "
                        "Requires 'file_path' and 'content'.")
    args_schema: Type[BaseModel] = WriteFileInput
    writer: IFileWriter = CrewAIFileWriter() # Use the existing wrapper

    def _run(self, file_path: str, content: str) -> str:
        logger.debug(f"FileWriterTool executing: path='{file_path}'")
        result = self.writer.write(path=file_path, content=content)
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
        logger.debug(f"FileReplacerTool executing: path='{file_path}'")
        result = self.replacer.replace(path=file_path, diff=diff)
        status = result.get('status', 'failure')
        message = result.get('message', 'No message provided.')
        return f"File Replace Status: {status}. Message: {message}"

class GodotSyntaxValidatorTool(BaseTool):
    name: str = "Godot Syntax Validator"
    description: str = ("Validates the syntax of a given Godot GDScript code snippet using the Godot executable. "
                        "Requires 'script_content'.")
    args_schema: Type[BaseModel] = ValidateSyntaxInput
    validator: ISyntaxValidator = GodotSyntaxValidator() # Use the existing wrapper

    def _run(self, script_content: str) -> str:
        logger.debug("GodotSyntaxValidatorTool executing.")
        result = self.validator.validate(script_content=script_content)
        status = result.get('status', 'failure')
        errors = result.get('errors')
        if status == 'success':
            return "Syntax validation successful."
        else:
            return f"Syntax validation failed. Errors:\n{errors or 'No specific error details provided.'}"

class RemappingLogicTool(BaseTool):
    name: str = "Remapping Logic Analyzer"
    description: str = ("Analyzes a list of failed task reports for a work package to determine if remapping (re-running Step 4) is advisable. "
                        "Requires 'failed_tasks' (a list of dictionaries).")
    args_schema: Type[BaseModel] = RemappingCheckInput
    # No wrapper needed, directly use the static method from RemappingLogic
    remapping_logic: Type[RemappingLogic] = RemappingLogic

    def _run(self, failed_tasks: List[Dict[str, Any]]) -> str:
        logger.debug(f"RemappingLogicTool executing with {len(failed_tasks)} failed tasks.")
        try:
            should_remap = self.remapping_logic.should_remap_package(failed_tasks=failed_tasks)
            if should_remap:
                # Optionally generate feedback here too, or let the agent do it based on this output
                feedback = self.remapping_logic.generate_mapping_feedback(failed_tasks)
                logger.info("RemappingLogicTool determined remapping IS recommended.")
                return f"Remapping Recommended: Yes. Reason: Failure patterns suggest mapping issues.\nFeedback:\n{feedback}"
            else:
                logger.info("RemappingLogicTool determined remapping is NOT recommended.")
                return "Remapping Recommended: No. Failure patterns do not strongly indicate a need for remapping."
        except Exception as e:
            logger.error(f"Error during RemappingLogicTool execution: {e}", exc_info=True)
            return f"Remapping Recommended: Error. Failed to analyze failures: {e}"

# Example of how to instantiate (likely done in Step5Executor)
# file_writer_tool = FileWriterTool()
# file_replacer_tool = FileReplacerTool()
# syntax_validator_tool = GodotSyntaxValidatorTool()
# remapping_tool = RemappingLogicTool()
