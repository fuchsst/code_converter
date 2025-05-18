# src/tools/godot_file_reader_tool.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import os
import src.config as config
from src.tools.framework_tools_wrapper import CrewAIFileReader # Reusing the existing reader logic
from src.core.tool_interfaces import IFileReader
from src.logger_setup import get_logger

logger = get_logger(__name__)

class GodotReadFileInput(BaseModel):
    """Input schema for GodotFileReaderTool."""
    file_path: str = Field(..., description="The path to the Godot project file. Can be 'res://', absolute, or relative to GODOT_PROJECT_DIR.")

class GodotFileReaderTool(BaseTool):
    name: str = "Godot Project File Reader"
    description: str = (
        "Reads the entire content of a specified file within a Godot project. "
        "Interprets 'res://' paths relative to GODOT_PROJECT_DIR. "
        "Other relative paths are also considered relative to GODOT_PROJECT_DIR. "
        "Absolute paths are used as is. Requires 'file_path'. "
        "Returns the file content or an error message if reading fails."
    )
    args_schema: Type[BaseModel] = GodotReadFileInput
    reader: IFileReader = CrewAIFileReader()

    def _run(self, file_path: str) -> str:
        logger.debug(f"GodotFileReaderTool executing: received path='{file_path}'")
        resolved_path_str: str
        
        if os.path.isabs(file_path):
            resolved_path_str = file_path
            logger.debug(f"GodotFileReaderTool: Path '{file_path}' is absolute.")
        elif file_path.startswith("res://"):
            godot_project_dir = config.GODOT_PROJECT_DIR
            if not godot_project_dir:
                logger.error("GodotFileReaderTool: GODOT_PROJECT_DIR not configured. Cannot resolve 'res://' path.")
                return "Error: GODOT_PROJECT_DIR not configured for 'res://' path."
            
            relative_path = file_path[len("res://"):]
            resolved_path_str = os.path.abspath(os.path.join(godot_project_dir, relative_path))
            logger.debug(f"GodotFileReaderTool: Converted 'res://' path '{file_path}' to absolute path: '{resolved_path_str}'")
        else: # Relative path, not starting with res://, assume relative to GODOT_PROJECT_DIR
            godot_project_dir = config.GODOT_PROJECT_DIR
            if not godot_project_dir:
                logger.error("GodotFileReaderTool: GODOT_PROJECT_DIR not configured. Cannot resolve relative path.")
                return "Error: GODOT_PROJECT_DIR not configured for relative path."
            
            resolved_path_str = os.path.abspath(os.path.join(godot_project_dir, file_path))
            logger.debug(f"GodotFileReaderTool: Resolved relative path '{file_path}' to absolute path: '{resolved_path_str}' using GODOT_PROJECT_DIR.")

        if not os.path.exists(resolved_path_str):
            logger.warning(f"GodotFileReaderTool: File not found at resolved path: {resolved_path_str} (original: {file_path})")
            return f"Error: File not found at '{resolved_path_str}' (from original path '{file_path}')."
        if not os.path.isfile(resolved_path_str):
            logger.warning(f"GodotFileReaderTool: Path is not a file: {resolved_path_str}")
            return f"Error: Path '{resolved_path_str}' is not a file."
            
        result = self.reader.read(path=resolved_path_str)
        status = result.get('status', 'failure')

        if status == 'success':
            return result.get('content', '')
        else:
            message = result.get('message', f'Failed to read file {resolved_path_str}.')
            logger.warning(f"GodotFileReaderTool failed for path '{resolved_path_str}' (original: '{file_path}'): {message}")
            return f"Error: {message}"
