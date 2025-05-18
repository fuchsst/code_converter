# src/tools/cpp_file_reader_tool.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any
import os
import src.config as config
from src.tools.framework_tools_wrapper import CrewAIFileReader # Reusing the existing reader logic
from src.core.tool_interfaces import IFileReader
from src.logger_setup import get_logger

logger = get_logger(__name__)

class CppReadFileInput(BaseModel):
    """Input schema for CppFileReaderTool."""
    file_path: str = Field(..., description="The relative path to the C++ source file within the C++ project directory.")

class CppFileReaderTool(BaseTool):
    name: str = "C++ File Reader"
    description: str = (
        "Reads the entire content of a specified C++ source file. "
        "Requires 'file_path' relative to the C++ project directory (defined by CPP_PROJECT_DIR). "
        "Returns the file content or an error message if reading fails."
    )
    args_schema: Type[BaseModel] = CppReadFileInput
    reader: IFileReader = CrewAIFileReader()

    def _run(self, file_path: str) -> str:
        logger.debug(f"CppFileReaderTool executing: received relative path='{file_path}'")
        
        if os.path.isabs(file_path):
            logger.warning(f"CppFileReaderTool received an absolute path '{file_path}'. It's designed for relative paths to CPP_PROJECT_DIR. Proceeding with absolute path.")
            resolved_path_str = file_path
        else:
            cpp_project_dir = config.CPP_PROJECT_DIR
            if not cpp_project_dir:
                logger.error("CppFileReaderTool: CPP_PROJECT_DIR not configured. Cannot resolve relative path.")
                return "Error: CPP_PROJECT_DIR not configured."
            
            resolved_path_str = os.path.abspath(os.path.join(cpp_project_dir, file_path))
            logger.debug(f"CppFileReaderTool: Resolved relative path '{file_path}' to absolute path: '{resolved_path_str}' using CPP_PROJECT_DIR.")

        if not os.path.exists(resolved_path_str):
            logger.warning(f"CppFileReaderTool: File not found at resolved path: {resolved_path_str} (original: {file_path})")
            return f"Error: File not found at '{resolved_path_str}' (from original path '{file_path}')."
        if not os.path.isfile(resolved_path_str):
            logger.warning(f"CppFileReaderTool: Path is not a file: {resolved_path_str}")
            return f"Error: Path '{resolved_path_str}' is not a file."

        result = self.reader.read(path=resolved_path_str)
        status = result.get('status', 'failure')

        if status == 'success':
            return result.get('content', '')
        else:
            message = result.get('message', f'Failed to read file {resolved_path_str}.')
            logger.warning(f"CppFileReaderTool failed for path '{resolved_path_str}' (original: '{file_path}'): {message}")
            return f"Error: {message}"
