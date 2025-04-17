# src/tools/framework_tools_wrapper.py
import os
import re
from typing import Any, Dict
# Import the interfaces these wrappers will implement
from ..core.tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator
# Import the actual local validator function
from .godot_validator_tool import validate_gdscript_syntax
# Import standard CrewAI tools for read/write
from crewai_tools import FileWriterTool, FileReadTool
from src.logger_setup import get_logger

logger = get_logger(__name__)

# --- Concrete Tool Wrappers ---

class CrewAIFileWriter(IFileWriter):
    """Implements IFileWriter using crewai_tools.FileWriteTool."""
    def __init__(self):
        # Instantiate the CrewAI tool internally
        # Note: FileWriteTool takes directory and filename separately in its _run method.
        self._tool = FileWriterTool()
        logger.debug("Initialized CrewAIFileWriter using crewai_tools.FileWriterTool")

    def write(self, path: str, content: str) -> Dict[str, Any]:
        logger.debug(f"CrewAIFileWriter: Writing to path='{path}'")
        try:
            # Separate directory and filename for FileWriteTool
            directory = os.path.dirname(path) or "." # Default to current dir if no dir part
            filename = os.path.basename(path)
            if not filename:
                 raise ValueError("Path must include a filename.")

            # Call the tool's run method
            result_message = self._tool._run(directory=directory, file_name=filename, text=content)
            logger.info(f"CrewAI FileWriteTool successful for: {path}. Message: {result_message}")
            # Assume success if no exception, result_message might confirm
            return {'status': 'success', 'message': result_message}
        except Exception as e:
            logger.error(f"CrewAI FileWriteTool failed for {path}: {e}", exc_info=True)
            return {'status': 'failure', 'message': f'FileWriteTool error: {e}'}

class CrewAIFileReader(IFileReader):
    """Implements IFileReader using crewai_tools.FileReadTool."""
    def __init__(self):
        self._tool = FileReadTool()
        logger.debug("Initialized CrewAIFileReader using crewai_tools.FileReadTool")

    def read(self, path: str) -> Dict[str, Any]:
        logger.debug(f"CrewAIFileReader: Reading from path='{path}'")
        try:
            # Call the tool's run method
            content = self._tool._run(file_path=path)
            logger.info(f"CrewAI FileReadTool successful for: {path}")
            return {'status': 'success', 'content': content}
        except FileNotFoundError:
            logger.warning(f"CrewAI FileReadTool: File not found at {path}")
            return {'status': 'failure', 'message': 'File not found'}
        except Exception as e:
            logger.error(f"CrewAI FileReadTool failed for {path}: {e}", exc_info=True)
            return {'status': 'failure', 'message': f'FileReadTool error: {e}'}

class CustomFileReplacer(IFileReplacer):
    """Implements IFileReplacer using custom Python logic for SEARCH/REPLACE blocks."""
    def __init__(self):
        logger.debug("Initialized CustomFileReplacer")

    def replace(self, path: str, diff: str) -> Dict[str, Any]:
        logger.debug(f"CustomFileReplacer: Replacing in path='{path}', diff='{diff[:60]}...'")
        try:
            # 1. Parse the diff string
            # Use regex that handles potential leading/trailing whitespace within blocks
            match = re.match(r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE\s*$", diff, re.DOTALL)
            if not match:
                logger.error(f"Invalid diff format provided for {path}. Diff:\n{diff}")
                return {'status': 'failure', 'message': 'Invalid diff format'}

            search_block = match.group(1)
            replace_block = match.group(2)
            # logger.debug(f"Parsed search_block (len={len(search_block)}), replace_block (len={len(replace_block)})")

            # 2. Read the original file content
            if not os.path.exists(path):
                 logger.warning(f"CustomFileReplacer: File not found at {path}")
                 return {'status': 'failure', 'message': 'File not found'}

            # Use UTF-8, handle potential errors
            try:
                # Read lines to handle different line endings consistently
                with open(path, 'r', encoding='utf-8') as f:
                    original_lines = f.readlines()
                original_content = "".join(original_lines) # Join back for string searching
            except Exception as read_err:
                 logger.error(f"CustomFileReplacer: Error reading file {path}: {read_err}", exc_info=True)
                 return {'status': 'failure', 'message': f'Error reading file: {read_err}'}

            # 3. Check if search_block exists (handle potential line ending differences)
            # Normalize line endings in both search_block and original_content for comparison
            normalized_search_block = search_block.replace('\r\n', '\n')
            normalized_original_content = original_content.replace('\r\n', '\n')

            if normalized_search_block not in normalized_original_content:
                logger.warning(f"CustomFileReplacer: Normalized search_block not found in {path}")
                # Log snippets for debugging (be careful with large blocks)
                search_snippet = normalized_search_block[:100].replace('\n', '\\n') + ('...' if len(normalized_search_block) > 100 else '')
                content_snippet = normalized_original_content[:200].replace('\n', '\\n') + ('...' if len(normalized_original_content) > 200 else '')
                logger.debug(f"Search snippet (normalized): '{search_snippet}'")
                logger.debug(f"Content snippet (normalized): '{content_snippet}'")
                return {'status': 'failure', 'message': 'search_block not found'}

            # 4. Perform the replacement (only first occurrence) using normalized content
            # Note: This replaces based on normalized content, which might be slightly risky if
            # the only difference was line endings, but necessary for reliable matching.
            # The replacement block uses its original line endings.
            modified_content = normalized_original_content.replace(normalized_search_block, replace_block, 1)

            # 5. Write the new content back to the file, preserving original line endings if possible,
            #    or defaulting to OS default. Python's 'w' mode usually handles this.
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            except Exception as write_err:
                 logger.error(f"CustomFileReplacer: Error writing file {path}: {write_err}", exc_info=True)
                 return {'status': 'failure', 'message': f'Error writing file: {write_err}'}

            logger.info(f"CustomFileReplacer successful for: {path}")
            return {'status': 'success', 'message': f'Content replaced successfully in {os.path.basename(path)}.'}

        except Exception as e:
            logger.error(f"CustomFileReplacer failed unexpectedly for {path}: {e}", exc_info=True)
            return {'status': 'failure', 'message': f'Unexpected replace error: {e}'}


class GodotSyntaxValidator(ISyntaxValidator):
    """Implements ISyntaxValidator using the validate_gdscript_syntax function."""
    def validate(self, script_content: str) -> Dict[str, Any]:
        logger.debug("GodotSyntaxValidator: Calling local validate_gdscript_syntax function.")
        # This wrapper *does* call the actual implementation directly,
        # as it's a local Python function.
        try:
            # Ensure the function exists and is callable
            if not callable(validate_gdscript_syntax):
                 raise TypeError("validate_gdscript_syntax is not callable")

            result = validate_gdscript_syntax(script_content=script_content)
            # Ensure the result matches the expected interface structure
            if not isinstance(result, dict) or 'status' not in result:
                 logger.error(f"validate_gdscript_syntax returned unexpected format: {result}")
                 return {'status': 'failure', 'errors': 'Validator function returned invalid format.'}
            return result
        except Exception as e:
            logger.error(f"Error calling validate_gdscript_syntax: {e}", exc_info=True)
            return {'status': 'failure', 'errors': f"Error during validation call: {e}"}
