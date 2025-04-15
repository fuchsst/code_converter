# src/core/tool_interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class IFileWriter(ABC):
    """Interface for writing content to a file."""
    @abstractmethod
    def write(self, path: str, content: str) -> Dict[str, Any]:
        """
        Writes content to the specified file path.

        Args:
            path (str): The path of the file to write to.
            content (str): The content to write.

        Returns:
            Dict[str, Any]: A dictionary containing the result, e.g.,
                            {'status': 'success', 'message': 'File written.'} or
                            {'status': 'failure', 'message': 'Error details...'}.
        """
        pass

class IFileReplacer(ABC):
    """Interface for replacing content within a file."""
    @abstractmethod
    def replace(self, path: str, diff: str) -> Dict[str, Any]:
        """
        Replaces content in the specified file using a diff string.

        Args:
            path (str): The path of the file to modify.
            diff (str): The diff string in SEARCH/REPLACE format.
                       Example: "<<<<<<< SEARCH\n[find]\n=======\n[replace]\n>>>>>>> REPLACE"

        Returns:
            Dict[str, Any]: A dictionary containing the result, e.g.,
                            {'status': 'success', 'message': 'Replacement successful.'} or
                            {'status': 'failure', 'message': 'search_block not found' | 'Error details...'}.
        """
        pass

class IFileReader(ABC):
    """Interface for reading content from a file."""
    @abstractmethod
    def read(self, path: str) -> Dict[str, Any]:
        """
        Reads content from the specified file path.

        Args:
            path (str): The path of the file to read.

        Returns:
            Dict[str, Any]: A dictionary containing the result, e.g.,
                            {'status': 'success', 'content': 'File content...'} or
                            {'status': 'failure', 'message': 'File not found' | 'Error details...'}.
        """
        pass


class ISyntaxValidator(ABC):
    """Interface for validating script syntax."""
    @abstractmethod
    def validate(self, script_content: str) -> Dict[str, Any]:
        """
        Validates the syntax of the provided script content.

        Args:
            script_content (str): The script content to validate.

        Returns:
            Dict[str, Any]: A dictionary containing the result, e.g.,
                            {'status': 'success', 'errors': None} or
                            {'status': 'failure', 'errors': 'Error details...'}.
        """
        pass
