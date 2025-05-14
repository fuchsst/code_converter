# src/tools/cpp_code_analysis_tool.py
from crewai.tools import BaseTool
from src.logger_setup import get_logger
from src.core.context_manager import ContextManager
import src.config as config # For MAX_CONTEXT_TOKENS if needed, or default in method

logger = get_logger(__name__)

class CppCodeAnalysisTool(BaseTool):
    name: str = "CppCodeAnalysisTool"
    description: str = (
        "Analyzes C++ source code for a specified package ID to identify key classes, "
        "functions, data structures, and logic flow. Provides insights into the "
        "overall purpose and architecture, including a list of source files and their roles."
    )
    context_manager: ContextManager

    def __init__(self, context_manager: ContextManager, **kwargs):
        super().__init__(**kwargs)
        self.context_manager = context_manager
        self.name = "CppCodeAnalysisTool" # Ensure name is set
        self.description = ( # Ensure description is set
            "Analyzes C++ source code for a specified package ID to identify key classes, "
            "functions, data structures, and logic flow. Provides insights into the "
            "overall purpose and architecture, including a list of source files and their roles."
        )

    def _run(self, package_id: str, max_tokens: int = 8000) -> str:
        """
        Analyzes the C++ code for the specified package.
        
        Args:
            package_id: The ID of the package to analyze.
            max_tokens: Maximum tokens to consider for the source code content.
                        Defaults to 8000, but can be overridden.
                        The actual config.MAX_CONTEXT_TOKENS might be a better default
                        if the tool is expected to provide as much as possible.
                        However, agents should be mindful of overall token limits.
            
        Returns:
            A detailed analysis of the C++ code, formatted as a string.
        """
        try:
            # Get source code content using the context_manager
            # The max_tokens here is for the source code part itself.
            # The agent calling this tool should be aware of its own context window.
            source_code = self.context_manager.get_work_package_source_code_content(
                package_id, max_tokens=max_tokens
            )
            
            if not source_code:
                return f"# C++ Code Analysis for Package {package_id}\n\nNo source code available or retrieved for analysis."
            
            # Get file list with roles
            source_files_list = self.context_manager.get_source_file_list(package_id)
            
            file_list_str = "## Source Files and Roles:\n"
            if source_files_list:
                file_list_str += "\n".join(
                    [f"- `{f['file_path']}`: {f['role']}" for f in source_files_list]
                )
            else:
                file_list_str += "_No source file information available._"
            
            # The tool's role is to provide the raw information for an LLM agent to analyze.
            # So, it should return the code and file list, clearly formatted.
            # The LLM agent (CppAnalyst) will then perform the actual "analysis" (understanding).
            
            analysis_output = [
                f"# C++ Code Context for Package {package_id}",
                file_list_str,
                "\n## Combined Source Code Content (up to token limit):",
                "```cpp",
                source_code,
                "```"
            ]
            
            return "\n\n".join(analysis_output)
        
        except Exception as e:
            logger.error(f"Error in CppCodeAnalysisTool for package {package_id}: {e}", exc_info=True)
            return f"Error during C++ code analysis for package {package_id}: {e}"

# Example usage (for testing or development):
# if __name__ == '__main__':
#     # Mocking ContextManager and StateManager for standalone test
#     class MockStateManager:
#         def get_package_info(self, pkg_id):
#             if pkg_id == "test_pkg":
#                 return {"source_files": [{"file_path": "src/main.cpp", "role": "entry_point"}]}
#             return None
#         def get_work_package_root(self, pkg_id):
#             return "." # Dummy path
#         def get_config(self, key, default=None): return default

#     class MockContextManager:
#         def __init__(self):
#             self.state_manager = MockStateManager()
#         def get_work_package_source_code_content(self, package_id, max_tokens):
#             if package_id == "test_pkg":
#                 return "// Sample C++ Code\nint main() { return 0; }"
#             return None
#         def get_source_file_list(self, package_id):
#             if package_id == "test_pkg":
#                 return [{"file_path": "src/main.cpp", "role": "entry_point"}]
#             return []

#     mock_cm = MockContextManager()
#     tool = CppCodeAnalysisTool(context_manager=mock_cm)
    
#     result = tool._run(package_id="test_pkg")
#     print(result)
    
#     result_no_code = tool._run(package_id="non_existent_pkg")
#     print(result_no_code)
