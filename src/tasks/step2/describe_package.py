# src/tasks/describe_package.py
from crewai import Task, Agent
from typing import List, Dict
from src.logger_setup import get_logger
import json
from pydantic import BaseModel, Field # Import Pydantic components

logger = get_logger(__name__)

# --- Pydantic Models for Structured Output ---
class FileRole(BaseModel):
    """Defines the structure for describing a single file's role."""
    file_path: str = Field(..., description="The relative path of the file.")
    role: str = Field(..., description="A brief description of the file's role within the package.")

class PackageDescriptionOutput(BaseModel):
    """Defines the overall expected JSON output structure for the task."""
    package_description: str = Field(..., description="A concise description of the package's overall purpose.")
    file_roles: List[FileRole] = Field(..., description="A list detailing the role of each file in the package.")


class DescribePackageTask:
    """
    CrewAI Task definition for generating a description and file roles
    for a given code package using an LLM agent.
    """
    def create_task(self, agent: Agent, package_files: List[str], context: str) -> Task:
        """
        Creates the CrewAI Task instance for describing the package.

        Args:
            agent (Agent): The PackageDescriberAgent instance.
            package_files (List[str]): The list of file paths in this package.
            context (str): The context string containing code snippets or interfaces
                           for the files in the package, assembled by ContextManager.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating DescribePackageTask for agent: {agent.role} with {len(package_files)} files.")
        file_list_str = "\n".join([f"- `{f}`" for f in package_files])

        return Task(
            description=(
                "You are an expert C++ software architect analyzing code for conversion.\n"
                "Analyze the provided C++ code package context. The package contains the following files:\n"
                f"{file_list_str}\n\n"
                "Context (may contain full code, interfaces, or just paths):\n"
                "--- START OF CONTEXT ---\n"
                f"{context}\n"
                "--- END OF CONTEXT ---\n\n"
                "Based *only* on the provided context, your goal is to generate a JSON object.\n"
                "This JSON object MUST contain:\n"
                "1.  `package_description`: A concise (1-2 sentence) description of the package's overall purpose or functionality.\n"
                "2.  `file_roles`: A list of JSON objects, one for each file listed above. Each object MUST have:\n"
                "    *   `file_path`: The relative path of the file (exactly as listed above).\n"
                "    *   `role`: A brief (max 15 words) description of that specific file's likely role *within this package*.\n\n"
                "Ensure the `file_roles` list contains an entry for **every file** listed in the input.\n"
                "**CRITICAL:** Your output MUST be ONLY the raw JSON object string. No introductory text, no explanations, no markdown code fences (like ```json), just the JSON itself starting with `{` and ending with `}`."
            ),
            expected_output=(
                "A **single, valid JSON object string** adhering strictly to the specified structure: "
                "{ \"package_description\": \"...\", \"file_roles\": [ { \"file_path\": \"...\", \"role\": \"...\" }, ... ] }. "
                "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
                "\n\nExample of the required raw JSON output format:\n"
                "{\n"
                "  \"package_description\": \"Handles user authentication and session management.\",\n"
                "  \"file_roles\": [\n"
                "    {\n"
                "      \"file_path\": \"auth/login.cpp\",\n"
                "      \"role\": \"Implements the user login logic and validation.\"\n"
                "    },\n"
                "    {\n"
                "      \"file_path\": \"auth/session.h\",\n"
                "      \"role\": \"Defines the session data structure and management functions.\"\n"
                "    }\n"
                "    // ... entry for every file ...\n"
                "  ]\n"
                "}"
            ),
            agent=agent,
            output_json=PackageDescriptionOutput
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from src.agents.package_describer import PackageDescriberAgent # Need agent for task
#
#     # Placeholder LLM for structure testing
#     class MockLLM:
#         model_name = "mock-llm"
#         def invoke(self, *args, **kwargs): # Mock invoke if needed by Crew
#             # Return a dummy JSON string matching the expected output structure
#             dummy_output = {
#                 "package_description": "Mock description for testing.",
#                 "file_roles": [
#                     {"file_path": "test/file1.cpp", "role": "Mock role for file 1."},
#                     {"file_path": "test/file2.h", "role": "Mock role for file 2."}
#                 ]
#             }
#             return json.dumps(dummy_output)
#
#     mock_llm = MockLLM()
#     agent_creator = PackageDescriberAgent(llm=mock_llm)
#     describer_agent = agent_creator.get_agent()
#
#     # Dummy context and files for testing
#     test_files = ["test/file1.cpp", "test/file2.h"]
#     test_context = """
#     // File: test/file1.cpp
#     void function1() {}
#
#     // File: test/file2.h
#     class Class2 {};
#     """
#
#     task_creator = DescribePackageTask()
#     describe_task = task_creator.create_task(describer_agent, test_files, test_context)
#     print("DescribePackageTask created:")
#     print(f"Description: {describe_task.description}")
#     print(f"Expected Output: {describe_task.expected_output}")
#
#     # Example of how the executor might run it (simplified)
#     # from crewai import Crew, Process
#     # crew = Crew(agents=[describer_agent], tasks=[describe_task], verbose=1)
#     # result = crew.kickoff() # This would call the mock LLM's invoke
#     # print("\nMock Crew Kickoff Result:")
#     # print(result) # Should print the dummy JSON string
