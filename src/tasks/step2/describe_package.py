# src/tasks/describe_package.py
from crewai import Task, Agent
from typing import List, Optional
from src.models.package_models import PackageDescriptionOutput
from src.logger_setup import get_logger

logger = get_logger(__name__)


def create_describe_package_task(
                agent: Agent,
                package_files: List[str],
                context: str,
                instructions: Optional[str] = None) -> Task:
    """
    Creates the CrewAI Task instance for describing the package.

    Args:
        agent (Agent): The PackageDescriberAgent instance.
        package_files (List[str]): The list of file paths in this package.
        context (str): The context string containing code snippets or interfaces
                        for the files in the package, assembled by ContextManager.
        instructions (Optional[str]): General instructions to prepend to the task description.

    Returns:
        Task: The CrewAI Task object.
    """
    logger.info(f"Creating DescribePackageTask for agent: {agent.role} with {len(package_files)} files.")
    file_list_str = "\n".join([f"- `{f}`" for f in package_files])

    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions:**\n{instructions}\n\n---\n\n"

    full_description += (
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
    )

    return Task(
        description=full_description,
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
