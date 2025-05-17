# src/tasks/refine_descriptions.py
from crewai import Task, Agent
from typing import Dict, Any, Optional
from src.models.package_models import RefinedDescriptionsOutput
from src.logger_setup import get_logger
from collections import Counter

logger = get_logger(__name__)



def create_refined_descriptions_task(agent: Agent,
                                     all_packages_data: Dict[str, Any],
                                     instructions: Optional[str] = None) -> Task:
    """
    Creates the CrewAI Task instance for refining descriptions.

    Args:
        agent (Agent): The PackageRefinementAgent instance.
        all_packages_data (Dict[str, Any]): A dictionary containing the data
                                                for all packages, including their
                                                initial descriptions and file roles.
        instructions (Optional[str]): General instructions to prepend to the task description.

    Returns:
        Task: The CrewAI Task object.
    """
    logger.info(f"Creating RefineDescriptionsTask for agent: {agent.role} with {len(all_packages_data)} packages.")

    # --- Calculate file uniqueness ---
    file_counts = Counter()
    for pkg_data in all_packages_data.values():
        files = pkg_data.get("files", [])
        if isinstance(files, list):
            # Use set to count each file only once per package in this stage
            file_counts.update(set(files))

    # --- Construct detailed context string ---
    context_parts = []
    for pkg_name, pkg_data in all_packages_data.items():
        initial_desc = pkg_data.get("description", "N/A")
        file_roles_list = pkg_data.get("file_roles", [])
        files_in_package = pkg_data.get("files", [])

        # Create a lookup map for file roles
        role_map = {role.get("file_path"): role.get("role", "N/A")
                    for role in file_roles_list if isinstance(role, dict)}

        unique_files = sorted([f for f in files_in_package if file_counts.get(f, 0) == 1])
        shared_files = sorted([f for f in files_in_package if file_counts.get(f, 0) > 1])

        part = f"--- Package: {pkg_name} ---\n"
        part += f"Initial Description: {initial_desc}\n"

        part += "Unique Files (Only in this package):\n"
        if unique_files:
            part += "\n".join([f"- {f}: {role_map.get(f, 'Role not found')}" for f in unique_files]) + "\n"
        else:
            part += "- None\n"

        part += "Shared Files (Also in other packages):\n"
        if shared_files:
            part += "\n".join([f"- {f}: {role_map.get(f, 'Role not found')}" for f in shared_files]) + "\n"
        else:
            part += "- None\n"
        # Removed the separate File Roles section

        context_parts.append(part)

    context_str = "\n\n".join(context_parts) # Separate packages by double newline
    # --- End context string construction ---


    # Estimate token count (optional, but good for large contexts)
    # from src.core.context_manager import count_tokens # Assuming this utility exists
    # context_tokens = count_tokens(context_str)
    # logger.info(f"Estimated token count for refinement context: {context_tokens}")

    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions:**\n{instructions}\n\n---\n\n"

    full_description += (
        "You are a Holistic Software Architect reviewing initial package descriptions.\n"
        "Below is a structured summary of all identified software packages. Each package includes its initial description, lists of files unique to it (with their roles), and files shared with other packages (with their roles):\n"
        "--- START OF ALL PACKAGES SUMMARY ---\n"
        f"{context_str}\n"
        "--- END OF ALL PACKAGES SUMMARY ---\n\n"
        "Your task is to refine the `package_description` for **each** package listed above.\n"
        "Consider the following when refining:\n"
        "1.  **Uniqueness vs. Shared Files:** Pay close attention to the 'Unique Files' list and their roles. These often define the core, specific purpose of the package. 'Shared Files' and their roles might indicate dependencies, common utilities, or interfaces used by multiple packages.\n"
        "2.  **Context:** How does this package relate to the others based on shared files and overall structure? Does the initial description capture its role in the larger system?\n"
        "3.  **Clarity & Conciseness:** Is the description clear, concise (ideally 1-3 sentences), and easy to understand?\n"
        "4.  **Accuracy:** Does the description accurately reflect the functionality suggested by the file roles, especially considering the unique files?\n"
        "5.  **Consistency:** Ensure descriptions use consistent terminology where appropriate.\n\n"
        "**Output Requirements:**\n"
        "Generate a **single, valid JSON object** as a string. This object must contain a single key `package_descriptions` whose value is a list of objects. Each object in the list must have:\n"
        "*   `package_id`: The unique identifier for the package (e.g., \"package_1\", \"package_2\") exactly as it appears in the input summary.\n"
        "*   `package_description`: The **refined** description string for the corresponding package.\n"
        "Include an entry for **every package** present in the input summary.\n"
        "**CRITICAL:** Your output MUST be ONLY the raw JSON object string. No introductory text, no explanations, no markdown code fences (like ```json), just the JSON dictionary itself starting with `{` and ending with `}`."
    )

    return Task(
        description=full_description, # Use the combined description
        expected_output=(
            "A **single, valid JSON object string** conforming to the RefinedDescriptionsOutput Pydantic model. "
            "It must contain the key 'package_descriptions' whose value is a list of objects, each with 'package_id' and 'package_description'. "
            "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
            "\n\nExample of the required raw JSON output format:\n"
            "{\n"
            "  \"package_descriptions\": [\n"
            "    { \"package_id\": \"core_utils\", \"package_description\": \"Refined description for core utilities, used by many other packages.\" },\n"
            "    { \"package_id\": \"feature_A\", \"package_description\": \"Refined description for feature A, depends on core_utils.\" },\n"
            "    { \"package_id\": \"feature_B\", \"package_description\": \"Refined description for feature B, depends on core_utils and feature_A.\" }\n"
            "    // ... entry for every package in the input ...\n"
            "  ]\n"
            "}"
        ),
        agent=agent,
        output_pydantic=RefinedDescriptionsOutput # Use the Pydantic model for validation
    )
