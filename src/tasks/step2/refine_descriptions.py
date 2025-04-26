# src/tasks/refine_descriptions.py
from crewai import Task, Agent
from typing import Dict, Any, List
from src.logger_setup import get_logger
import json
from collections import Counter
from pydantic import BaseModel, Field

logger = get_logger(__name__)

# --- Pydantic Model for Structured Output ---
# We expect a dictionary where keys are package names (strings)
# and values are the refined description strings.
class RefinedDescriptionsOutput(BaseModel):
    package_descriptions: Dict[str, str] = Field(..., description="A dictionary mapping package names to their refined descriptions.")
    package_order: List[str] = Field(..., description="A list of package names in the order they should be migrated (packges witb core functionality that is used by a lot of other packages).")



class RefineDescriptionsTask:
    """
    CrewAI Task definition for refining package descriptions based on
    the context of all initially described packages.
    """
    def create_task(self, agent: Agent, all_packages_data: Dict[str, Any]) -> Task:
        """
        Creates the CrewAI Task instance for refining descriptions.

        Args:
            agent (Agent): The PackageRefinementAgent instance.
            all_packages_data (Dict[str, Any]): A dictionary containing the data
                                                 for all packages, including their
                                                 initial descriptions and file roles.

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

        return Task(
            description=(
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
                "Generate a **single, valid JSON object** as a string. This object must be a dictionary where:\n"
                "*   The **keys** are the package names (e.g., \"package_1\", \"package_2\") exactly as they appear in the input summary.\n"
                "*   The **values** are the **refined** `package_description` strings for each corresponding package.\n"
                "Include an entry for **every package** present in the input summary.\n"
                "**CRITICAL:** Your output MUST be ONLY the raw JSON object string. No introductory text, no explanations, no markdown code fences (like ```json), just the JSON dictionary itself starting with `{` and ending with `}`."
            ),
            expected_output=(
                "A **single, valid JSON object string** conforming to the RefinedDescriptionsOutput Pydantic model. "
                "It must contain the keys 'package_descriptions' (a dictionary mapping package names to refined descriptions) and 'package_order' (a list of package names in migration order). "
                "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
                "\n\nExample of the required raw JSON output format:\n"
                "{\n"
                "  \"package_descriptions\": {\n"
                "    \"core_utils\": \"Refined description for core utilities, used by many other packages.\",\n"
                "    \"feature_A\": \"Refined description for feature A, depends on core_utils.\",\n"
                "    \"feature_B\": \"Refined description for feature B, depends on core_utils and feature_A.\"\n"
                "    // ... entry for every package in the input ...\n"
                "  },\n"
                "  \"package_order\": [\"core_utils\", \"feature_A\", \"feature_B\"]\n"
                "  // ... all package names listed in a logical migration order ...\n"
                "}"
            ),
            agent=agent,
            output_pydantic=RefinedDescriptionsOutput # Use the Pydantic model for validation
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from src.agents.package_refiner import PackageRefinementAgent
#
#     # Dummy data for testing
#     test_packages = {
#         "package_1": {
#             "description": "Initial desc for pkg 1",
#             "file_roles": [{"file_path": "pkg1/a.cpp", "role": "Does thing A"}],
#             "files": ["pkg1/a.cpp"],
#             "total_tokens": 100
#         },
#         "package_2": {
#             "description": "Initial desc for pkg 2 - utils",
#             "file_roles": [{"file_path": "pkg2/b.h", "role": "Helper functions"}],
#             "files": ["pkg2/b.h"],
#             "total_tokens": 50
#         }
#     }
#
#     # Placeholder LLM
#     class MockLLM:
#         model_name = "mock-llm"
#         def invoke(self, *args, **kwargs):
#             dummy_output = {
#                 "package_1": "Refined: Package 1 implements core feature A, using utils from Pkg 2.",
#                 "package_2": "Refined: Package 2 provides common utility functions."
#             }
#             return json.dumps(dummy_output)
#
#     mock_llm = MockLLM()
#     agent_creator = PackageRefinementAgent()
#     refiner_agent = agent_creator.get_agent(llm_instance=mock_llm)
#
#     task_creator = RefineDescriptionsTask()
#     refine_task = task_creator.create_task(refiner_agent, test_packages)
#
#     print("RefineDescriptionsTask created:")
#     # print(f"Description: {refine_task.description}") # Can be very long
#     print(f"Expected Output Format Hint: {refine_task.expected_output}")
#
#     # Example of how the executor might run it (simplified)
#     # from crewai import Crew, Process
#     # crew = Crew(agents=[refiner_agent], tasks=[refine_task], verbose=1)
#     # result = crew.kickoff()
#     # print("\nMock Crew Kickoff Result (Refinement):")
#     # print(result) # Should print the dummy JSON dictionary string
