# src/tasks/identify_packages.py
from crewai import Task
from src.logger_setup import get_logger
from pydantic import BaseModel, Field
from typing import List, Dict
# Import agent definition if needed for type hinting or direct reference
# from agents.package_identifier import PackageIdentifierAgent # Example

logger = get_logger(__name__)

# Define Pydantic Models for structured output
class PackageDetails(BaseModel):
    description: str = Field(..., description="A brief string describing the package's likely purpose.")
    files: List[str] = Field(..., description="A list of file paths (relative to the project root) belonging to this package.")

class WorkPackagesOutput(BaseModel):
    packages: Dict[str, PackageDetails] = Field(..., description="A dictionary where keys are unique package identifiers (e.g., 'package_1', 'core_utils') and values are objects containing package description and file list.")


class IdentifyWorkPackagesTask:
    """
    CrewAI Task definition for identifying work packages from the C++ include graph.
    """
    def create_task(self, agent):
        """
        Creates the CrewAI Task instance for identifying work packages.

        Args:
            agent (Agent): The PackageIdentifierAgent instance responsible for this task.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating IdentifyWorkPackagesTask for agent: {agent.role}")
        return Task(
            description=(
                "Analyze the provided C++ project's include graph (JSON string in 'include_graph_json' input). "
                "Your goal is to identify logical, reasonably self-contained work packages (groups of related files) "
                "suitable for incremental conversion to Godot. "
                "Consider file relationships shown in the graph and directory structure implied by paths. "
                "Prioritize grouping files based on features or modules, aiming to minimize includes *between* packages. "
                "Each package should represent a manageable unit for conversion (e.g., 5-20 related files, depending on complexity). "
                "Provide a brief 'description' for each identified package explaining its likely purpose."
            ),
            expected_output=( # Keep this for LLM guidance, but Pydantic model enforces structure
                "A JSON object structured according to the WorkPackagesOutput Pydantic model. Keys are unique package identifiers "
                "and values are objects containing: \n"
                "1. 'description': A brief string describing the package's likely purpose.\n"
                "2. 'files': A list of file paths (relative to the project root) belonging to this package."
                "\nExample:\n"
                "{\n"
                "  \"packages\": {\n" # Note: Added 'packages' key due to wrapper model
                "    \"audio_subsystem\": {\n"
                "      \"description\": \"Handles sound loading and playback.\",\n"
                "      \"files\": [\"src/audio/manager.cpp\", \"src/audio/sound.h\", ...]\n"
                "    },\n"
                "    \"rendering_core\": {\n"
                "      \"description\": \"Core rendering pipeline components.\",\n"
                "      \"files\": [\"src/render/renderer.cpp\", \"src/render/shader.h\", ...]\n"
                "    }\n"
                "  }\n"
                "}"
            ),
            agent=agent,
            # inputs={'include_graph_json': '...'} # Actual input provided during crew kickoff
            # context=None # Context is typically managed externally or via agent memory
            output_pydantic=WorkPackagesOutput
            # output_file="analysis_results/work_packages.json" # CrewAI can optionally save output
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.package_identifier import PackageIdentifierAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = PackageIdentifierAgent()
#     package_agent = agent_creator.get_agent()
#
#     task_creator = IdentifyWorkPackagesTask()
#     identify_task = task_creator.create_task(package_agent)
#     print("IdentifyWorkPackagesTask created:")
#     print(f"Description: {identify_task.description}")
#     print(f"Expected Output: {identify_task.expected_output}")
