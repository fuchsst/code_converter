# src/agents/structure_definer.py
from crewai import Agent
from logger_setup import get_logger
import config
# Assuming api_utils handles LLM configuration and calls via CrewAI's mechanism

logger = get_logger(__name__)


class StructureDefinerAgent:
    """
    CrewAI Agent responsible for proposing a Godot 4.x project structure
    (nodes, scenes, script organization) for a given C++ work package.
    """
    def __init__(self):
        # The actual LLM instance will be managed by CrewAI based on its setup
        # or potentially passed during agent creation if customizing LLM per agent.
        logger.info(f"Initializing StructureDefinerAgent (LLM configuration managed by CrewAI/global setup using model like: {config.MAPPER_MODEL})") # Assuming mapper model is suitable

    def get_agent(self):
        """Creates and returns the CrewAI Agent instance."""
        return Agent(
            role="Godot Architecture Designer",
            goal=(
                "Based on the provided C++ work package definition (file list, description) and "
                "selected C++ code snippets, propose a logical Godot 4.x project structure as a **JSON object**, "
                f"adhering to SOLID principles and promoting good separation of concerns. The target language for scripts should be {config.TARGET_LANGUAGE}. "
                "The JSON output must conform to the structure specified in the task's expected_output, including keys like `target_language`, `base_directory`, `scenes` (with `file_path`, `root_node_name`, `root_node_type`, `script_path`, `children`, `mapping_notes`), and `scripts` (with `file_path`, `attached_to_node`, `attached_to_scene`, `responsibilities`, `mapping_notes`). "
                "Analyze the C++ code to suggest appropriate scene/node hierarchies. Crucially, propose script responsibilities that are focused and decoupled (e.g., separate scripts for input handling, state management, UI logic, core mechanics) rather than large, monolithic scripts. Format the entire proposal as a single, valid JSON object."
            ),
            backstory=(
                "You are a seasoned game developer and software architect with deep expertise in Godot Engine 4.x architecture, "
                "SOLID design principles, and best practices for maintainable game code. You excel at translating requirements and existing code structures "
                "(even from different languages/engines like C++) into well-organized, decoupled Godot projects. "
                "You understand scene composition, node inheritance, signal usage, and how to structure scripts effectively for clarity, testability, and performance, avoiding overly complex single scripts."
            ),
            # llm=... # Let CrewAI handle LLM based on global config or Crew setup
            verbose=True,
            allow_delegation=False, # This agent focuses on its specific design task
            # memory=True # Consider if memory is needed across potential retries for the *same* package
            # tools=[] # This agent primarily analyzes context and generates structure, likely no tools needed
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = StructureDefinerAgent()
#     structure_agent = agent_creator.get_agent()
#     print("StructureDefinerAgent created:")
#     print(f"Role: {structure_agent.role}")
#     print(f"Goal: {structure_agent.goal}")
