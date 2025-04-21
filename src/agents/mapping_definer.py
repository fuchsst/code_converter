# src/agents/mapping_definer.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
# Assuming api_utils handles LLM configuration and calls via CrewAI's mechanism

logger = get_logger(__name__)


class MappingDefinerAgent:
    """
    CrewAI Agent responsible for defining the mapping strategy and generating
    a detailed task list for converting a C++ work package to Godot, based on
    the proposed Godot structure.
    """
    def __init__(self):
        # LLM configuration will now be passed explicitly to get_agent
        logger.info(f"Initializing MappingDefinerAgent (LLM instance will be provided)")

    def get_agent(self, llm_instance: BaseLLM = None):
        """
        Creates and returns the CrewAI Agent instance.

        Args:
            llm_instance: An optional pre-configured LLM instance to use.
        """
        return Agent(
            role="C++ to Godot Conversion Planner",
            goal=(
                f"Analyze the provided context for a C++ work package, including C++ code snippets, the proposed Godot structure, general instructions, work-package summaries, existing Godot files, potentially existing conversion tasks and existing Godot file content. Your goal is to generate TWO outputs: "
                f"1. A **Mapping Strategy (Markdown)** outlining the high-level conversion approach to {config.TARGET_LANGUAGE}, referencing the proposed Godot structure and addressing any existing mapping/files provided. "
                f"2. A **Structured Task List (JSON)** conforming strictly to the `MappingOutput` Pydantic model (with `package_id`, `task_groups`, and detailed `tasks` including `task_title`, `task_description`, `input_source_files`, `output_godot_file`). The tasks must be granular and map C++ functionality to the specific Godot files defined in the proposed structure."
            ),
            backstory=(
                "You are a meticulous software architect specializing in migrating C++ projects to Godot Engine 4.x. You excel at understanding complex C++ codebases and designing corresponding Godot structures based on SOLID principles. Your strength lies in breaking down the conversion process into logical, feature-based task groups and highly specific, actionable tasks. You carefully consider existing project context, including previously generated structures or mappings, to ensure consistency and refinement. You always output your plan in the precise format required: a Markdown strategy followed by a structured JSON task list."
            ),
            llm=llm_instance,
            verbose=True,
            allow_delegation=False, # This agent performs the core mapping logic
            memory=False, # Mapping for each package should be independent
            # tools=[] # This agent primarily analyzes context and generates structured output
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = MappingDefinerAgent()
#     mapping_agent = agent_creator.get_agent()
#     print("MappingDefinerAgent created:")
#     print(f"Role: {mapping_agent.role}")
#     print(f"Goal: {mapping_agent.goal}")
