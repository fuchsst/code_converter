# src/agents/structure_definer.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
# Assuming api_utils handles LLM configuration and calls via CrewAI's mechanism

logger = get_logger(__name__)


class StructureDefinerAgent:
    """
    CrewAI Agent responsible for proposing a Godot 4.x project structure
    (nodes, scenes, script organization) for a given C++ work package.
    """
    def get_agent(self, llm_instance: BaseLLM = None) -> Agent:
        """
        Creates and returns the CrewAI Agent instance.

        Args:
            llm_instance: An optional pre-configured LLM instance to use.
        """
        return Agent(
            role="Godot Architecture Designer",
            goal=(
                "Analyze a C++ work package context (including description, files, code snippets, and potentially general instructions) "
                "and propose a logical Godot 4.x project structure (scenes, nodes, scripts) adhering to SOLID principles and Godot best practices. "
                f"The target language for scripts MUST be '{config.TARGET_LANGUAGE}'. "
                "Output the structure strictly as a JSON object conforming to the GodotStructureOutput Pydantic model defined in the task."
            ),
            backstory=(
                "You are an expert Godot Engine architect with extensive experience in designing scalable and maintainable "
                "game projects. You specialize in translating requirements and existing code structures (like C++) "
                "into well-organized Godot projects that follow SOLID principles and best practices for scene composition, "
                "node hierarchy, and script decoupling. Your task is to analyze the provided C++ package information and any instructions, "
                "then propose an optimal Godot structure, outputting it precisely in the specified JSON format."
            ),
            llm=llm_instance,
            verbose=True,
            allow_delegation=False,
            memory=False
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = StructureDefinerAgent()
#     structure_agent = agent_creator.get_agent()
#     print("StructureDefinerAgent created:")
#     print(f"Role: {structure_agent.role}")
#     print(f"Goal: {structure_agent.goal}")
