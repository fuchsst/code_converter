# src/agents/structure_definer.py
from crewai import Agent
from logger_setup import get_logger
import config
# Assuming api_utils handles LLM configuration and calls via CrewAI's mechanism

logger = get_logger(__name__)

# TODO: Ensure the LLM instance used by CrewAI is correctly configured globally
#       or passed explicitly during Agent initialization if needed.
#       Referencing config.MAPPER_MODEL (or a dedicated structure model if defined).

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
                "selected C++ code snippets, propose a logical Godot 4.x project structure. "
                "This includes suggesting scene layouts, node hierarchies (e.g., using Node2D, Control, CharacterBody3D), "
                "script names and their potential responsibilities (in GDScript or C# as per target), "
                "and how the C++ functionality might map to Godot concepts. "
                "Focus on creating a clear, maintainable, and idiomatic Godot structure."
            ),
            backstory=(
                "You are a seasoned game developer with deep expertise in Godot Engine 4.x architecture "
                "and best practices. You excel at translating requirements and existing code structures "
                "(even from different languages/engines like C++) into well-organized Godot projects. "
                "You understand scene composition, node inheritance, signal usage, and how to structure "
                "scripts effectively for clarity and performance."
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
