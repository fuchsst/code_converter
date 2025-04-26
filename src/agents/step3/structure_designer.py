# src/agents/step3/structure_designer.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

def get_structure_designer_agent(llm_instance: BaseLLM) -> Agent:
    """
    Creates and returns the configured CrewAI Agent instance for designing
    the Godot project structure for a specific package.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Godot Architecture Designer",
        goal=(
            f"Design a logical Godot 4.x project structure (scenes, nodes, scripts, resources, migration scripts) "
            f"for converting the C++ work package described in the context. "
            f"Base the design on the C++ code analysis and global project context analysis provided. "
            f"Adhere strictly to SOLID principles and Godot best practices ({config.TARGET_LANGUAGE}). "
            f"Consider existing structure definitions for refinement if provided. "
            f"Output the designed structure components (lists of scenes, scripts, resources, etc.) and descriptive notes."
        ),
        backstory=(
            "You are an expert Godot Engine architect specializing in migrating C++ projects. "
            "You translate C++ functionality and structure analysis into well-organized, idiomatic Godot projects. "
            "You focus on creating maintainable, scalable structures using scenes, nodes, scripts, and resources effectively, "
            "while respecting existing project conventions identified in the global context analysis."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # Designs structure based on context.
    )
