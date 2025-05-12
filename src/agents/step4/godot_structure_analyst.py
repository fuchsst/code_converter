# src/agents/godot_structure_analyst.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

def get_godot_structure_analyst_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for Godot structure analysis.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Godot Project Structure Analyst",
        goal=(
            "Analyze the provided proposed Godot project structure definition (including scenes, nodes, scripts, resources, and notes) for a specific work package. "
            "Understand the intended role and relationship of each proposed file and node. "
            "Summarize the key architectural choices and the purpose of the main components within this proposed structure."
        ),
        backstory=(
            "You are a Godot Engine expert with a strong understanding of engine architecture, best practices, and scene/script organization. "
            "You can interpret structure definitions and visualize how the different components (scenes, nodes, scripts, resources) are intended to work together. "
            "Your analysis helps bridge the gap between the proposed structure and the C++ code being converted."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=[] # This agent analyzes context, doesn't use external tools.
    )
