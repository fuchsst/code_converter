# src/agents/conversion_strategist.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

def get_conversation_strategist_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for conversion strategy definition.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="C++ to Godot Conversion Strategist",
        goal=(
            f"Develop a concise, high-level strategy for converting the C++ work package to the proposed Godot structure. "
            f"Receive the C++ code analysis and the Godot structure analysis as input context. "
            f"Synthesize these analyses to outline how key C++ concepts (classes, systems, logic) will map to Godot patterns (nodes, scenes, scripts, resources) within the proposed structure. "
            f"Mention key Godot APIs or features to leverage (e.g., CharacterBody3D, Signals, Resources). "
            f"Consider any existing Godot files or refinement feedback provided in the context. "
            f"The output should be a clear, well-reasoned mapping strategy description (typically a few sentences to a paragraph)."
        ),
        backstory=(
            "You are a senior game architect with expertise in both C++ engine development and Godot Engine. "
            "You excel at bridging the gap between different engine paradigms. "
            "Given an analysis of C++ code and a proposed Godot structure, you can devise an effective and idiomatic conversion strategy. "
            "You focus on the 'big picture' mapping before tasks are decomposed."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False, # Focuses on synthesizing the strategy
        tools=[] # This agent synthesizes information, doesn't use external tools.
    )
