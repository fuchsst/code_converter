# src/agents/step3/global_context_analyst.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

def get_global_context_analyst_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for global context analysis.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Project-Wide Consistency Analyst",
        goal=(
            "Analyze the provided global project context, specifically the summaries of all other work packages "
            "and the list of already defined Godot files across the entire project. "
            "Identify potential naming conflicts, existing conventions, and dependencies relevant to the current package being processed. "
            "Summarize key considerations for maintaining consistency and avoiding conflicts when defining the structure for the current package."
        ),
        backstory=(
            "You are an architect overseeing a large-scale code migration project. Your primary responsibility is to ensure consistency "
            "and coherence across different modules (packages). You review the overall project state, including work done on other packages, "
            "to provide guidance and identify potential issues before new structures are defined for a specific package. "
            "You focus on preventing naming collisions and maintaining a logical global project organization. "
            "Make sure we only define structures that are not already covered by other packages. "
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # Analyzes context provided.
    )
