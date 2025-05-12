# src/agents/cpp_code_analyst.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

def get_cpp_code_analyst_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for C++ code analysis.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Expert C++ Code Analyst",
        goal=(
            "Analyze the provided C++ source code snippets and file contents for a specific work package. "
            "Identify key classes, functions, data structures, logic flow, and the overall purpose of the code within that package. "
            "Summarize your findings clearly, focusing on aspects relevant for conversion to Godot Engine."
        ),
        backstory=(
            "You are a seasoned C++ software engineer with deep experience in analyzing large codebases, particularly in the gaming domain. "
            "You can quickly understand complex C++ code, identify core functionalities, and extract the essential information needed for migration planning. "
            "Your analysis forms the foundation for mapping C++ concepts to a new engine like Godot."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=[] # This agent analyzes context, doesn't use external tools.
    )
