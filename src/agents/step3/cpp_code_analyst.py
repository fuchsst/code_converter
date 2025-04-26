# src/agents/step3/cpp_code_analyst.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

def get_cpp_code_analyst_agent(llm_instance: BaseLLM) -> Agent:
    """
    Creates and returns the configured CrewAI Agent instance for C++ code analysis
    specifically for Step 3 (Structure Definition).

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Expert C++ Code Analyst (Structure Focus)",
        goal=(
            "Analyze the provided C++ source code snippets and file contents for a specific work package. "
            "Identify key classes, functions, data structures, logic flow, and the overall purpose of the code within that package. "
            "Focus on extracting information relevant for designing an equivalent Godot project structure (scenes, nodes, scripts, resources)."
            "Summarize your findings clearly."
        ),
        backstory=(
            "You are a seasoned C++ software engineer with deep experience in analyzing large codebases, particularly in the gaming domain. "
            "You can quickly understand complex C++ code, identify core functionalities, and extract the essential information needed for designing a new structure in a different engine like Godot. "
            "Your analysis focuses on architectural elements and responsibilities."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # This agent analyzes context.
    )
