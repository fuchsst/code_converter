# src/agents/task_decomposer.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)


def get_task_decomposer_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for task decomposition.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role="Conversion Task Decomposition Specialist",
        goal=(
            "Break down the high-level C++ to Godot conversion strategy into a detailed list of granular, actionable tasks. "
            "Receive the C++ code analysis, Godot structure analysis, and the conversion strategy as input context. "
            "For each logical feature or component identified in the strategy and structure: "
            "1. Define a 'Task Group' with a title, description, and relevant Godot features. "
            "2. Within each group, create specific 'MappingTask' items. Each task must have: "
            "   - `task_title`: Concise title. "
            "   - `task_description`: Detailed explanation linking C++ source elements to Godot targets and implementation approach. "
            "   - `input_source_files`: List of relevant C++ source files. "
            "   - `output_godot_file`: The exact target Godot file path (script, scene, resource) from the proposed structure. "
            "   - `target_element`: (Optional) Specific function/node within the target file. "
            "Ensure tasks are small enough for a code generation agent (Step 5) to handle effectively. "
            "The final output should be a list of these structured Task Groups, ready for JSON formatting."
        ),
        backstory=(
            "You are a technical project manager and software designer with a knack for breaking down complex problems into manageable steps. "
            "You understand both C++ and Godot development workflows. Given a high-level strategy and technical analyses, you can create a detailed, step-by-step implementation plan suitable for development teams (or AI agents). "
            "You ensure that each task is well-defined, references the correct source and target files, and aligns with the overall strategy and proposed architecture."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False, # Focuses on the decomposition task
        tools=[] # This agent structures information, doesn't use external tools.
    )
