# src/agents/code_refiner.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)


def get_code_refinement_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for code refinement based on project validation errors.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing tools the agent can use (e.g., FileReaderTool).
    """
    reader_tool_name = next((t.name for t in tools if t.name == "File Reader"), "File Reader")
    return Agent(
        role=f"{config.TARGET_LANGUAGE} Code Refinement Specialist",
        goal=(
            f"Refine the {config.TARGET_LANGUAGE} code within a specific file based on Godot project validation errors reported for that file. "
            f"Receive the absolute path to the file (`target_file_path`) and the relevant validation error message(s) as input context. "
            f"Use the '{reader_tool_name}' tool to read the current content of the `target_file_path`. "
            f"Analyze the errors and the file content, then generate a corrected version of the *entire file content*. "
            f"Focus solely on fixing the reported errors while preserving the original logic as much as possible. "
            f"If correction is not possible or unclear from the errors, return the original file content along with an error message explaining why. "
            f"The output should be ONLY the refined code string (full file content) or the original content plus an error message."
        ),
        backstory=(
            f"You are a debugging expert for {config.TARGET_LANGUAGE} in Godot Engine 4.x, specializing in fixing errors identified during project-level validation. "
            f"You understand that errors can stem from syntax issues, incorrect type hints, or problems integrating with other project parts (like autoloads or custom classes). "
            f"You use the provided error messages and the '{reader_tool_name}' tool to fetch the current state of the problematic file. "
            f"You carefully examine the code in context and apply the necessary fixes to resolve the reported issues. "
            f"Your goal is to produce corrected file content that should pass the next project validation check."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools
    )
