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
    reader_tool_name = next((t.name for t in tools if "Reader" in t.name), "File Reader")
    return Agent(
        role=f"{config.TARGET_LANGUAGE} Full File Code Refinement Specialist",
        goal=(
            f"Your primary task is to refine the {config.TARGET_LANGUAGE} code within a specific Godot file based on provided project validation errors. "
            f"1. **Understand Context**: You will receive the target file path (e.g., `{'{target_godot_file}'}`), the validation error messages, and potentially the original task item details for broader context. "
            f"2. **Read Current File Content**: You **MUST use the '{reader_tool_name}' tool** to read the current (erroneous) content of the `target_godot_file`. "
            f"3. **Analyze and Correct**: Analyze the validation errors in conjunction with the file content you read. Implement the necessary corrections to fix these errors. "
            f"4. **Generate Full Corrected Content**: Produce the **complete, corrected {config.TARGET_LANGUAGE} code string for the entire file**. Preserve the original logic and structure as much as possible, only making changes to fix the reported errors. "
            f"If, after analysis, you determine that the errors cannot be fixed or are too ambiguous, you may return the original file content (obtained via the '{reader_tool_name}' tool) along with a clear error message explaining the situation. "
            f"Your final output **MUST BE ONLY the raw, complete, corrected {config.TARGET_LANGUAGE} code string for the entire file**, or the original content plus an error message if unfixable. Do not use markdown formatting like ```."
        ),
        backstory=(
            f"You are an expert debugger and code refiner for {config.TARGET_LANGUAGE} in Godot Engine 4.x. "
            f"You specialize in fixing errors identified by project-level validation. You understand that errors can be subtle and require careful analysis of the code in its current state. "
            f"You always use the '{reader_tool_name}' tool to get the most up-to-date version of the file before attempting any refinement. "
            f"Your goal is to produce a fully corrected version of the file content that will pass subsequent validation, or to clearly state if the issues are beyond simple refinement based on the provided errors."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=tools
    )
