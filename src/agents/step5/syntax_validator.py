# src/agents/syntax_validator.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
# Import BaseTool if defining tools here, or Tool type hint if passing tools
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)

def get_project_validation_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for project validation
    after a file modification.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing the instantiated validation tool(s) (e.g., GodotProjectValidatorTool).
    """
    tool_name = tools[0].name if tools else 'Godot Project Validator'
    return Agent(
        role="Godot Project Post-Modification Validator",
        goal=(
            f"Validate the Godot project's integrity after a file modification, focusing on errors related to the changed file. "
            f"Receive the absolute path to the Godot project directory (`godot_project_path`) and the `res://` path of the modified file (`target_file_path`) as input context. "
            f"Execute the '{tool_name}' tool, providing these two paths. "
            f"Report the exact output from the tool, which will indicate overall project validation success or list only the errors relevant to the `target_file_path`."
        ),
        backstory=(
            f"You are a meticulous quality assurance agent specializing in Godot Engine 4.x projects. "
            f"Your role is to verify that recent code changes haven't introduced parsing or integration errors within the project context. "
            f"You use a specific tool ('{tool_name}') that runs the Godot editor headlessly to check the entire project, but you focus your report on issues directly related to the file that was just modified. "
            f"You accurately report the tool's findings, indicating success or detailing the relevant errors."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools # Assign the validation tool(s)
    )
