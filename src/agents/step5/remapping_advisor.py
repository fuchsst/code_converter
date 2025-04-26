# src/agents/remapping_advisor.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)

def get_remapping_advisor_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for remapping advice.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing the instantiated remapping logic tool(s)
                (e.g., RemappingLogicTool).
    """
    tool_names = [t.name for t in tools if hasattr(t, 'name')]
    return Agent(
        role="Code Conversion Failure Analyst",
        goal=(
            f"Analyze the summary of failed tasks provided for a specific work package. "
            f"Use the '{tools[0].name if tools else 'Remapping Logic Analyzer'}' tool, passing it the necessary failure details from the context. "
            f"Based *only* on the tool's output (which indicates if remapping is advised), formulate a final response. "
            f"The response should clearly state whether remapping is recommended for the package and include the reason provided by the tool."
        ),
        backstory=(
            f"You are an analytical agent specializing in diagnosing issues in automated code conversion pipelines. "
            f"You receive summaries of task failures for a completed work package. "
            f"Your specific function is to use a dedicated tool that encapsulates complex logic to determine if the pattern of failures suggests a fundamental problem with the initial mapping definition (from Step 4). "
            f"You don't make the decision yourself; you execute the tool with the provided failure data and report its recommendation accurately. "
            f"This feedback helps the overall process adapt and improve."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools # Assign the remapping logic tool(s)
    )
