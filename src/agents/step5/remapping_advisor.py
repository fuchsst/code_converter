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
            f"Your task is to analyze a list of failed task item results (provided in your task description) for a work package. "
            f"1. Extract the list of failed task item dictionaries from your task description. "
            f"2. Use your '{tool_names[0] if tool_names else 'Remapping Logic Analyzer'}' tool, passing this list as the 'failed_tasks' argument. "
            f"3. Your final output **MUST BE the exact JSON string returned by this tool.** This JSON string will conform to the `RemappingAdvice` model. "
            f"Do not add any extra text, explanations, or markdown formatting around the JSON string."
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
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=tools # Assign the remapping logic tool(s)
    )
