# src/agents/syntax_validator.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
# Import BaseTool if defining tools here, or Tool type hint if passing tools
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)

def get_syntax_validation_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for syntax validation.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing the instantiated validation tool(s) (e.g., GodotSyntaxValidatorTool).
    """
    return Agent(
        role=f"{config.TARGET_LANGUAGE} Syntax Validation Specialist",
        goal=(
            f"Validate the syntax of the provided {config.TARGET_LANGUAGE} code snippet using the "
            f"'{tools[0].name if tools else 'Godot Syntax Validator'}' tool. " # Dynamically reference tool name if possible
            f"Receive the code snippet as input context. Execute the validation tool with the code. "
            f"Report the exact output from the tool (either success message or error details)."
        ),
        backstory=(
            f"You are a meticulous quality assurance agent specializing in {config.TARGET_LANGUAGE} code for Godot Engine 4.x. "
            f"Your sole focus is on verifying the syntactic correctness of code snippets provided to you. "
            f"You use a specific tool designed to interface with the Godot engine's validator. "
            f"You don't write or modify code; you only run the validation tool and report its findings accurately."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools # Assign the validation tool(s)
    )
