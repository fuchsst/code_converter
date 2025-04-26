# src/agents/code_refiner.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)


def get_code_refinement_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for code refinement.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role=f"{config.TARGET_LANGUAGE} Code Refinement Specialist",
        goal=(
            f"Refine the provided {config.TARGET_LANGUAGE} code snippet based on the syntax validation errors reported. "
            f"Receive the original generated code and the validation error message(s) as input context. "
            f"Analyze the errors and the code, then generate a corrected version of the code snippet. "
            f"Focus solely on fixing the reported syntax errors while preserving the original logic. "
            f"If correction is not possible or unclear, indicate failure. "
            f"The output should be ONLY the refined code string or an error message if refinement failed."
        ),
        backstory=(
            f"You are a debugging expert for {config.TARGET_LANGUAGE} in Godot Engine 4.x. "
            f"You specialize in analyzing syntax errors reported by the Godot validator and correcting the code accordingly. "
            f"You carefully examine the erroneous code and the error messages to pinpoint the issue and apply the necessary fixes. "
            f"Your goal is to produce syntactically valid code that maintains the intended functionality of the original snippet. "
            f"You do not perform validation yourself; you rely on the error feedback provided."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # This agent does not use external tools directly
    )
