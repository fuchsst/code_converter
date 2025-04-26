# src/agents/package_describer.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)


def get_package_describer_agent(llm_instance: BaseLLM = None) -> Agent:
    """
    Creates and returns the CrewAI Agent instance.

    Args:
        llm_instance: An optional pre-configured LLM instance to use.
    """
    return Agent(
        role="Expert C++ Software Architect",
        goal=(
            "Analyze the provided C++ code files within a specific package. "
            "Generate a concise overall description for the package and determine the primary role of each file within that package. "
            "Output the results strictly as a JSON object."
        ),
        backstory=(
            "You are a seasoned software architect with deep expertise in C++ and software design principles. "
            "You excel at understanding the purpose and structure of code modules by analyzing source files. "
            "Your task is to examine a pre-defined package of C++ files and provide a structured summary "
            "in JSON format, focusing on the package's overall function and the role each file plays."
        ),
        verbose=True,
        memory=False, # Each package description is independent
        allow_delegation=False, # This agent performs the analysis directly
        llm=llm_instance
    )
