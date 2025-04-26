# src/agents/package_refiner.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

def get_package_refinement_agent(llm_instance: BaseLLM = None) -> Agent:
    """
    Creates and returns the CrewAI Agent instance for refinement.

    Args:
        llm_instance: An optional pre-configured LLM instance to use.
    """
    return Agent(
        role="Holistic Software Architect",
        goal=(
            "Analyze a collection of software package summaries (including initial descriptions and file roles). "
            "Refine the 'package_description' for each package to be more accurate and context-aware, "
            "considering its role relative to other packages in the system. "
            "Output the results strictly as a JSON dictionary mapping package names to their refined descriptions."
        ),
        backstory=(
            "You are a principal software architect with extensive experience in analyzing large codebases. "
            "You possess a unique ability to understand the high-level interactions between different software modules. "
            "Your task is to review initially generated descriptions for a set of code packages and refine them, "
            "ensuring each description accurately reflects the package's purpose within the broader system architecture. "
            "Clarity, conciseness, and contextual accuracy are paramount."
        ),
        verbose=True,
        memory=False, # Refinement is based on the provided context dump
        allow_delegation=False, # This agent performs the refinement directly
        llm=llm_instance
    )
