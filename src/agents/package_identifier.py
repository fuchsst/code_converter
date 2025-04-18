# src/agents/package_identifier.py
from crewai import Agent
from src.logger_setup import get_logger
import src.config as config
# Assuming api_utils handles LLM configuration and calls
# from core.api_utils import get_llm_client # Or however the LLM instance is managed

logger = get_logger(__name__)

# LLM configuration is handled by the Crew during execution based on the instance
# passed to the Crew object (e.g., in the StepExecutor).

class PackageIdentifierAgent:
    """
    CrewAI Agent responsible for analyzing the include graph and proposing
    logical work packages for the C++ to Godot conversion.
    """
    def __init__(self):
        # LLM configuration is managed by the Crew during execution
        logger.info(f"Initializing PackageIdentifierAgent (LLM configuration managed by Crew using model like: {config.ANALYZER_MODEL})")

    def get_agent(self):
        """Creates and returns the CrewAI Agent instance."""
        return Agent(
            role="C++ Codebase Analyst",
            goal=(
                "Analyze the provided C++ project's include graph (JSON) to identify "
                "logical, reasonably self-contained work packages (groups of related files) "
                "suitable for incremental conversion to Godot. Prioritize grouping files "
                "based on features, modules (suggested by directory structure), or minimizing "
                "inter-package includes."
            ),
            backstory=(
                "You are an expert software architect specializing in analyzing large C++ "
                "codebases. Your strength lies in understanding code structure and dependencies, "
                "even from approximate include graphs. You can identify cohesive modules and "
                "propose logical partitions to break down complex systems into manageable units "
                "for phased migration or refactoring."
            ),
            # llm=... # LLM is set by the Crew instance during kickoff
            verbose=True,
            allow_delegation=False,
            # memory=True # Consider if memory is needed across iterations within this agent's task
            # tools=[] # This agent likely doesn't need tools, it analyzes provided data
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = PackageIdentifierAgent()
#     package_agent = agent_creator.get_agent()
#     print("PackageIdentifierAgent created:")
#     print(f"Role: {package_agent.role}")
#     print(f"Goal: {package_agent.goal}")
