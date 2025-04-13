# src/agents/package_identifier.py
from crewai import Agent
from logger_setup import get_logger
import config
# Assuming api_utils handles LLM configuration and calls
# from core.api_utils import get_llm_client # Or however the LLM instance is managed

logger = get_logger(__name__)

# TODO: Configure the actual LLM instance (e.g., Gemini)
# llm = get_llm_client(model_name=config.GEMINI_MODEL_NAME) # Example

class PackageIdentifierAgent:
    """
    CrewAI Agent responsible for analyzing the include graph and proposing
    logical work packages for the C++ to Godot conversion.
    """
    def __init__(self):
        # TODO: Replace placeholder LLM with actual configured instance
        self.llm_placeholder = "PlaceholderLLM_PackageIdentifier"
        logger.info(f"Initializing PackageIdentifierAgent with LLM: {self.llm_placeholder}")

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
            # llm=self.llm_placeholder, # TODO: Use actual LLM instance
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
