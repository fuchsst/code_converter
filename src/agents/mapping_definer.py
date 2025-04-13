# src/agents/mapping_definer.py
from crewai import Agent
from logger_setup import get_logger
import config
# Assuming api_utils handles LLM configuration and calls via CrewAI's mechanism

logger = get_logger(__name__)

# TODO: Ensure the LLM instance used by CrewAI is correctly configured globally
#       or passed explicitly during Agent initialization if needed.
#       Referencing config.MAPPER_MODEL. This step likely requires a powerful model.

class MappingDefinerAgent:
    """
    CrewAI Agent responsible for defining the mapping strategy and generating
    a detailed task list for converting a C++ work package to Godot, based on
    the proposed Godot structure.
    """
    def __init__(self):
        # LLM configuration managed by CrewAI/global setup
        logger.info(f"Initializing MappingDefinerAgent (LLM configuration managed by CrewAI/global setup using model like: {config.MAPPER_MODEL})")

    def get_agent(self):
        """Creates and returns the CrewAI Agent instance."""
        return Agent(
            role="C++ to Godot Conversion Strategist",
            goal=(
                "Analyze the provided C++ work package (file list, code snippets), the proposed "
                "Godot project structure (Markdown format), and the target language "
                f"({config.TARGET_LANGUAGE}). Your goal is twofold:\n"
                "1.  **Define a Mapping Strategy:** Create a concise Markdown document outlining the high-level approach for converting the C++ code to the proposed Godot structure. Describe how key C++ classes, functions, and patterns will map to Godot nodes, scenes, scripts, and APIs (e.g., mapping C++ vectors to Godot Vectors, C++ game loops to _process/_physics_process, etc.).\n"
                "2.  **Generate Actionable Task List:** Create a detailed JSON list of specific, granular tasks required to implement the conversion based on the strategy. Each task should be small enough to be handled by a code generation/editing agent and include details like: target Godot file, specific function/method to create/modify, corresponding C++ source file/function for reference, and brief instructions on the required logic or mapping."
            ),
            backstory=(
                "You are a highly experienced software engineer specializing in cross-language and cross-engine code migration, particularly between C++ and game engines like Godot. You have a deep understanding of C++ idioms, Godot Engine 4.x architecture (GDScript/C#), and common game development patterns. You can meticulously analyze code, devise effective porting strategies, and break down complex conversion processes into precise, manageable implementation steps."
            ),
            # llm=... # Let CrewAI handle LLM
            verbose=True,
            allow_delegation=False, # This agent performs the core mapping logic
            # memory=True # Consider if memory is needed for retries on the same package
            # tools=[] # This agent primarily analyzes context and generates structured output
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = MappingDefinerAgent()
#     mapping_agent = agent_creator.get_agent()
#     print("MappingDefinerAgent created:")
#     print(f"Role: {mapping_agent.role}")
#     print(f"Goal: {mapping_agent.goal}")
