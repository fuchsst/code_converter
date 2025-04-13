# src/agents/code_processor.py
from crewai import Agent
from logger_setup import get_logger
import config
from core.api_utils import call_gemini_api # Direct call needed for internal looping
from tools.godot_validator_tool import validate_gdscript_syntax # Import the validator tool

# Note: We are NOT creating a separate CodePatcherTool.
# This agent will use write_to_file or replace_in_file via the orchestrator/CrewAI framework
# based on instructions generated for the LLM.

logger = get_logger(__name__)

# TODO: Ensure the LLM instance used by CrewAI is correctly configured globally
#       or passed explicitly during Agent initialization if needed.
#       Referencing config.GENERATOR_EDITOR_MODEL.

class CodeProcessorAgent:
    """
    CrewAI Agent responsible for executing the conversion tasks defined in Step 4.
    It generates/modifies Godot code based on a JSON task list, validates it,
    and potentially handles internal refinement based on validation results.
    Relies on the orchestrator to handle file writing/patching via tools.
    """
    def __init__(self):
        # LLM configuration managed by CrewAI/global setup
        logger.info(f"Initializing CodeProcessorAgent (LLM configuration managed by CrewAI/global setup using model like: {config.GENERATOR_EDITOR_MODEL})")
        # Make the validator tool available to the agent instance if needed for direct calls,
        # although CrewAI typically manages tool execution based on LLM requests.
        self.validator_tool = validate_gdscript_syntax

    def get_agent(self):
        """Creates and returns the CrewAI Agent instance."""
        # Define the tools this agent *can request* CrewAI to use.
        # The actual file writing/patching might be handled by the orchestrator interpreting
        # the agent's output, or the agent could request write_to_file/replace_in_file directly
        # if CrewAI is configured to allow agent access to these fundamental tools.
        # For now, let's assume it can request validation.
        available_tools = [self.validator_tool]

        return Agent(
            role=f"C++ to {config.TARGET_LANGUAGE} Code Translator",
            goal=(
                f"Execute a list of conversion tasks provided in JSON format. For each task: "
                f"1. Analyze the task details (description, target file/element, source C++/elements, mapping notes). "
                f"2. Use the provided C++ code context and mapping notes to generate the required {config.TARGET_LANGUAGE} code snippet or full file content. "
                f"3. **Crucially, decide whether to output the *entire* modified target file content OR just the *specific code block/function* that needs changing.** Base this decision on the task's scope (e.g., creating a new file vs. modifying a function). Clearly indicate your choice. "
                f"4. If outputting a specific block, ensure it's clearly delimited and includes necessary context (like function signature) for replacement. "
                f"5. Optionally, request syntax validation of the generated {config.TARGET_LANGUAGE} code using the available tool. "
                f"6. Report the generated code (full file or block) and validation status. If validation fails, attempt to fix the syntax error based on the feedback and re-validate (up to 2 attempts)."
            ),
            backstory=(
                f"You are a meticulous programmer specialized in translating C++ code into idiomatic {config.TARGET_LANGUAGE} for the Godot Engine 4.x. "
                f"You follow instructions precisely from a task list, referencing provided C++ snippets and mapping guidelines. "
                f"You write clean, functional {config.TARGET_LANGUAGE} code. You understand the importance of syntax validation and can attempt basic corrections based on validator feedback. "
                f"You clearly communicate whether your output is a complete file or a specific code block intended for replacement."
            ),
            # llm=... # Let CrewAI handle LLM
            verbose=True,
            allow_delegation=False, # Focuses on executing the defined tasks
            # memory=True # Might be useful for the internal refinement loop on validation errors
            tools=available_tools # Make the validator tool available
        )

    # Note: The internal looping logic (iterating through JSON tasks, calling LLM,
    # handling validation feedback) as described in concept.md (Option A)
    # would typically be implemented within a custom Tool or directly within the
    # Agent's execution logic if using a framework that allows overriding agent execution flow.
    # In standard CrewAI, each task in the JSON list might need to become a separate CrewAI Task,
    # which contradicts the concept's goal (Option B).
    # For now, the Agent's goal reflects the intended *behavior*, but implementation details
    # depend on how CrewAI tasks are structured by the orchestrator (main.cli.py).
    # The orchestrator might need to loop through the JSON tasks and create CrewAI tasks dynamically,
    # or a custom tool could encapsulate this loop.

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     agent_creator = CodeProcessorAgent()
#     processor_agent = agent_creator.get_agent()
#     print("CodeProcessorAgent created:")
#     print(f"Role: {processor_agent.role}")
#     print(f"Goal: {processor_agent.goal}")
#     print(f"Tools: {[tool.name for tool in processor_agent.tools]}")
