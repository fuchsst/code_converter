# src/agents/json_output_formatter.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
# Import the Pydantic model for reference in goal/backstory
from src.tasks.step4.define_mapping import MappingOutput

logger = get_logger(__name__)


def get_json_output_fomratter_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for JSON formatting.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    # Dynamically create an example for the goal description if possible,
    # otherwise use a static placeholder.
    try:
        # Create a placeholder example conforming to MappingOutput structure
        example_structure = MappingOutput(
            package_id="placeholder_pkg",
            mapping_strategy="Placeholder strategy.",
            task_groups=[] # Empty list for brevity in example
        ).model_dump_json(indent=2) # Get JSON string representation
    except Exception:
        example_structure = "{ \"package_id\": \"...\", \"mapping_strategy\": \"...\", \"task_groups\": [...] }"


    return Agent(
        role="JSON Formatting Specialist",
        goal=(
            "Receive the final conversion strategy (string) and the decomposed task groups (likely as a structured list or object) as input context. "
            "Combine these elements into a **single JSON object** that strictly conforms to the `MappingOutput` Pydantic model structure. "
            "Ensure the output includes `package_id` (from context), `mapping_strategy`, and `task_groups` with all nested tasks correctly formatted. "
            "**CRITICAL:** Your final output MUST be ONLY the raw JSON object string. No introductory text, explanations, or markdown fences."
            f"\n\nExample of the required JSON structure:\n```json\n{example_structure}\n```"
        ),
        backstory=(
            "You are a data formatting expert specializing in JSON structures defined by Pydantic models. "
            "You take structured or semi-structured planning data and meticulously assemble it into a valid JSON object according to a precise schema. "
            "You pay close attention to detail, ensuring all required fields are present and correctly formatted. Your output is always clean, raw JSON."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        # output_json=True, # This might be set on the Task instead
        # output_pydantic=MappingOutput, # This might be set on the Task instead
        tools=[] # This agent formats data, doesn't use external tools.
    )
