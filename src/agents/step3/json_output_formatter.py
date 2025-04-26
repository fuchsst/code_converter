# src/agents/step3/json_output_formatter.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
# Import the Pydantic model for reference in goal/backstory
from src.tasks.define_structure import GodotStructureOutput

logger = get_logger(__name__)

def get_json_output_formatter_agent(llm_instance: BaseLLM) -> Agent:
    """
    Creates and returns the configured CrewAI Agent instance for formatting
    the proposed Godot structure into the final JSON output.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    # Dynamically create an example for the goal description if possible,
    # otherwise use a static placeholder.
    try:
        # Create a placeholder example conforming to GodotStructureOutput structure
        example_structure = GodotStructureOutput(
            scenes=[], # Empty lists for brevity
            scripts=[],
            resources=[],
            migration_scripts=[],
            notes="Placeholder notes."
        ).model_dump_json(indent=2) # Get JSON string representation
    except Exception:
        example_structure = "{ \"scenes\": [...], \"scripts\": [...], \"resources\": [...], \"migration_scripts\": [...], \"notes\": \"...\" }"

    return Agent(
        role="JSON Formatting Specialist (Godot Structure)",
        goal=(
            "Receive the designed Godot structure components (lists of scenes, scripts, resources, migration scripts, and notes) as input context. "
            "Assemble these components into a **single JSON object** that strictly conforms to the `GodotStructureOutput` Pydantic model structure. "
            "Ensure all required fields (`scenes`, `scripts`, `resources`, `migration_scripts`) are present, even if empty lists, and that `notes` is included if provided. "
            "**CRITICAL:** Your final output MUST be ONLY the raw JSON object string. No introductory text, explanations, or markdown fences."
            f"\n\nExample of the required JSON structure:\n```json\n{example_structure}\n```"
        ),
        backstory=(
            "You are a data formatting expert specializing in JSON structures defined by Pydantic models, specifically for Godot project structures. "
            "You take structured design components and meticulously assemble them into a valid JSON object according to the precise `GodotStructureOutput` schema. "
            "You pay close attention to detail, ensuring all required fields are present and correctly formatted. Your output is always clean, raw JSON."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # This agent formats data.
    )
