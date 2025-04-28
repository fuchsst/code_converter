# src/agents/step3/json_output_formatter.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
from src.tasks.step3.define_structure import GodotNode, GodotResource, GodotScene, GodotScript, GodotStructureOutput, MigrationScript

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
    # Create a more representative example conforming to GodotStructureOutput structure
    # IMPORTANT: This example MUST strictly follow the Pydantic models in define_structure.py
    example_structure = GodotStructureOutput(
        scenes=[
            GodotScene(
                path="res://scenes/player/Player.tscn",
                nodes=[
                    GodotNode(
                        name="Player",
                        type="CharacterBody2D",
                        node_path="/", # Root node
                        script_path="res://scripts/player/Player.gd"
                    ),
                    GodotNode(
                        name="Sprite",
                        type="Sprite2D",
                        node_path="/Player/", # Child of Player
                        script_path=None # No script attached directly
                    ),
                    GodotNode(
                        name="CollisionShape",
                        type="CollisionShape2D",
                        node_path="/Player/", # Child of Player
                        script_path=None
                    )
                ]
            )
        ],
        scripts=[
            GodotScript(
                path="res://scripts/player/Player.gd",
                purpose="Player movement logic, state management, and interaction handling."
            ),
            GodotScript(
                path="res://scripts/utils/InputManager.gd",
                purpose="Handles player input actions and mappings. Should be autoloaded."
            )
        ],
        resources=[
            GodotResource(
                path="res://resources/player/player_stats.tres",
                purpose="Stores base player stats like health, speed.",
                script="res://scripts/player/PlayerStatsResource.gd" # Assuming a script defines this resource type
            ),
                GodotResource(
                path="res://assets/player_sprite_sheet.tres",
                purpose="Texture resource for the player's animated sprite.",
                script=None # Built-in resource type
            )
        ],
        migration_scripts=[
            MigrationScript(
                script_type="Python", # Example: Python script for asset conversion
                purpose="Converts legacy player textures from PNG to WebP format.",
                path="migration_scripts/convert_player_textures.py",
                related_resource=GodotResource( # Define the resource it affects/creates
                        path="res://assets/player_sprite_sheet.webp", # Example output path
                        purpose="WebP version of player sprite sheet.",
                        script=None
                )
            )
        ],
        notes="Initial structure definition for the core player package. Input handling separated. Added player stats resource."
    ).model_dump_json(indent=2)

    return Agent(
        role="JSON Formatting Specialist (Godot Structure)",
        goal=(
            "Receive the designed Godot structure components (lists of scenes, scripts, resources, migration scripts, and notes) as input context. "
            "Assemble these components into a **single JSON object** that strictly conforms to the `GodotStructureOutput` Pydantic model structure. "
            "Ensure all required fields (`scenes`, `scripts`, `resources`, `migration_scripts`) are present, even if empty lists, and that `notes` is included if provided. "
            "**CRITICAL:** Your final output MUST be ONLY the raw JSON object string, starting with `{` and ending with `}`. "
            "ABSOLUTELY NO introductory text, concluding remarks, explanations, apologies, or markdown formatting (like ```json or ```) should be included in your final response. "
            "Your entire response must be the JSON object itself."
            f"\n\nExample of the required JSON structure:\n```json\n{example_structure}\n```"
        ),
        backstory=(
            "You are a data formatting expert specializing in JSON structures defined by Pydantic models, specifically for Godot project structures. "
            "You take structured design components and meticulously assemble them into a valid JSON object according to the precise `GodotStructureOutput` schema. "
            "You pay close attention to detail, ensuring all required fields are present and correctly formatted. "
            "Your output is ALWAYS clean, raw JSON, without any surrounding text or formatting. You never use markdown fences."
        ),
        llm=llm_instance,
        verbose=False,
        allow_delegation=False,
        tools=[] # This agent formats data.
    )
