# src/tasks/define_structure.py
from crewai import Task, Agent
from src.models.structure_models import GodotNode, GodotResource, GodotScene, GodotScript, GodotStructureOutput, MigrationScript
import src.config as config
from src.logger_setup import get_logger
from typing import Optional

logger = get_logger(__name__)


def create_hierarchical_define_structure_task(manager_agent: Agent,
                                              context: str,
                                              package_id: str,
                                              instructions: Optional[str] = None) -> Task:
    """
    Creates the CrewAI Task instance for defining the Godot structure, assigned to the manager.

    Args:
        manager_agent (Agent): The Manager agent overseeing the hierarchical process.
        context (str): The comprehensive context string assembled by ContextManager.
        package_id (str): The ID of the work package being processed.
        instructions (Optional[str]): General instructions to prepend to the task description.

    Returns:
        Task: The CrewAI Task object assigned to the manager.
    """
    logger.info(f"Creating HierarchicalDefineStructureTask for Manager (Package: {package_id})")

    # Example generation remains similar, describing the FINAL output format
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


    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions to Consider:**\n{instructions}\n\n---\n\n"

    full_description += (
        f"**Your Goal as Manager:** Orchestrate the definition of a Godot project structure for C++ work package '{package_id}'. Coordinate specialized agents using the provided context.\n\n"

        "**Provided Context Includes:**\n"
        "- C++ source code for the package.\n"
        "- Summaries of all other packages.\n"
        "- List of already defined Godot files across the project.\n"
        "- Potentially, an existing structure definition for refinement.\n"
        "--- START OF PROVIDED CONTEXT ---\n"
        f"{context}\n"
        "--- END OF PROVIDED CONTEXT ---\n\n"

        "**Your Orchestration Steps (Delegate to appropriate agents):**\n"
        "1.  **Analyze C++ Code:** Delegate to 'CppCodeAnalystAgent'. Provide relevant C++ source from context. Goal: Understand C++ code purpose, structure, key elements.\n"
        "2.  **Analyze Global Context:** Delegate to 'GlobalContextAnalystAgent'. Provide package summaries and existing Godot file list from context. Goal: Identify potential conflicts and ensure consistency.\n"
        "3.  **Design Structure:** Delegate to 'StructureDesignerAgent'. Provide outputs from steps 1 & 2, plus any existing structure definition for refinement. Goal: Design the Godot structure (scenes, nodes, scripts, resources, migration scripts, notes) for *this* package.\n"
        "4.  **Format Output:** Delegate to 'JsonOutputFormatterAgent'. Provide the designed structure components from step 3. Goal: Combine inputs into a single, valid JSON object string conforming precisely to the `GodotStructureOutput` Pydantic model.\n\n"

        "**Final Output:** The final result of this orchestration MUST be the single, raw JSON object string produced by the 'JsonOutputFormatterAgent'."
    )

    return Task(
        description=full_description,
        expected_output=(
            "A **single, valid JSON object string** adhering strictly to the `GodotStructureOutput` model structure. "
            "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
            f"\n\nExample of the required raw JSON output format:\n{example_structure}"
        ),
        agent=manager_agent, # Task assigned to the manager
        output_pydantic=GodotStructureOutput # Ensure final output matches the model
    )
