# src/tasks/define_structure.py
import json
from crewai import Task, Agent
import src.config as config
from src.logger_setup import get_logger
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

logger = get_logger(__name__)

# --- Pydantic Models for Structured Output ---

class GodotScript(BaseModel):
    """Defines a proposed Godot script file."""
    path: str = Field(..., description="The proposed relative path for the script within the Godot project (e.g., 'res://scripts/my_package/player_controller.gd').")
    purpose: str = Field(..., description="A brief description of the script's main responsibility.")
    attached_to_scene: Optional[str] = Field(description="The path of the scene file this script is primarily associated with, if applicable.") # Removed default=None
    attached_to_node: Optional[str] = Field(description="The name/path of the node within the scene this script is attached to, if applicable.") # Removed default=None

class GodotResource(BaseModel):
    """Defines a proposed Godot script file."""
    path: str = Field(..., description="The proposed relative path for the resource within the Godot project (e.g., 'res://resources/my_package/player_settings.tres').")
    purpose: str = Field(..., description="A brief description of the resource's content.")
    script: Optional[str] = Field(description="The path of the gdscript file that defines the structure of this resource.")

class GodotNode(BaseModel):
    """Defines a node within a proposed Godot scene."""
    name: str = Field(..., description="The name of the node.")
    type: str = Field(..., description="The Godot type of the node (e.g., 'Node2D', 'CharacterBody3D', 'Sprite2D').")
    node_path: str = Field(..., description="Path in the scene (e.g. '/' if it is the root node, '/Ship/Cockpit/' if it is a child of the nested Cockpit node).")
    script_path: Optional[str] = Field(description="The path to the script attached to this node, if any (should match a path from the 'scripts' list).")

class GodotScene(BaseModel):
    """Defines a proposed Godot scene file."""
    path: str = Field(..., description="The proposed relative path for the scene file within the Godot project (e.g., 'res://scenes/my_package/main_level.tscn').")
    nodes: List[GodotNode] = Field(..., description="The nodes in the scene.")

class MigrationScript(BaseModel):
    """Defines a proposed Godot scene file."""
    script_type: str = Field(..., description="Either 'Godot' or 'Python'.")
    purpose: str = Field(..., description="A brief description of the script's main responsibility.")
    path: str = Field(..., description="The proposed relative path for the Godot or Python script file (e.g., 'res://migration_scripts/asset_type/convert_dds.py').")
    related_resource: GodotResource = Field(..., description="The resource generated (converted to) by this script.")

class GodotStructureOutput(BaseModel):
    """Defines the overall expected JSON output structure for the Godot structure proposal."""
    scenes: List[GodotScene] = Field(..., description="A list of proposed scene definitions.")
    scripts: List[GodotScript] = Field(..., description="A list of proposed script definitions.")
    resources: List[GodotResource] = Field(..., description="A list of proposed resources.")
    migration_scripts: List[MigrationScript] = Field(..., description="A list of migration scripts.")
    notes: Optional[str] = Field(description="Optional overall notes about the proposed structure or mapping considerations.")


class DefineStructureTask:
    """
    CrewAI Task definition for proposing a Godot project structure based on
    a C++ work package.
    """
    def create_task(self, agent: Agent, context: str) -> Task:
        """
        Creates the CrewAI Task instance for defining the Godot structure.

        Args:
            agent (Agent): The StructureDefinerAgent instance responsible for this task.
            context (str): The context string containing work package info and C++ code snippets,
                           assembled by the ContextManager.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating DefineStructureTask for agent: {agent.role}")

        # Incorporate the context directly into the description for the agent
        full_description = (
            "You are an expert Godot Engine architect specializing in C++ to Godot conversion.\n"
            "Analyze the provided context, which includes:\n"
            "  - Information about the **current C++ work package** (ID, description, file list).\n"
            "  - Potentially relevant **C++ code snippets** from the current package's files.\n"
            "  - **Summaries of ALL work packages** (`all_package_summaries`) including their descriptions and file lists, to understand the overall project context.\n"
            "  - Potentially an **existing Godot structure definition** (`existing_package_structure`) for THIS package from a previous run (if available).\n"
            "  - A list of **all existing Godot scene and script files** (`all_existing_godot_files`) defined across ALL packages in previous runs, to avoid naming conflicts and understand the global structure.\n"
            "  - Potentially general **instructions** under an 'Instructions' heading.\n\n"
            "Your goal is to propose a logical Godot 4.x project structure for converting the **current** package. "
            f"The target language for scripts MUST be '{config.TARGET_LANGUAGE}'.\n\n"
            "The proposed structure MUST:\n"
            "- Adhere strictly to SOLID principles (Single Responsibility, Open/Closed, etc.).\n"
            "- Promote good separation of concerns (e.g., UI, logic, data).\n"
            "- Be idiomatic to Godot 4.x best practices (scene composition, node usage).\n"
            "- Follow Godot folder and file structure recommendations (e.g. assets, resources, scenes, scripts).\n"
            "- Be maintainable and testable.\n"
            "- Clearly reflect the functionality of the original C++ package.\n"
            "- Be consistent with the overall project structure suggested by `all_package_summaries` and `all_existing_godot_files`.\n\n"
            "Your proposal needs to define:\n"
            "1.  `resources`: A list of resources (`.tres`) related to this work package.\n"
            "2.  `scenes`: A list of scene definitions (`.tscn`). Each scene has alist of nodes its type, path and name. Ensure scene paths are unique across the project (check `all_existing_godot_files`).\n"
            "3.  `scripts`: A list of script definitions (`.gd`). Each script needs a path, purpose, and potentially the scene/node it's attached to. Ensure scripts have focused responsibilities and unique paths (check `all_existing_godot_files`).\n"
            "4.  `migration_scripts`: Godot or Python scripts to convert media files and resources to Godot formats.\n"
            "5.  `notes`: Overall notes about the structure, mapping to the original code and considerations on Godot native feature as replacment.\n\n"
            "Consider the following when designing the structure:\n"
            "- **Existing Structure:** If `existing_package_structure` is provided, review it. You can refine it, correct errors, or propose a completely new structure if the existing one is unsuitable. Explain your reasoning in the `notes` if you deviate significantly.\n"
            "- **Global Context:** Use `all_package_summaries` to understand how this package fits into the larger application. Use `all_existing_godot_files` to ensure your proposed scene/script paths are unique and consistent with potentially existing conventions.\n"
            "- **C++ Code:** Base your node/scene/script breakdown on the functionality observed in the provided C++ snippets.\n"
            "- **Instructions:** Adhere to any general instructions provided.\n\n"
            "--- START OF PROVIDED CONTEXT ---\n"
            f"{context}\n"
            "--- END OF PROVIDED CONTEXT ---\n\n"
            "**CRITICAL:** Your output MUST be ONLY the raw JSON object string conforming to the GodotStructureOutput model. No introductory text, no explanations, no markdown code fences (like ```json), just the JSON itself starting with `{` and ending with `}`."

        )

        # Generate an example based on the Pydantic model for the expected_output
        example_output_dict = {
            "scenes": [
            {
                "path": "res://scenes/example_package/main_scene.tscn",
                "nodes": [
                {
                    "name": "MainScene",
                    "type": "Node2D",
                    "node_path": "/", # Root node path
                    "script_path": "res://scripts/example_package/main_scene.gd"
                },
                {
                    "name": "Player",
                    "type": "CharacterBody2D",
                    "node_path": "/MainScene/Player", # Example child path relative to root
                    "script_path": "res://scripts/example_package/player_controller.gd"
                },
                {
                    "name": "HUD",
                    "type": "CanvasLayer",
                    "node_path": "/MainScene/HUD", # Example child path relative to root
                    "script_path": "res://scripts/example_package/hud.gd"
                }
                ]
            }
            ],
            "scripts": [
            {"path": "res://scripts/example_package/main_scene.gd", "purpose": "Coordinates overall scene logic and setup.", "attached_to_scene": "res://scenes/example_package/main_scene.tscn", "attached_to_node": "MainScene"},
            {"path": "res://scripts/example_package/player_controller.gd", "purpose": "Handles player movement, input, and state.", "attached_to_scene": "res://scenes/example_package/main_scene.tscn", "attached_to_node": "Player"},
            {"path": "res://scripts/example_package/hud.gd", "purpose": "Manages the Heads-Up Display elements.", "attached_to_scene": "res://scenes/example_package/main_scene.tscn", "attached_to_node": "HUD"},
            {"path": "res://scripts/example_package/player_stats_resource.gd", "purpose": "Defines the structure for the PlayerStats resource.", "attached_to_scene": None, "attached_to_node": None}
            ],
            "resources": [
             {"path": "res://resources/example_package/player_stats.tres", "purpose": "Stores player base statistics.", "script": "res://scripts/example_package/player_stats_resource.gd"}
            ],
            "migration_scripts": [
            {
                "script_type": "Python", # Or "Godot"
                "purpose": "Converts legacy texture format to PNG.",
                "path": "res://migration_scripts/textures/convert_legacy_tex.py",
                # Example related resource (e.g., the .import file generated)
                "related_resource": {"path": "res://assets/textures/converted_texture.png.import", "purpose": "Import settings for converted texture.", "script": None}
            }
            ],
            "notes": "This structure separates player logic from the main scene and HUD. Includes a custom resource for player stats and a sample migration script."
        }
        # Convert dict to JSON string for the example
        example_json_output = json.dumps(example_output_dict, indent=2)


        return Task(
            description=full_description,
            expected_output=(
                "A **single, valid JSON object string** adhering strictly to the `GodotStructureOutput` model structure. "
                "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
                f"\n\nExample of the required raw JSON output format:\n{example_json_output}"
            ),
            agent=agent,
            output_json=GodotStructureOutput # Use output_json to enforce Pydantic model
            # output_file="analysis_results/package_structure_proposal.json" # Optional: Save output directly
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.structure_definer import StructureDefinerAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = StructureDefinerAgent()
#     structure_agent = agent_creator.get_agent()
#
#     # Dummy context for testing
#     test_context = """
#     **Work Package Definition (JSON):**
#     ```json
#     {
#       "description": "Handles sound loading and playback.",
#       "files": ["src/audio/manager.cpp", "src/audio/sound.h"]
#     }
#     ```
#
#     **File:** `src/audio/manager.h`
#     ```cpp
#     class AudioManager {
#     public:
#         static AudioManager& instance();
#         void playSound(const std::string& soundId);
#         void setVolume(float volume);
#     };
#     ```
#     """
#
#     task_creator = DefineStructureTask()
#     define_task = task_creator.create_task(structure_agent, test_context)
#     print("DefineStructureTask created:")
#     print(f"Description: {define_task.description}")
#     print(f"Expected Output: {define_task.expected_output}")
