# src/tasks/define_mapping.py
import json
from crewai import Task, Agent
from src.logger_setup import get_logger
import src.config as config
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

logger = get_logger(__name__)

# --- Pydantic Models for Structured JSON Output ---
class MappingTask(BaseModel):
    """Defines a single, granular conversion task."""
    task_title: str = Field(..., description="A concise title for the task (e.g., 'Implement Player Movement Input').")
    task_description: str = Field(..., description="A short description explaining why this task is needed, how it relates to the C++ source, and the general implementation approach in Godot.")
    input_source_files: List[str] = Field(..., description="List of relevant C++ source/header file paths for reference.")
    # source_cpp_elements: List[str] = Field(..., description="Specific C++ functions, classes, or members to reference.") # Optional refinement
    output_godot_file: str = Field(..., description="The relative path to the target Godot script/scene/resource to be created/modified (e.g., 'res://scripts/player/player_movement.gd'). Must align with Step 3 structure.")
    target_element: Optional[str] = Field(description="The specific function, method, or node within the Godot file (e.g., '_physics_process', 'setup_animations').") # Optional refinement

class TaskGroup(BaseModel):
    """Groups related tasks under a specific feature or component."""
    group_title: str = Field(..., description="Title for the task group (e.g., 'Player Character Controller', 'Inventory System UI').")
    feature_description: str = Field(..., description="A longer description of the overall feature or component this group implements.")
    godot_features: str = Field(..., description="Description of the specific Godot nodes, APIs, or patterns that will be used to implement this feature (e.g., 'CharacterBody3D for movement, AnimationPlayer for animations, Custom Resource for stats').")
    tasks: List[MappingTask] = Field(..., description="A list of granular tasks belonging to this group.")

class MappingOutput(BaseModel):
    """Defines the overall expected JSON output structure for the mapping definition."""
    package_id: str = Field(..., description="The ID of the work package this mapping belongs to.")
    mapping_strategy: Optional[str] = Field(description="Optional high-level summary of the mapping approach for this package.")
    task_groups: List[TaskGroup] = Field(..., description="A list of task groups detailing the conversion plan.")

class DefineMappingTask:
    """
    CrewAI Task definition for defining the C++ to Godot mapping strategy
    and generating a structured, actionable task list grouped by features,
    outputting a single JSON object.
    """

    def create_task(self, agent: Agent, context: str, output_json: Optional[Any] = MappingOutput) -> Task: # Added output_json parameter
        """
        Creates the CrewAI Task instance for defining the mapping.

        Args:
            agent (Agent): The MappingDefinerAgent instance responsible for this task.
            context (str): The context string assembled by ContextManager, containing:
                           - C++ work package info (ID, description, files).
                           - Relevant C++ source code snippets.
                           - The proposed Godot structure.
                           - Overview over of all work packages.
                           - List of existing Godot output files in the target project.
                           - Potentially existing conversion tasks for refinement.
                           - Potentially content of existing Godot files.
                           - General instructions.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating DefineMappingTask for agent: {agent.role}")

        # Generate an example based on the Pydantic model for the expected_output
        example_output_dict = MappingOutput(
            package_id="example_pkg_id",
            mapping_strategy=(
                "The C++ player controller logic will be mapped to a Godot CharacterBody3D script.\n"
                "Input handling will use Godot's Input singleton.\n"
                "Player stats will be managed via a custom PlayerStats Resource."
            ),
            task_groups=[
                TaskGroup(
                    group_title="Player Movement",
                    feature_description="Handles the core movement logic for the player character, including input processing and physics interaction.",
                    godot_features="Utilizes CharacterBody3D for physics, Input singleton for actions, and potentially AnimationPlayer for movement states.",
                    tasks=[
                        MappingTask(
                            task_title="Implement Basic Movement",
                            task_description="Map C++ movement vectors and input checks to CharacterBody3D velocity and Input.get_vector within _physics_process.",
                            input_source_files=["code/player/controller.cpp", "src/core/vector.h"],
                            output_godot_file="res://scripts/player/player_movement.gd",
                            target_element=None
                        ),
                        MappingTask(
                            task_title="Implement Jump Action",
                            task_description="Translate C++ jump logic (e.g., checking ground state, applying impulse) to Godot using CharacterBody3D.is_on_floor() and adjusting velocity.y on 'jump' input action.",
                            input_source_files=["code/player/controller.cpp"],
                            output_godot_file="res://scripts/player/player_movement.gd",
                            target_element=None
                        )
                    ]
                ),
                TaskGroup(
                    group_title="Player Stats",
                    feature_description="Manages player statistics like health and stamina, potentially using a custom resource.",
                    godot_features="Defines a custom Resource script (e.g., PlayerStats.gd) and creates a .tres file. Scripts will access/modify this resource.",
                    tasks=[
                        MappingTask(
                            task_title="Define PlayerStats Resource Script",
                            task_description="Create the PlayerStats.gd script extending Resource, defining exported variables for health, stamina, etc., based on C++ player data structures.",
                            input_source_files=["code/player/player_data.h"],
                            output_godot_file="res://scripts/resources/player_stats.gd",
                            target_element=None
                        ),
                        MappingTask(
                            task_title="Create Default PlayerStats Resource File",
                            task_description="Create the player_stats.tres file using the PlayerStats.gd script, setting default values.",
                            input_source_files=["code/player/player_data.h"],
                            output_godot_file="res://resources/player_stats.tres",
                            target_element=None
                        )
                    ]
                )
            ]
        ).model_dump() # Convert Pydantic model to dict for JSON example
        example_json_output = json.dumps(example_output_dict, indent=2)


        # Incorporate the context directly into the description for the agent
        full_description = (
            "Analyze the provided context, which includes:\n"
            "- The **current C++ work package** definition (ID, description, files).\n"
            "- Relevant **C++ source code snippets** from the package.\n"
            "- The proposed **Godot project structure for this package.\n"
            "- An overview of all working packages for the conversion.\n"
            "- A list of **existing Godot output files** currently in the target project directory.\n"
            "- Potentially, an **existing conversion draft** for this package from a previous run (if provided, refine it).\n"
            "- Potentially, the **content of existing Godot files** referenced in the existing conversion draft (if provided).\n"
            "- General **instructions**.\n"
            f"- The target language: **{config.TARGET_LANGUAGE}**.\n\n"

            "Your primary goal is to create a detailed plan for converting the C++ code within this package to the proposed Godot structure. You must generate **TWO** distinct outputs:\n\n"

            "1.  **Mapping Strategy (Markdown):**\n"
            "    - Write a concise Markdown document outlining the high-level conversion strategy.\n"
            "    - Describe how key C++ classes, functions, data structures, and patterns identified in the code snippets will map to the proposed Godot nodes, scenes, resources, and scripts defined in the **Markdown Structure Hierarchy**.\n"
            "    - Emphasize how the mapping maintains the separation of concerns and leverages Godot's features effectively.\n"
            "    - Reference specific elements from the **Markdown Structure Hierarchy** (e.g., scene paths, script names) where appropriate.\n"
            "    - Mention potential challenges, Godot APIs to use, or areas requiring careful attention.\n"
            "    - If refining an `existing_mapping_json`, explain the refinements made based on it and the `referenced_godot_content`.\n\n"

            "2.  **Structured Task List (JSON):**\n"
            "    - Generate a JSON object conforming precisely to the `MappingOutput` Pydantic model (see example in expected output).\n"
            "    - The JSON must contain `package_id` and a list of `task_groups`.\n"
            "    - Each `task_group` should represent a logical feature or component from the C++ package.\n"
            "    - Each `task_group` must have `group_title`, `feature_description`, `godot_features` (describing Godot nodes/APIs used), and a list of `tasks`.\n"
            "    - Each `task` within a group must have `task_title`, `task_description` (explaining the 'why' and 'how', referencing C++ source), `input_source_files` (relevant C++ files), and `output_godot_file` (the specific target Godot file path from the Step 3 structure).\n"
            "    - Ensure tasks are granular, focused, and provide enough detail for a code generation agent (Step 5) to attempt implementation.\n"
            "    - **Crucially, the `output_godot_file` paths MUST align with the scene/script/resource paths defined in the Markdown Structure Hierarchy provided in the context.**\n\n"

            "--- START OF PROVIDED CONTEXT ---\n"
            f"{context}\n"
            "--- END OF PROVIDED CONTEXT ---\n\n"

            "Your goal is to create a detailed plan for converting the C++ code within this package to the proposed Godot structure. You must generate **ONLY ONE** output: a **single JSON object** conforming precisely to the `MappingOutput` Pydantic model.\n\n"

            "The JSON object MUST contain:\n"
            "1.  `package_id`: The ID of the current work package.\n"
            "2.  `mapping_strategy`: A concise (few sentences) high-level conversion strategy. Describe how key C++ elements map to the proposed Godot structure, leveraging Godot features, mentioning challenges, and referencing the provided structure. If refining, explain refinements.\n"
            "3.  `task_groups`: A list of task groups detailing the conversion plan.\n"
            "    - Each `task_group` represents a logical feature/component and must have `group_title`, `feature_description`, `godot_features` (nodes/APIs used), and a list of `tasks`.\n"
            "    - Each `task` within a group must have `task_title`, `task_description` (why/how, referencing C++), `input_source_files`, and `output_godot_file`.\n"
            "    - Ensure tasks are granular and actionable for a code generation agent.\n"
            "    - **Crucially, `output_godot_file` paths MUST align with the scene/script/resource paths defined in the proposed Godot structure provided in the context.**\n\n"

            "**CRITICAL FORMATTING:** Your entire response MUST be ONLY the raw JSON object string conforming to the `MappingOutput` model. No introductory text, no explanations, no markdown code fences (like ```json), just the JSON itself starting with `{` and ending with `}`."
        )

        return Task(
            description=full_description,
            expected_output=(
                "A **single, valid JSON object string** adhering strictly to the `MappingOutput` model structure (containing `package_id`, `mapping_strategy`, and `task_groups`). "
                "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
                f"\n\nExample of the required raw JSON output format:\n{example_json_output}"
            ),
            agent=agent,
            output_json=output_json # Use Pydantic model for validation by default
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.mapping_definer import MappingDefinerAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = MappingDefinerAgent()
#     mapping_agent = agent_creator.get_agent()
#
#     # Dummy context for testing
#     test_context = """
#     **Work Package Definition (JSON):**
#     ```json
#     {
#       "description": "Handles player movement and input.",
#       "files": ["src/player/player.cpp", "src/player/player.h", "src/input/input_handler.h"]
#     }
#     ```
#
#     **Proposed Godot Structure (JSON):**
#     ```json
#     {
#       "scenes": [{"path": "res://src/player/player.tscn", "nodes": [{"name": "Player", "type": "CharacterBody3D", "script_path": "res://src/player/player_movement.gd"}]}],
#       "scripts": [{"path": "res://src/player/player_movement.gd", "purpose": "Handles movement logic."}],
#       "resources": [],
#       "migration_scripts": [],
#       "notes": "Basic player structure."
#     }
#     ```
#
#     **File:** `src/player/player.h`
#     ```cpp
#     class CppPlayer {
#         Vector3 position;
#         Vector3 velocity;
#         void updateMovement(float delta);
#         void handleJump();
#     };
#     ```
#     """
#
#     task_creator = DefineMappingTask()
#     define_task = task_creator.create_task(mapping_agent, test_context)
#     print("DefineMappingTask created:")
#     print(f"Description: {define_task.description}")
#     print(f"Expected Output: {define_task.expected_output}")
