# src/tasks/define_mapping.py
import json
from crewai import Task, Agent
from src.logger_setup import get_logger
import src.config as config
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Type

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


class HierarchicalDefineMappingTask:
    """
    CrewAI Task definition for the MANAGER agent to orchestrate the definition
    of the C++ to Godot mapping strategy and task list using a hierarchical crew.
    """

    def create_task(self, manager_agent: Agent, context: str, package_id: str) -> Task:
        """
        Creates the CrewAI Task instance for defining the mapping.
        Args:
            manager_agent (Agent): The Manager agent overseeing the hierarchical process.
            context (str): The comprehensive context string assembled by ContextManager, including
                           full C++ source, Godot structure details, global summaries, etc.
            package_id (str): The ID of the work package being processed.

        Returns:
            Task: The CrewAI Task object assigned to the manager.
        """
        logger.info(f"Creating HierarchicalDefineMappingTask for Manager (Package: {package_id})")

        # Example generation remains the same as it describes the FINAL output format
        example_output_dict = MappingOutput(
            package_id=package_id, # Use the actual package_id
            mapping_strategy=(
                "Map C++ player controller to Godot CharacterBody3D script. Use Input singleton for actions. Manage stats via custom PlayerStats Resource."
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
            "**Your Goal as Manager:** Orchestrate the definition of a detailed conversion mapping plan for the C++ work package '{package_id}'. You will coordinate a team of specialized agents using the provided context.\n\n"

            "**Provided Context Includes:**\n"
            "- Full C++ source code for the package.\n"
            "- Proposed Godot project structure (JSON and notes) from Step 3.\n"
            "- Global context (summaries of all packages, existing Godot files).\n"
            "- Potentially, existing mapping/Godot files for refinement and feedback from previous runs.\n\n"
            "--- START OF PROVIDED CONTEXT ---\n"
            f"{context}\n"
            "--- END OF PROVIDED CONTEXT ---\n\n"

            "**Your Orchestration Steps (Delegate to appropriate agents):**\n"
            "1.  **Analyze C++ Code:** Delegate to 'CppCodeAnalystAgent'. Provide relevant C++ source from context. Goal: Understand C++ code purpose, structure, key elements.\n"
            "2.  **Analyze Godot Structure:** Delegate to 'GodotStructureAnalystAgent'. Provide proposed Godot structure (JSON, notes) from context. Goal: Understand the intended Godot architecture.\n"
            "3.  **Define Strategy:** Delegate to 'ConversionStrategistAgent'. Provide the outputs from steps 1 & 2, plus any existing mapping/feedback from context. Goal: Develop a high-level C++ to Godot mapping strategy string.\n"
            "4.  **Decompose Tasks:** Delegate to 'TaskDecomposerAgent'. Provide outputs from steps 1, 2, & 3. Goal: Break down the strategy into granular Task Groups and Mapping Tasks (list of structured objects).\n"
            "5.  **Format Output:** Delegate to 'JsonOutputFormatterAgent'. Provide the strategy string (from step 3) and the decomposed task groups (from step 4), plus the `package_id` ('{package_id}'). Goal: Combine inputs into a single, valid JSON object string conforming precisely to the `MappingOutput` Pydantic model.\n\n"

            "**Final Output:** The final result of this orchestration MUST be the single, raw JSON object string produced by the 'JsonOutputFormatterAgent'."
        )

        return Task(
            description=full_description,
            expected_output=(
                "A **single, valid JSON object string** adhering strictly to the `MappingOutput` Pydantic model structure. "
                "This JSON object contains the `package_id`, the `mapping_strategy` (as a string), and a list of `task_groups`, where each group contains detailed, actionable `tasks` for the conversion process. "
                "The output MUST NOT contain any text before or after the JSON object, and MUST NOT include markdown formatting like ```json."
                f"\n\nExample of the required raw JSON output format:\n{example_json_output}"
            ),
            agent=manager_agent, # Task assigned to the manager
            # Context is implicitly passed via the description and crew memory
            output_pydantic=MappingOutput # Ensure final output matches the model
        )
