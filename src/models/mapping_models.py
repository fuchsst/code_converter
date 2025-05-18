# src/models/mapping_models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MappingTask(BaseModel):
    """Defines a single, granular conversion task."""
    task_title: str = Field(..., description="A concise title for the task (e.g., 'Implement Player Movement Input').")
    task_description: str = Field(..., description="A short description explaining why this task is needed, how it relates to the C++ source, and the general implementation approach in Godot.")
    input_source_files: List[str] = Field(..., description="List of relevant C++ source/header file paths for reference.")
    output_godot_file: str = Field(..., description="The relative path to the target Godot script/scene/resource to be created/modified (e.g., 'res://scripts/player/player_movement.gd'). Must align with Step 3 structure.")
    target_element: Optional[str] = Field(None, description="The specific function, method, or node within the Godot file (e.g., '_physics_process', 'setup_animations').")

class TaskGroup(BaseModel):
    """Groups related tasks under a specific feature or component."""
    group_title: str = Field(..., description="Title for the task group (e.g., 'Player Character Controller', 'Inventory System UI').")
    feature_description: str = Field(..., description="A longer description of the overall feature or component this group implements.")
    godot_features: str = Field(..., description="Description of the specific Godot nodes, APIs, or patterns that will be used to implement this feature (e.g., 'CharacterBody3D for movement, AnimationPlayer for animations, Custom Resource for stats').")
    tasks: List[MappingTask] = Field(..., description="A list of granular tasks belonging to this group.")

class MappingOutput(BaseModel):
    """Defines the overall expected JSON output structure for the mapping definition."""
    package_id: str = Field(..., description="The ID of the work package this mapping belongs to.")
    mapping_strategy: str = Field(..., description="High-level summary of the mapping approach for this package.")
    task_groups: List[TaskGroup] = Field(..., description="A list of task groups detailing the conversion plan.")

# --- New Model for Flow State ---
class DefineMappingFlowState(BaseModel):
    """
    Manages the state of the DefineMappingPipelineFlow.
    The 'id' field for Flow persistence will be the package_id.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the flow instance, MUST be the package_id. Set during kickoff.")
    package_id: Optional[str] = Field(None, description="The ID of the work package being processed. Set during kickoff.")
    initial_context_str: Optional[str] = Field(None, description="The comprehensive context string provided at the start of the flow for the package.")

    # Contexts prepared for specific agents/steps from the initial_context_str or other sources
    cpp_source_for_analyst: Optional[str] = Field(None, description="Specific C++ source code content for the CppCodeAnalystAgent.")
    godot_structure_for_analyst: Optional[str] = Field(None, description="Specific Godot project structure definition for the GodotStructureAnalystAgent.")
    existing_mapping_for_strategist: Optional[str] = Field(None, description="Existing mapping JSON string for the ConversionStrategistAgent, if any.")
    feedback_for_strategist: Optional[str] = Field(None, description="Feedback text for the ConversionStrategistAgent, if any.")
    general_instructions: Optional[str] = Field(None, description="General instructions for the overall mapping process, if any, extracted from initial_context_str.")

    # Raw outputs from intermediate tasks
    cpp_analysis_raw: Optional[str] = Field(None, description="Raw textual output from the C++ analysis task.")
    godot_analysis_raw: Optional[str] = Field(None, description="Raw textual output from the Godot structure analysis task.")
    strategy_raw: Optional[str] = Field(None, description="Raw textual output of the conversion strategy.")
    # Storing task_groups as a JSON string as agent output might be a string.
    # The flow will parse this before passing to the final assembly task.
    task_groups_json_str: Optional[str] = Field(None, description="JSON string representing the list of TaskGroup objects from the decomposition step.")

    # Final output
    final_mapping_output: Optional[MappingOutput] = Field(None, description="The final structured MappingOutput object.")

    # Flow status tracking
    current_step_name: str = Field("initial", description="Name of the current or last executed/attempted step in the flow.")
    error_message: Optional[str] = Field(None, description="Any error message encountered during flow execution.")
    is_complete: bool = Field(False, description="Flag indicating if the flow has completed successfully.")
    is_failed: bool = Field(False, description="Flag indicating if the flow has failed.")
    # You could add retry_count, timestamps for steps, etc., if needed for more advanced state tracking.

    class Config:
        validate_assignment = True # Ensures type checking on attribute assignment
