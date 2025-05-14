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
