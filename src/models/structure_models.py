# --- Pydantic Models for Structured Output ---
from typing import List, Optional
from pydantic import BaseModel, Field


class GodotScript(BaseModel):
    """Defines a proposed Godot script file."""
    path: str = Field(..., description="The proposed relative path for the script within the Godot project (e.g., 'res://scripts/my_package/player_controller.gd').")
    purpose: str = Field(..., description="A brief description of the script's main responsibility.")

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
