# src/tasks/step4/decompose_component_task.py
from crewai import Task, Agent # Added Agent for type hinting
from typing import Dict, Any
from src.logger_setup import get_logger
from src.models.mapping_models import MappingTask # For expected output structure reference

logger = get_logger(__name__)

def create_decompose_component_task(
    agent: Agent, 
    component_type: str, 
    component_data: Dict[str, Any], 
    cpp_analysis: str, 
    godot_analysis: str, 
    strategy: str, 
    package_id: str, 
    component_index: int
) -> Task:
    """
    Creates a task for decomposing a specific component (script, resource, scene, etc.) 
    into granular conversion tasks.

    Args:
        agent: The TaskDecomposerAgent that will perform this task.
        component_type: Type of component (e.g., "scripts", "resources", "scenes").
        component_data: Dictionary containing data for this specific component 
                        (e.g., path, purpose, nodes for a scene).
        cpp_analysis: The textual analysis of the C++ code.
        godot_analysis: The textual analysis of the proposed Godot structure.
        strategy: The overall high-level conversion strategy string.
        package_id: The ID of the work package.
        component_index: A unique index for this component within its type category 
                         (e.g., if it's the 3rd script, index could be 2).

    Returns:
        A CrewAI Task object.
    """
    
    # --- Prepare Component-Specific Context for the Description ---
    component_path = component_data.get('path', f'Unknown_{component_type}_{component_index}')
    component_purpose = component_data.get('purpose', 'Not specified')
    
    component_name_for_log = component_path
    if 'name' in component_data and component_type == 'scenes':
        component_name_for_log = f"{component_data['name']} ({component_path})"
    
    logger.info(f"Creating DecomposeComponentTask for {component_type} '{component_name_for_log}' in package {package_id}")

    component_context_str = f"**Component Type:** {component_type.capitalize()}\n"
    component_context_str += f"**Component Path/Identifier:** `{component_path}`\n"

    if component_type == "scripts":
        component_context_str += f"**Stated Purpose:** {component_purpose}\n"
        if component_data.get('class_name'):
            component_context_str += f"**Class Name:** `{component_data.get('class_name')}`\n"
    elif component_type == "resources":
        component_context_str += f"**Stated Purpose:** {component_purpose}\n"
        if component_data.get('script'):
            component_context_str += f"**Associated Script:** `{component_data.get('script')}`\n"
        elif component_data.get('type'):
             component_context_str += f"**Resource Type:** `{component_data.get('type')}`\n"
    elif component_type == "migration_scripts":
        component_context_str += f"**Stated Purpose:** {component_purpose}\n"
        if component_data.get('script_type'):
            component_context_str += f"**Script Type:** {component_data.get('script_type')}\n"
    elif component_type == "scenes":
        scene_name = component_data.get('name', 'Unnamed Scene')
        component_context_str += f"**Scene Name:** {scene_name}\n"
        component_context_str += f"**Stated Purpose:** {component_purpose}\n"
        nodes_info_parts = []
        for i, node in enumerate(component_data.get('nodes', [])):
            node_str = (
                f"  - Node {i+1}: {node.get('name', 'Unnamed')} (`{node.get('type', 'Unknown')}`)"
                f"{f', Script: `{node.get("script_path")}`' if node.get('script_path') else ''}"
                f"{f', Purpose: {node.get("purpose")}' if node.get('purpose') else ''}"
            )
            nodes_info_parts.append(node_str)
        if nodes_info_parts:
            component_context_str += "**Key Nodes in Scene:**\n" + "\n".join(nodes_info_parts) + "\n"
        else:
            component_context_str += "**Nodes:** _No specific nodes detailed in structure._\n"

    # --- Build the Full Task Description ---
    description = (
        f"**Task: Decompose {component_type.upper()} Component into Conversion Tasks for Package '{package_id}'**\n\n"
        f"**Objective:**\n"
        f"Based on the overall conversion strategy and detailed analyses, break down the implementation of the "
        f"following specific Godot component into a list of granular, actionable `MappingTask` items. "
        f"Each `MappingTask` should be small enough for a subsequent code generation agent to handle effectively.\n\n"
        f"--- TARGET GODOT COMPONENT DETAILS ---\n{component_context_str.strip()}\n--- END COMPONENT DETAILS ---\n\n"
        f"--- OVERALL CONVERSION STRATEGY (Reference) ---\n{strategy}\n--- END STRATEGY ---\n\n"
        f"--- C++ CODE ANALYSIS (Reference) ---\n{cpp_analysis}\n--- END C++ ANALYSIS ---\n\n"
        f"--- GODOT STRUCTURE ANALYSIS (Reference) ---\n{godot_analysis}\n--- END GODOT ANALYSIS ---\n\n"
        f"**Your Decomposition Process:**\n"
        f"1.  **Identify C++ Counterparts:** Determine which parts of the C++ codebase (classes, functions, data) are relevant to implementing this specific Godot component, according to the strategy.\n"
        f"2.  **Map to Godot Elements:** For each relevant C++ part, decide how it maps to Godot elements within this component (e.g., specific methods in a script, node configurations in a scene, properties in a resource).\n"
        f"3.  **Define Granular Tasks:** Create a `MappingTask` for each distinct piece of work. For each `MappingTask`, provide:\n"
        f"    - `task_title`: A clear, concise title (e.g., 'Implement Player Input Handling in player.gd').\n"
        f"    - `task_description`: A detailed explanation of what C++ logic/data to convert, how to implement it in Godot for this component, and why it's needed.\n"
        f"    - `input_source_files`: A list of relevant C++ source/header file paths for reference.\n"
        f"    - `output_godot_file`: The specific target Godot file path for this task (e.g., `res://scripts/player.gd`, `res://scenes/main_level.tscn`). This MUST align with the `Component Path/Identifier` above if it's the primary file for this component, or be a related file if the task involves, for example, a utility script used by this component.\n"
        f"    - `target_element`: (Optional) The specific function, method name, node path within a scene, or section within the `output_godot_file` this task focuses on (e.g., `_physics_process`, `Player/Camera3D`, `// Player Stats Section`).\n\n"
        f"Focus on creating specific, actionable, and self-contained tasks. The output should be a list of these `MappingTask` items."
    )
    
    expected_output = (
        "A list of `MappingTask` items (structured as dictionaries or JSON objects) for this specific component. Each item must contain:\n"
        "- `task_title` (string): Clear and concise title.\n"
        "- `task_description` (string): Detailed conversion approach, linking C++ to Godot.\n"
        "- `input_source_files` (list of strings): Relevant C++ source file paths.\n"
        "- `output_godot_file` (string): Target Godot file path (e.g., res://path/to/file.gd or .tscn or .tres).\n"
        "- `target_element` (string, optional): Specific function, method, node path, or section in the output file."
    )

    return Task(
        name=f"DecomposeComponent_{component_type}_{package_id}_{component_index}",
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
        # The output of this task should be a list of objects/dictionaries,
        # each conforming to the MappingTask Pydantic model.
        # output_json=True or output_pydantic=List[MappingTask] could be used if the agent is prompted
        # to produce a JSON string representing this list.
    )
