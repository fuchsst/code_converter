# src/utils/formatting_utils.py
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
from src.logger_setup import get_logger

logger = get_logger(__name__)

def _build_tree(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to build a tree structure from flat node paths."""
    tree = {}
    for node_data in nodes:
        path_str = node_data.get("node_path", "")
        if not path_str or path_str == "/": # Handle root node or invalid path
             # Place root node directly if path is just "/"
             if path_str == "/":
                  tree['/'] = {'data': node_data, 'children': {}}
             else:
                  logger.warning(f"Node '{node_data.get('name')}' has invalid or missing node_path: '{path_str}'. Skipping.")
             continue

        # Normalize path: remove leading/trailing slashes, split
        parts = path_str.strip('/').split('/')
        current_level = tree
        full_path_processed = ""

        for i, part in enumerate(parts):
            full_path_processed += "/" + part
            is_last_part = (i == len(parts) - 1)

            if is_last_part:
                # If this is the node itself, add its data
                if part not in current_level:
                    current_level[part] = {'data': node_data, 'children': {}}
                else:
                    # Node already exists (e.g., parent was processed), add data
                    current_level[part]['data'] = node_data
                    # Ensure children dict exists if it was just a placeholder parent
                    if 'children' not in current_level[part]:
                         current_level[part]['children'] = {}
            else:
                # If this is a parent path part, ensure it exists
                if part not in current_level:
                    # Create placeholder parent node if it doesn't exist
                    current_level[part] = {'data': None, 'children': {}}
                elif 'children' not in current_level[part]:
                     # Ensure children dict exists if it was created implicitly before
                     current_level[part]['children'] = {}

                # Move to the next level
                current_level = current_level[part]['children']

    # Special handling for root node if defined with path "/"
    if '/' in tree:
         root_node_entry = tree.pop('/')
         # Merge children of explicitly defined root with top-level nodes
         # This assumes the root node's children paths start from the root, e.g., "/Root/Child"
         # The logic above should handle placing children correctly under their parents.
         # We just need to ensure the root node itself is represented at the top level.
         # Let's represent the root node data at the top level if it exists.
         # The tree structure built above should inherently handle the hierarchy.
         # We might return the root node entry directly if it's the only top-level item.
         if not tree: # If only the root node was defined
              return {'/': root_node_entry} # Keep it distinct? Or return root_node_entry['children']? Let's keep it distinct.
         else:
              # This case is complex: root defined AND other top-level nodes? Might indicate inconsistent paths.
              # Let's merge the root's children into the main tree if possible, or log a warning.
              logger.warning("Structure has both an explicit root node ('/') and other top-level nodes. Hierarchy might be ambiguous.")
              # For simplicity, let's just add the root node back as a top-level entry.
              tree['/'] = root_node_entry


    return tree


def _format_node(node_data: Dict[str, Any]) -> str:
    """Formats a single node's data into a string."""
    if not node_data:
        return "[Unknown Node]" # Placeholder for implicitly created parents
    name = node_data.get('name', '[Unnamed]')
    node_type = node_data.get('type', '[Unknown Type]')
    script = node_data.get('script_path')
    formatted = f"{name} ({node_type})"
    if script:
        formatted += f" -> {os.path.basename(script)}" # Show only script filename
    return formatted

def _format_tree_recursive(tree: Dict[str, Any], indent: str = "") -> List[str]:
    """Recursive helper to format the tree into markdown lines."""
    lines = []
    # Sort keys for consistent output, handle root node ('/') potentially
    keys = sorted(tree.keys())
    if '/' in keys: # Ensure root node comes first if present
         keys.remove('/')
         keys.insert(0, '/')

    for name, node_info in tree.items():
        node_data = node_info.get('data')
        # Format the current node
        # Handle root node display slightly differently if needed
        display_name = name if name != '/' else node_data.get('name', 'Root') if node_data else 'Root'

        # Use node_data if available, otherwise create a placeholder string
        if node_data:
             formatted_node = _format_node(node_data)
             lines.append(f"{indent}- {formatted_node}")
        elif name == '/': # Handle case where root was implicitly created but has children
             lines.append(f"{indent}- [Root Node Placeholder]") # Indicate it's a placeholder
        else:
             # This case represents an intermediate path component that wasn't explicitly defined as a node
             # Example: /ShipBase/Subsystems/Engine where Subsystems wasn't in the nodes list
             lines.append(f"{indent}- [{name} (Implicit Parent)]")


        # Recursively format children
        children = node_info.get('children', {})
        if children:
            lines.extend(_format_tree_recursive(children, indent + "  "))
    return lines

def format_structure_to_markdown(structure_data: Dict[str, Any]) -> str:
    """
    Formats the Godot structure definition (from JSON) into a hierarchical
    Markdown representation.

    Args:
        structure_data: The dictionary loaded from package_..._structure.json.

    Returns:
        A string containing the Markdown formatted structure.
    """
    markdown_lines = []
    scenes = structure_data.get("scenes", [])
    scripts = structure_data.get("scripts", [])
    resources = structure_data.get("resources", [])
    migration_scripts = structure_data.get("migration_scripts", [])
    notes = structure_data.get("notes")

    if not scenes and not scripts and not resources and not migration_scripts:
        return "No structure defined (empty scenes, scripts, resources, migration_scripts)."

    markdown_lines.append("### Proposed Godot Structure:")

    # Format Scenes
    if scenes:
        markdown_lines.append("\n**Scenes & Nodes:**")
        for scene in scenes:
            scene_path = scene.get('path', '[Unknown Scene Path]')
            nodes = scene.get('nodes', [])
            markdown_lines.append(f"\n- **`{scene_path}`**")
            if nodes:
                # Build and format the tree for this scene
                try:
                    node_tree = _build_tree(nodes)
                    markdown_lines.extend(_format_tree_recursive(node_tree, indent="  "))
                except Exception as e:
                     logger.error(f"Error building node tree for scene '{scene_path}': {e}", exc_info=True)
                     markdown_lines.append("  - Error formatting node hierarchy.")
            else:
                markdown_lines.append("  - (No nodes defined)")

    # Format Standalone Scripts (those not directly tied to a scene node in the structure)
    # Note: The tree formatting above already shows scripts attached to nodes.
    # This section lists scripts defined in the "scripts" list, potentially including resources scripts.
    if scripts:
        markdown_lines.append("\n**Scripts:**")
        for script in scripts:
            script_path = script.get('path', '[Unknown Script Path]')
            purpose = script.get('purpose', '')
            # attached_scene = script.get('attached_to_scene') # Info already in scene tree
            # attached_node = script.get('attached_to_node') # Info already in scene tree
            markdown_lines.append(f"- `{script_path}`: {purpose}")

     # Format Resources
    if resources:
        markdown_lines.append("\n**Resources:**")
        for resource in resources:
            res_path = resource.get('path', '[Unknown Resource Path]')
            purpose = resource.get('purpose', '')
            script_ref = resource.get('script')
            line = f"- `{res_path}`: {purpose}"
            if script_ref:
                line += f" (Script: `{os.path.basename(script_ref)}`)"
            markdown_lines.append(line)

    # Format Migration Scripts
    if migration_scripts:
        markdown_lines.append("\n**Migration Scripts:**")
        for m_script in migration_scripts:
            m_path = m_script.get('path', '[Unknown Migration Script Path]')
            m_purpose = m_script.get('purpose', '')
            m_type = m_script.get('script_type', 'Unknown')
            markdown_lines.append(f"- `{m_path}` ({m_type}): {m_purpose}")


    # Add Notes
    if notes:
        markdown_lines.append("\n**Notes:**")
        markdown_lines.append(notes)

    return "\n".join(markdown_lines)

# Example Usage (for testing)
if __name__ == '__main__':
    example_structure = {
      "scenes": [
        {
          "path": "res://scenes/ships_weapons/base_ship.tscn",
          "nodes": [
            { "name": "ShipBase", "type": "RigidBody3D", "node_path": "/", "script_path": "res://scripts/ship/ship_base.gd" },
            { "name": "Model", "type": "Node3D", "node_path": "/ShipBase/Model", "script_path": None },
            { "name": "CollisionShape", "type": "CollisionShape3D", "node_path": "/ShipBase/CollisionShape", "script_path": None },
            { "name": "WeaponSystem", "type": "Node", "node_path": "/ShipBase/WeaponSystem", "script_path": "res://scripts/ship/weapon_system.gd" },
            { "name": "Hardpoint1", "type": "Marker3D", "node_path": "/ShipBase/WeaponSystem/Hardpoint1", "script_path": None },
            { "name": "ShieldSystem", "type": "Node", "node_path": "/ShipBase/ShieldSystem", "script_path": "res://scripts/ship/shield_system.gd" },
            { "name": "DamageSystem", "type": "Node", "node_path": "/ShipBase/DamageSystem", "script_path": "res://scripts/ship/damage_system.gd" },
            { "name": "EngineSystem", "type": "Node", "node_path": "/ShipBase/EngineSystem", "script_path": "res://scripts/ship/engine_system.gd" },
            { "name": "AIController", "type": "Node", "node_path": "/ShipBase/AIController", "script_path": "res://scripts/ai/ai_controller.gd" },
            { "name": "AnimationPlayer", "type": "AnimationPlayer", "node_path": "/ShipBase/AnimationPlayer", "script_path": None },
            { "name": "Subsystems", "type": "Node3D", "node_path": "/ShipBase/Subsystems", "script_path": None },
            { "name": "PowerCore", "type": "Node", "node_path": "/ShipBase/Subsystems/PowerCore", "script_path": None }, # Implicit parent example
            { "name": "CMeasureComponent", "type": "Node", "node_path": "/ShipBase/CMeasureComponent", "script_path": "res://scripts/cmeasure/cmeasure_component.gd" }
          ]
        },
        {
             "path": "res://scenes/ui/hud.tscn",
             "nodes": [
                  {"name": "HUD", "type": "CanvasLayer", "node_path": "/HUD", "script_path": "res://scripts/ui/hud.gd"},
                  {"name": "HealthBar", "type": "TextureProgressBar", "node_path": "/HUD/HealthBar", "script_path": None}
             ]
        }
      ],
      "scripts": [
        {"path": "res://scripts/ship/ship_base.gd", "purpose": "Base logic for ship behavior.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "ShipBase"},
        {"path": "res://scripts/ship/weapon_system.gd", "purpose": "Manages weapon firing and state.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "WeaponSystem"},
        {"path": "res://scripts/ship/shield_system.gd", "purpose": "Handles shield strength and regeneration.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "ShieldSystem"},
        {"path": "res://scripts/ship/damage_system.gd", "purpose": "Processes incoming damage.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "DamageSystem"},
        {"path": "res://scripts/ship/engine_system.gd", "purpose": "Controls ship movement and thrust.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "EngineSystem"},
        {"path": "res://scripts/ai/ai_controller.gd", "purpose": "AI logic for controlling the ship.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "AIController"},
        {"path": "res://scripts/cmeasure/cmeasure_component.gd", "purpose": "Handles countermeasures.", "attached_to_scene": "res://scenes/ships_weapons/base_ship.tscn", "attached_to_node": "CMeasureComponent"},
        {"path": "res://scripts/resources/ship_stats.gd", "purpose": "Defines the ShipStats custom resource.", "attached_to_scene": None, "attached_to_node": None},
        {"path": "res://scripts/ui/hud.gd", "purpose": "Updates HUD elements.", "attached_to_scene": "res://scenes/ui/hud.tscn", "attached_to_node": "HUD"}
      ],
      "resources": [
        {"path": "res://resources/ships/fighter_stats.tres", "purpose": "Stats for the fighter ship.", "script": "res://scripts/resources/ship_stats.gd"}
      ],
      "migration_scripts": [
         {"path": "res://migration/convert_textures.py", "purpose": "Converts old DDS textures.", "script_type": "Python"}
      ],
      "notes": "This structure defines a base ship scene with common systems attached as child nodes. A separate HUD scene is included."
    }

    markdown_output = format_structure_to_markdown(example_structure)
    print(markdown_output)


def format_packages_summary_to_markdown(packages_summary_data: Dict[str, Any]) -> str:
    """
    Formats the global packages summary (from packages.json) into Markdown.

    Args:
        packages_summary_data: The dictionary loaded from packages.json.

    Returns:
        A string containing the Markdown formatted summary.
    """
    markdown_lines = ["### Global Packages Summary:"]
    if isinstance(packages_summary_data, dict) and "packages" in packages_summary_data:
        packages = packages_summary_data.get("packages", {})
        if not packages:
             markdown_lines.append("\n- (No packages defined in summary)")
             return "\n".join(markdown_lines)

        for pkg_id, pkg_data in packages.items():
            desc = pkg_data.get('description', 'N/A')
            files_data = pkg_data.get('files', {}) # Expecting dict format now based on Step 2 output
            markdown_lines.append(f"\n- **Package:** `{pkg_id}`")
            markdown_lines.append(f"  - **Description:** {desc}")
            if isinstance(files_data, dict) and files_data:
                markdown_lines.append("  - **Files & Roles:**")
                for f_path, f_role in files_data.items():
                     role_str = f_role if f_role else "N/A"
                     markdown_lines.append(f"    - `{f_path}`: {role_str}")
            elif isinstance(files_data, list) and files_data: # Fallback for old list format
                 markdown_lines.append(f"  - **Files:** {', '.join(f'`{f}`' for f in files_data)}")
            else:
                 markdown_lines.append("  - **Files:** (None listed or invalid format)")
    else:
        markdown_lines.append("\n- (Invalid or empty summary data structure)")

    return "\n".join(markdown_lines)


def format_existing_files_to_markdown(existing_files_list: List[str], title: str = "Existing Godot Output Files") -> str:
    """
    Formats a list of existing file paths into a Markdown list.

    Args:
        existing_files_list: A list of file path strings.
        title: The title for the Markdown section.

    Returns:
        A string containing the Markdown formatted list.
    """
    markdown_lines = [f"### {title}:"]
    if isinstance(existing_files_list, list) and existing_files_list:
        for file_path in existing_files_list:
            markdown_lines.append(f"- `{file_path}`")
    else:
        markdown_lines.append("\n- (None found or list is empty)")

    return "\n".join(markdown_lines)
