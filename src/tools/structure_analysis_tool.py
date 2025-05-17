# src/tools/structure_analysis_tool.py
from crewai.tools import BaseTool
from src.logger_setup import get_logger
from src.core.context_manager import ContextManager
from src.core.state_manager import StateManager # Added for loading artifacts
import json
from typing import Dict, Any, List

logger = get_logger(__name__)

class StructureAnalysisTool(BaseTool):
    name: str = "GodotStructureAnalysisTool"
    description: str = (
        "Analyzes the Godot project structure for a given package ID. "
        "It fetches the structure definition (JSON artifact) and provides a detailed "
        "Markdown-formatted architectural overview, including scenes, nodes, scripts, resources, and notes."
    )
    # context_manager: ContextManager # ContextManager might not be directly needed if StateManager handles artifact loading
    state_manager: StateManager # Tool will use StateManager to load the structure artifact
    context_manager: ContextManager # Keep for potential future use or if some info comes via CM

    def __init__(self, state_manager: StateManager, context_manager: ContextManager, **kwargs):
        super().__init__(state_manager=state_manager, context_manager=context_manager, **kwargs)
        self.state_manager = state_manager
        self.context_manager = context_manager
        self.name = "GodotStructureAnalysisTool"
        self.description = (
            "Analyzes the Godot project structure for a given package ID. "
            "It fetches the structure definition (JSON artifact) and provides a detailed "
            "Markdown-formatted architectural overview, including scenes, nodes, scripts, resources, and notes."
        )

    def _format_node_details(self, node: Dict[str, Any], indent_level: int = 1) -> str:
        indent = "  " * indent_level
        details = [f"{indent}- **{node.get('name', 'UnnamedNode')}** (`{node.get('type', 'UnknownType')}`)"]
        if node.get('node_path'):
            details.append(f"{indent}  - Path: `{node.get('node_path')}`")
        if node.get('script_path'):
            details.append(f"{indent}  - Script: `{node.get('script_path')}`")
        if node.get('purpose'):
            details.append(f"{indent}  - Purpose: {node.get('purpose')}")
        if node.get('children'):
            details.append(f"{indent}  - Children:")
            for child_node in node.get('children', []):
                details.append(self._format_node_details(child_node, indent_level + 2))
        return "\n".join(details)

    def _run(self, package_id: str) -> str:
        """
        Analyzes the Godot project structure for the specified package_id.
        It loads the structure JSON artifact associated with the package_id.
        
        Args:
            package_id: The ID of the package whose structure is to be analyzed.
            
        Returns:
            A detailed Markdown analysis of the structure, or an error message.
        """
        try:
            logger.info(f"Attempting to analyze Godot structure for package ID: {package_id}")
            pkg_info = self.state_manager.get_package_info(package_id)
            if not pkg_info:
                return f"Error: Package information not found for ID '{package_id}'."

            structure_artifact_filename = pkg_info.get('artifacts', {}).get('structure_json')
            if not structure_artifact_filename:
                return f"Error: Structure definition JSON artifact not found for package '{package_id}'."
            
            logger.info(f"Loading structure artifact: {structure_artifact_filename} for package {package_id}")
            structure_data = self.state_manager.load_artifact(structure_artifact_filename, expect_json=True)
            
            if not structure_data:
                return f"Error: Failed to load or parse structure JSON artifact '{structure_artifact_filename}' for package '{package_id}'."
            if not isinstance(structure_data, dict):
                 return f"Error: Loaded structure artifact '{structure_artifact_filename}' for package '{package_id}' is not a valid JSON object (dictionary)."

            logger.info(f"Successfully loaded structure data for package {package_id}.")
            analysis_parts = [f"# Godot Structure Analysis for Package: {package_id}"]

            # --- Overview ---
            scenes_data = structure_data.get("scenes", [])
            scripts_data = structure_data.get("scripts", [])
            resources_data = structure_data.get("resources", [])
            migration_scripts_data = structure_data.get("migration_scripts", [])
            
            analysis_parts.append("\n## I. Overview")
            analysis_parts.append(f"- **Total Scenes Defined:** {len(scenes_data)}")
            analysis_parts.append(f"- **Total Scripts Defined:** {len(scripts_data)}")
            analysis_parts.append(f"- **Total Resources Defined:** {len(resources_data)}")
            analysis_parts.append(f"- **Total Migration Scripts Defined:** {len(migration_scripts_data)}")

            # --- Scenes ---
            analysis_parts.append("\n## II. Scenes Details")
            if scenes_data:
                for i, scene in enumerate(scenes_data):
                    scene_name = scene.get('name', f"UnnamedScene_{i+1}")
                    scene_path = scene.get('path', 'N/A')
                    analysis_parts.append(f"\n### Scene {i+1}: `{scene_path}` ({scene_name})")
                    if scene.get('purpose'):
                        analysis_parts.append(f"- **Purpose:** {scene.get('purpose')}")
                    
                    nodes_data = scene.get("nodes", [])
                    if nodes_data:
                        analysis_parts.append("- **Key Nodes:**")
                        for node in nodes_data:
                            analysis_parts.append(self._format_node_details(node))
                    else:
                        analysis_parts.append("- _No nodes defined for this scene._")
                    if scene.get('notes'):
                        analysis_parts.append(f"- **Scene Notes:** {scene.get('notes')}")
            else:
                analysis_parts.append("_No scenes defined in the structure._")

            # --- Scripts (Concise) ---
            analysis_parts.append("\n## III. Scripts Details")
            if scripts_data:
                for script in scripts_data:
                    script_path = script.get('path', 'N/A')
                    class_name = script.get('class_name', '') # Default to empty if not present
                    purpose = script.get('purpose', 'N/A')
                    analysis_parts.append(f"- `{script_path}` ({class_name if class_name else 'No Class Name'}): {purpose}")
            else:
                analysis_parts.append("_No scripts defined in the structure._")

            # --- Resources (Concise) ---
            analysis_parts.append("\n## IV. Resources Details")
            if resources_data:
                for resource in resources_data:
                    res_path = resource.get('path', 'N/A')
                    res_type = resource.get('type', resource.get('script', 'N/A')) 
                    purpose = resource.get('purpose', 'N/A')
                    analysis_parts.append(f"- `{res_path}` (Type/Script: `{res_type}`): {purpose}")
            else:
                analysis_parts.append("_No custom resources defined in the structure._")

            # --- Migration Scripts (Concise) ---
            analysis_parts.append("\n## V. Migration Scripts Details")
            if migration_scripts_data:
                for mig_script in migration_scripts_data:
                    mig_path = mig_script.get('path', 'N/A')
                    script_type = mig_script.get('script_type', 'N/A')
                    purpose = mig_script.get('purpose', 'N/A')
                    analysis_parts.append(f"- `{mig_path}` (Type: `{script_type}`): {purpose}")
            else:
                analysis_parts.append("_No migration scripts defined in the structure._")
            
            # --- General Notes ---
            if structure_data.get("notes"):
                analysis_parts.append("\n## VI. General Structure Notes")
                analysis_parts.append(structure_data["notes"])
            
            return "\n".join(analysis_parts)
        
        except Exception as e:
            logger.error(f"Error in {self.name} for package '{package_id}': {e}", exc_info=True)
            return f"Error during Godot structure analysis for package '{package_id}': {e}"
