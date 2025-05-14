# src/tasks/step4/analyze_godot_structure_task.py
from crewai import Task, Agent 
from typing import Dict, Any 
from src.logger_setup import get_logger

logger = get_logger(__name__)

def create_analyze_godot_structure_task(agent: Agent, context: str, package_id: str) -> Task:
    """
    Creates a task for analyzing the proposed Godot project structure.

    Args:
        agent: The GodotStructureAnalystAgent that will perform this task.
        context: A string containing the Godot project structure definition
                 and any relevant notes for the specified package.
        package_id: The ID of the work package being processed.

    Returns:
        A CrewAI Task object.
    """
    logger.info(f"Creating AnalyzeGodotStructureTask for agent {agent.role} (Package: {package_id})")
    
    description = (
        f"**Task: Analyze Proposed Godot Project Structure for Package '{package_id}'**\n\n"
        f"**Objective:**\n"
        f"Thoroughly analyze the provided Godot project structure definition. Your goal is to deeply understand "
        f"the intended role, purpose, and interrelationships of all proposed components (scenes, nodes, scripts, resources). "
        f"Focus on identifying key architectural choices, organizational patterns, and how different elements are meant to collaborate.\n\n"
        f"**Provided Godot Structure Context for Package '{package_id}':**\n"
        f"```\n{context}\n```\n"
        f"--- END OF PROVIDED CONTEXT ---\n\n"
        f"**Your Analysis Should Cover:**\n"
        f"1.  **Scene Composition:** For each scene, describe its main purpose, key nodes, and how these nodes are organized (hierarchy, important properties, assigned scripts).\n"
        f"2.  **Script Functionality:** For each script, detail its intended purpose, what Godot objects it will be attached to, and its primary responsibilities.\n"
        f"3.  **Resource Usage:** Identify custom resources, their purpose, and how they are likely to be used by scripts or scenes.\n"
        f"4.  **Data Flow & Interactions:** Speculate on how data might flow between different parts of this structure and how components might interact (e.g., via signals, direct calls, resource sharing).\n"
        f"5.  **Overall Architecture:** Summarize the main architectural patterns or design choices evident in the proposed structure (e.g., component-based, scene-per-level, use of singletons).\n"
        f"6.  **Potential Challenges/Considerations:** (Optional) Note any areas that seem complex, ambiguous, or might require special attention during implementation.\n\n"
        f"Your output should be a comprehensive, structured textual analysis that will inform the subsequent conversion strategy and task decomposition phases."
    )
    
    expected_output = (
        "A detailed and structured textual analysis of the Godot project structure, covering:\n"
        "1.  **Scene Hierarchy and Organization:** Purpose, key nodes, and script assignments for each scene.\n"
        "2.  **Script Purposes and Relationships:** Intended functionality and connections for each script.\n"
        "3.  **Resource Usage Patterns:** How custom resources are defined and utilized.\n"
        "4.  **Integration Points & Data Flow:** How components are expected to interact.\n"
        "5.  **Overall Architectural Approach:** Summary of design patterns and choices.\n"
        "6.  (Optional) Potential implementation challenges or points of attention."
    )

    return Task(
        name=f"AnalyzeGodotStructure_{package_id}", 
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False
    )
