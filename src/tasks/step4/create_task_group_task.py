# src/tasks/step4/create_task_group_task.py
from crewai import Task, Agent # Added Agent for type hinting
from typing import List, Any, Dict # Added Any, Dict for type hinting
from src.logger_setup import get_logger
from src.models.mapping_models import TaskGroup, MappingTask # For output structure reference

logger = get_logger(__name__)

def create_task_group_task(
    agent: Agent, 
    component_type: str, 
    component_tasks_outputs: List[Any], # Expects a list of lists of MappingTask-like dicts/objects
    package_id: str # Added package_id for logging and context
) -> Task:
    """
    Creates a task for combining multiple component-specific task lists into a single TaskGroup.

    Args:
        agent: The TaskDecomposerAgent (or a dedicated grouping agent) that will perform this task.
        component_type: Type of component (e.g., "scripts", "resources", "scenes") this group is for.
        component_tasks_outputs: A list where each item is the output from a 
                                 `create_decompose_component_task`. Each such output is expected 
                                 to be a list of MappingTask-like dictionaries/objects.
        package_id: The ID of the work package for context.

    Returns:
        A CrewAI Task object.
    """
    logger.info(f"Creating CreateTaskGroupTask for {component_type} in package {package_id}")

    # --- Prepare Titles and Context for Task Description ---
    group_title_suggestions = {
        "scripts": "Script Implementation Tasks",
        "resources": "Resource Creation and Configuration Tasks",
        "migration_scripts": "Data Migration and Conversion Tasks",
        "scenes": "Scene Construction and Integration Tasks"
    }
    suggested_group_title = group_title_suggestions.get(component_type, f"{component_type.replace('_', ' ').title()} Component Tasks")

    # For the description, summarize what the input `component_tasks_outputs` represents.
    # The actual data will be passed in the task's context/memory.
    input_summary = (
        f"A collection of detailed `MappingTask` lists. Each list was generated for an individual "
        f"'{component_type}' component (e.g., a specific script file, a particular scene definition) "
        f"belonging to package '{package_id}'. Your goal is to consolidate all these individual tasks "
        f"into one cohesive `TaskGroup` for all '{component_type}'."
    )

    # --- Build the Full Task Description ---
    description = (
        f"**Task: Create a Unified TaskGroup for all '{component_type.upper()}' Components in Package '{package_id}'**\n\n"
        f"**Objective:**\n"
        f"You have received multiple sets of `MappingTask` items. Each set corresponds to the detailed conversion tasks "
        f"for a single, specific '{component_type}' component (like one script, one scene, etc.). "
        f"Your job is to consolidate all these individual `MappingTask` items from all provided sets into a single, "
        f"well-defined `TaskGroup` object that represents all work for the '{component_type}' category.\n\n"
        f"**Input Summary (Actual data will be in context/memory):**\n{input_summary}\n\n"
        f"**Your TaskGroup Creation Steps:**\n"
        f"1.  **Define `group_title`:** Create a clear and descriptive title for this group. Suggested: \"{suggested_group_title}\".\n"
        f"2.  **Write `feature_description`:** Provide a comprehensive description that explains the overall purpose and role of all '{component_type}' components within the game/project for package '{package_id}'.\n"
        f"3.  **List `godot_features`:** Identify and list the key Godot nodes, APIs, patterns, or engine features that will generally be used across the implementation of these '{component_type}' components.\n"
        f"4.  **Compile `tasks`:** Collect all individual `MappingTask` items from ALL the input sets and combine them into a single, flat list under the `tasks` field of your `TaskGroup`.\n\n"
        f"The output should be a single, structured `TaskGroup` object (or a dictionary that can be parsed into one)."
    )
    
    expected_output = (
        f"A single `TaskGroup` object (structured as a dictionary or JSON object) for all '{component_type}' components, containing:\n"
        f"- `group_title` (string): e.g., \"{suggested_group_title}\".\n"
        f"- `feature_description` (string): Comprehensive description of the role of '{component_type}' in the project.\n"
        f"- `godot_features` (string): Key Godot features, APIs, or patterns relevant to implementing '{component_type}'.\n"
        f"- `tasks` (list of MappingTask-like objects/dictionaries): A combined list of ALL `MappingTask` items from all processed individual '{component_type}' components."
    )

    return Task(
        name=f"CreateTaskGroup_{component_type}_{package_id}",
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
        # The output of this task should be a single object/dictionary
        # conforming to the TaskGroup Pydantic model.
        # output_json=True or output_pydantic=TaskGroup could be used.
    )
