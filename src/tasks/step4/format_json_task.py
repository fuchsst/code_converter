# src/tasks/step4/format_json_task.py
from crewai import Task, Agent # Added Agent for type hinting
from typing import List, Dict, Any 
from src.logger_setup import get_logger
from src.models.mapping_models import MappingOutput, TaskGroup 

logger = get_logger(__name__)

def create_format_json_task( # Renamed from create_merge_task_groups_task
    agent: Agent, 
    strategy: str, 
    task_groups: List[Any], # Renamed from group_tasks_outputs to match MappingFlow call
    package_id: str
) -> Task:
    """
    Creates a task for assembling the final MappingOutput JSON object.
    
    Args:
        agent: The agent that will perform the task (JsonFormatterAgent).
        strategy: The conversion strategy string.
        task_groups: List of TaskGroup objects (or dicts representing them).
        package_id: Package ID.
        
    Returns:
        Task object for formatting the final JSON output.
    """
    logger.info(f"Creating FormatJsonTask for agent {agent.role} (Package: {package_id})")

    # Create a summary of task_groups for the description
    task_groups_description_placeholder = []
    if isinstance(task_groups, str): # If it's already a string placeholder from MappingFlow
        task_groups_description_placeholder.append(f"  - Placeholder for combined TaskGroups: {task_groups}")
    elif isinstance(task_groups, list):
        for i, group_output in enumerate(task_groups):
            title = "Unknown Group"
            if isinstance(group_output, TaskGroup) and hasattr(group_output, 'group_title'):
                title = group_output.group_title
            elif isinstance(group_output, dict) and 'group_title' in group_output:
                title = group_output['group_title']
            task_groups_description_placeholder.append(f"  - TaskGroup {i+1} (e.g., for '{title}')")
    
    group_outputs_context_for_description = "\n".join(task_groups_description_placeholder)
    if not group_outputs_context_for_description:
        group_outputs_context_for_description = "  - (No component task groups provided or placeholder used)"

    example_output_model = MappingOutput(
        package_id=package_id,
        mapping_strategy="Example: Convert C++ classes to Godot nodes, use signals for events.",
        task_groups=[
            TaskGroup(
                group_title="Sample Feature Group",
                feature_description="Implements a sample feature.",
                godot_features="CharacterBody3D, AnimationPlayer",
                tasks=[] 
            )
        ]
    )
    example_json_output = example_output_model.model_dump_json(indent=2)
    
    description = f"""
# Task: Create Final Mapping Output JSON for Package '{package_id}'

## Input Components:
You have been provided with:
1.  **Overall Conversion Strategy:**
    ```
    {strategy} 
    ```
    (Note: If this is a placeholder like '{{task_name.output}}', the actual strategy string will be injected at runtime.)

2.  **Combined Task Groups:** A list of structured TaskGroup objects.
    (Note: If this is a placeholder like '{{task_name.output}}', the actual list of TaskGroups will be injected at runtime. Content summarized here:)
{group_outputs_context_for_description}

## Your Task:
Your sole responsibility is to construct a single, valid JSON object that strictly conforms to the `MappingOutput` Pydantic schema.

You must:
1.  Set the `package_id` field to: "{package_id}"
2.  Use the provided "Overall Conversion Strategy" (which will be a string at runtime) as the value for the `mapping_strategy` field.
3.  Use the provided "Combined Task Groups" (which will be a list of TaskGroup objects/dictionaries at runtime) as the value for the `task_groups` field.

**CRITICAL:** Your final output MUST be ONLY the raw JSON object string. Do not include any introductory text, explanations, comments, or markdown formatting (like ```json) before or after the JSON object. The entire output must be parsable as a single JSON entity.
"""

    return Task(
        name=f"FormatFinalMappingJson_{package_id}", 
        description=description,
        expected_output=(
            f"A single, valid JSON object string that strictly conforms to the `MappingOutput` Pydantic model. "
            f"This JSON object must contain the `package_id` ('{package_id}'), the complete `mapping_strategy` string, "
            f"and a `task_groups` array containing all the TaskGroup objects/dictionaries received as input."
            f"\n\nExample of the required raw JSON output format:\n{example_json_output}"
        ),
        agent=agent,
        output_pydantic=MappingOutput, 
        async_execution=False
    )
