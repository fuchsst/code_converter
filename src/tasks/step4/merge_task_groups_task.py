# src/tasks/merge_task_groups_task.py
from crewai import Task
from typing import List, Dict, Any # Added Dict, Any for better type hinting if strategy/group_tasks_outputs are complex
from src.logger_setup import get_logger
from src.models.mapping_models import MappingOutput, TaskGroup # Assuming MappingTask is not directly used here but by TaskGroup

logger = get_logger(__name__)

def create_merge_task_groups_task(agent, strategy: str, group_tasks_outputs: List[Any], package_id: str) -> Task:
    """
    Creates a task for merging multiple TaskGroups and formatting the final JSON output.
    
    Args:
        agent: The agent that will perform the task.
        strategy: The conversion strategy string.
        group_tasks_outputs: List of outputs from component group tasks. Each item in this list
                             is expected to be a TaskGroup object (or a dict representing it).
        package_id: Package ID.
        
    Returns:
        Task object for merging task groups and creating final output.
    """
    # Serialize the actual task groups data to include in the description
    # This ensures the JSON formatter agent has the complete data to work with
    import json
    
    # Ensure the task groups are serializable
    serializable_task_groups = []
    for group in group_tasks_outputs:
        if isinstance(group, TaskGroup):
            serializable_task_groups.append(group.model_dump())
        elif isinstance(group, dict):
            serializable_task_groups.append(group)
        else:
            # Try to convert to dict if it has a __dict__ attribute
            try:
                serializable_task_groups.append(vars(group))
            except:
                logger.warning(f"Could not serialize task group of type {type(group)}. Using str representation.")
                serializable_task_groups.append(str(group))
    
    # Serialize the task groups to JSON
    try:
        task_groups_json = json.dumps(serializable_task_groups, indent=2)
    except Exception as e:
        logger.error(f"Error serializing task groups to JSON: {e}")
        # Fallback to a simple representation
        task_groups_json = str(group_tasks_outputs)
    
    # Create a summary for logging purposes
    group_outputs_description_placeholder = []
    for i, group in enumerate(serializable_task_groups):
        title = group.get('group_title', f"Group {i+1}")
        task_count = len(group.get('tasks', []))
        group_outputs_description_placeholder.append(f"  - TaskGroup {i+1}: '{title}' with {task_count} tasks")
    
    group_outputs_summary = "\n".join(group_outputs_description_placeholder)
    if not group_outputs_summary:
        group_outputs_summary = "  - (No component groups provided)"
    
    logger.info(f"Merging {len(serializable_task_groups)} task groups:\n{group_outputs_summary}")

    # Create an example of the required JSON structure for the expected_output field
    example_output_model = MappingOutput(
        package_id=package_id,
        mapping_strategy="Example: Convert C++ classes to Godot nodes, use signals for events.",
        task_groups=[ # Example with one TaskGroup for brevity
            TaskGroup(
                group_title="Sample Feature Group",
                feature_description="Implements a sample feature.",
                godot_features="CharacterBody3D, AnimationPlayer",
                tasks=[] # Empty tasks for brevity in example
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

2.  **Component Task Groups:** A list of structured TaskGroup objects, one for each relevant component type (scripts, resources, scenes, etc.). These have already been generated.
    ```json
{task_groups_json}
    ```

## Your Task:
Your primary responsibility is to assemble these components into a single, valid JSON object that strictly conforms to the `MappingOutput` schema.

Specifically, you need to:
1.  Set the `package_id` field to: "{package_id}"
2.  Use the provided "Overall Conversion Strategy" string as the value for the `mapping_strategy` field.
3.  Combine all the individual `TaskGroup` objects (from the "Component Task Groups" input) into a single list and assign it to the `task_groups` field.

**CRITICAL:** Your final output MUST be ONLY the raw JSON object string. Do not include any explanatory text, comments, or markdown formatting (like ```json) before or after the JSON object. The entire output must be parsable as a single JSON entity.
"""

    return Task(
        name="merge_and_format_mapping_output_json", # More descriptive name
        description=description,
        expected_output=(
            f"A single, valid JSON object string that strictly conforms to the `MappingOutput` Pydantic model. "
            f"This JSON object must contain the `package_id` ('{package_id}'), the complete `mapping_strategy` string, "
            f"and a `task_groups` array containing all the TaskGroup objects received as input."
            f"\n\nExample of the required raw JSON output format:\n{example_json_output}"
        ),
        agent=agent,
        # The context for this task will be the 'strategy' and 'group_tasks_outputs'
        # These should be passed to the agent when it executes this task.
        # CrewAI tasks can receive context through their description (as done here with placeholders if needed)
        # or more robustly through agent's memory or direct input when the task is executed.
        # The MappingFlow will need to ensure these are available.
        output_pydantic=MappingOutput, # Specify the Pydantic model for output validation
        async_execution=False
    )
