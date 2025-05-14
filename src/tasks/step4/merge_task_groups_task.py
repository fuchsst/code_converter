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
    # Create a combined context of all group outputs for the task description
    # The agent will receive these as structured data, not just placeholders in description.
    # However, for the description, a summary or placeholder is fine.
    group_outputs_description_placeholder = []
    for i, group_output in enumerate(group_tasks_outputs):
        # Try to get a title if it's a TaskGroup or dict, otherwise just use index
        title = "Unknown Group"
        if isinstance(group_output, TaskGroup) and hasattr(group_output, 'group_title'):
            title = group_output.group_title
        elif isinstance(group_output, dict) and 'group_title' in group_output:
            title = group_output['group_title']
        group_outputs_description_placeholder.append(f"  - TaskGroup {i+1} (e.g., for '{title}')")
    
    group_outputs_context_for_description = "\n".join(group_outputs_description_placeholder)
    if not group_outputs_context_for_description:
        group_outputs_context_for_description = "  - (No component groups provided)"

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
    (Content of these groups will be provided in the task's context/memory, summarized here:)
{group_outputs_context_for_description}

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
