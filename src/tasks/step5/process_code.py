# src/tasks/process_code.py
import json
from crewai import Task, Agent
from src.logger_setup import get_logger
import src.config as config
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import Optional, Literal

logger = get_logger(__name__)

# --- Pydantic Models for Task Context/Output (Optional but Recommended) ---
# Define Pydantic models for structured data passing between tasks if needed.
# For example, the output of the file operation task.

class TaskItemProcessingResult(BaseModel):
    """Represents the final outcome of processing a single task item."""
    task_id: str = Field(..., description="The ID of the original task item.")
    status: str = Field(..., description="Final status ('completed', 'failed').")
    target_godot_file: Optional[str] = Field(description="Target file path.")
    target_element: Optional[str] = Field(description="Target element.")
    error_message: Optional[str] = Field(description="Consolidated error message if failed.")
    # Add other relevant fields like validation status, file op status if needed for analysis

class RemappingAdvice(BaseModel):
    """Represents the output of the remapping analysis task."""
    recommend_remapping: bool = Field(..., description="True if remapping is recommended, False otherwise.")
    reason: Optional[str] = Field(description="Explanation for the recommendation.")
    feedback: Optional[str] = Field(description="Detailed feedback for Step 4 if remapping is recommended.")

class CodeGenerationResult(BaseModel):
    """Represents the structured output from the CodeGeneratorAgent."""
    generated_code: str = Field(..., description="The generated Godot code string.")
    output_format: Literal['FULL_FILE', 'CODE_BLOCK'] = Field(..., description="Indicates if the code is for a full file or a replacement block.")
    search_block: Optional[str] = Field(None, description="The exact code block to search for if output_format is 'CODE_BLOCK'.")


# --- Task Definitions for Hierarchical Process ---

def create_hierarchical_process_taskitem_task(
                    manager_agent: Agent, # Task assigned to the manager
                    task_item_details: Dict[str, Any], # Details of the specific item
                    package_context: str, # General context for the package
                    instructions: Optional[str] = None, # General instructions
                    dependent_tasks: Optional[List[Task]] = None) -> Task: # For potential dependencies *between* items
    """
    Creates the CrewAI Task for processing a single conversion task item.

    Args:
        manager_agent (Agent): The Manager agent overseeing the hierarchical process.
        task_item_details (Dict[str, Any]): The dictionary representing the single task item
                                            (e.g., from MappingTask Pydantic model).
        package_context (str): Shared context relevant to the entire package (e.g., C++ code).
                                Specific code snippets might be included here or passed via inputs.
        instructions (Optional[str]): General instructions to prepend to the task description.
        dependent_tasks (Optional[List[Task]]): Tasks for previous items in the package, if needed for context.

    Returns:
        Task: The CrewAI Task object assigned to the manager.
    """
    task_id = task_item_details.get('task_id', task_item_details.get('task_title', 'unknown_task'))
    target_file = task_item_details.get('output_godot_file', 'unknown_file')
    logger.info(f"Creating HierarchicalProcessTaskItem for Manager: Task ID '{task_id}', Target '{target_file}'")

    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions:**\n{instructions}\n\n---\n\n"

    # Construct description guiding the MANAGER
    full_description += (
        f"**Orchestrate the processing of Task Item '{task_id}' targeting '{target_file}'.**\n\n"
        f"**Task Item Details:**\n```json\n{json.dumps(task_item_details, indent=2)}\n```\n\n"
        f"**Package Context:**\n"
        "--- START OF PROVIDED CONTEXT ---\n"
        f"{package_context}\n"
        "--- END OF PROVIDED CONTEXT ---\n\n"
        f"**Your Orchestration Steps (Delegate to appropriate agents):**\n"
        f"1.  **Generate Code & Determine Format:** Delegate to the agent with role 'Expert C++ to {config.TARGET_LANGUAGE} Translator & Formatter'. Provide task details and context. Expect a JSON object string conforming to `CodeGenerationResult`. Parse this result to get `generated_code`, `output_format`, and `search_block`.\n"
        f"2.  **Initial File Operation:** Delegate to the agent with role 'File System Operations Specialist'. Provide these exact parameters:\n"
        f"    - `file_path`: The target file path `{target_file}`.\n"
        f"    - `content`: The `generated_code` from Step 1.\n"
        f"    - `output_format`: The `output_format` from Step 1.\n"
        f"    - `search_block`: The `search_block` from Step 1 (pass `None` or omit if not applicable).\n"
        f"    Determine the correct tool: if `output_format` is 'FULL_FILE', the tool name is 'File Writer'; if `output_format` is 'CODE_BLOCK', the tool name is 'File Content Replacer'.\n"
        f"    Delegate to the agent with role 'File System Operations Specialist', instructing it to use the determined tool name and providing the necessary parameters (`file_path`, `content`, `diff` if using 'File Content Replacer'). Record the status message returned by the agent.\n"
        f"3.  **Validate Project:** If Step 2 succeeded, delegate to the agent with role 'Godot Project Post-Modification Validator'. Provide these exact parameters:\n"
        f"    - `godot_project_path`: The absolute Godot project path (extracted from context).\n"
        f"    - `target_file_path`: The target file path `{target_file}`.\n"
        f"    This agent uses the 'Godot Project Validator' tool. Record the result (success or filtered errors).\n"
        f"4.  **Refine Code (If Validation Failed):** If Step 3 reported errors, delegate to the agent with role '{config.TARGET_LANGUAGE} Code Refinement Specialist (Project Context)'. Provide these exact parameters:\n"
        f"    - `target_file_path`: The target file path `{target_file}`.\n"
        f"    - `validation_errors`: The relevant validation errors reported in Step 3.\n"
        f"    This agent uses 'FileReaderTool' to get current content, refines it, and returns the *corrected code string*.\n"
        f"5.  **Re-Apply Refined Code (If Refined):** If Step 4 produced refined code, delegate to the agent with role 'File System Operations Specialist' again. Provide these exact parameters:\n"
        f"    - `file_path`: The target file path `{target_file}`.\n"
        f"    - `content`: The *refined code string* (full file content) from Step 4.\n"
        f"    Instruct the agent to use the tool named **'File Writer'** and provide the `file_path` and `content` parameters to it.\n"
        f"    Record the status message returned by the agent.\n"
        f"6.  **Re-Validate Project (If Refined):** If Step 5 was performed and succeeded, delegate to the agent with role 'Godot Project Post-Modification Validator' again, providing the same inputs as Step 3 (`godot_project_path` and `target_file_path`).\n"
        f"7.  **Consolidate Result:** Compile the final `TaskItemProcessingResult` JSON. Include `task_id`, `target_godot_file`, `target_element`. Set `status` to 'completed' only if the relevant validation step (Step 3 or Step 6) passed and the corresponding file operation (Step 2 or Step 5) succeeded. Otherwise, set `status` to 'failed' and include relevant error messages from any failed step (generation, file ops, validation, refinement)."
    )

    return Task(
        description=full_description,
        expected_output=(
            "A JSON object summarizing the final outcome of processing this single task item, conforming to the TaskItemProcessingResult structure. "
            "This includes 'task_id', 'status' ('completed' or 'failed'), 'target_godot_file', 'target_element', and 'error_message' if applicable."
        ),
        agent=manager_agent, # This task is assigned to the manager
        context=dependent_tasks, # Pass outputs of previous tasks if needed
        output_pydantic=TaskItemProcessingResult
        # async_execution=True # Consider for parallel processing of items if manager supports it well
    )


def create_analyze_package_failures_task(
                    advisor_agent: Agent,
                    all_item_results_context: List[Task], # Context comes from previous item tasks
                    instructions: Optional[str] = None) -> Task:
    """
    Creates the CrewAI Task for analyzing package failures.

    Args:
        advisor_agent (Agent): The RemappingAdvisorAgent instance.
        all_item_results_context (List[Task]): A list containing the Task objects for all
                                               processed items in the package. Their outputs
                                               (TaskItemProcessingResult JSON) will form the context.
        instructions (Optional[str]): General instructions to prepend to the task description.

    Returns:
        Task: The CrewAI Task object assigned to the RemappingAdvisorAgent.
    """
    logger.info(f"Creating AnalyzePackageFailuresTask for agent: {advisor_agent.role}")

    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions:**\n{instructions}\n\n---\n\n"

    full_description += (
        "Analyze the results of all processed task items for this work package, provided in the context. "
        "The context contains a list of JSON outputs (TaskItemProcessingResult) from the processing of each individual task item.\n\n"
        "1.  Identify all task items with a final status of 'failed'.\n"
        "2.  Extract the relevant details for these failed tasks (task_id, error messages, etc.).\n"
        "3.  Use the 'Remapping Logic Analyzer' tool, passing the extracted list of failed task details to it.\n"
        "4.  Based *only* on the output received from the 'Remapping Logic Analyzer' tool, formulate your final response.\n"
        "5.  Structure your final output as a JSON object conforming to the RemappingAdvice model, containing `recommend_remapping` (boolean), `reason` (string), and `feedback` (string)."
    )

    return Task(
        description=full_description,
        expected_output=(
            "A JSON object conforming to the RemappingAdvice model, indicating whether remapping is recommended, "
            "the reason, and detailed feedback if applicable."
        ),
        agent=advisor_agent,
        context=all_item_results_context,
        output_pydantic=RemappingAdvice
    )
