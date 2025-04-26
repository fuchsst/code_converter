# src/tasks/process_code.py
import json
from crewai import Task, Agent
from src.logger_setup import get_logger
import src.config as config
from typing import List, Dict, Any, Optional
from pydantic.v1 import BaseModel, Field # Use Pydantic v1 if needed for Task context/output

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


# --- Task Definitions for Hierarchical Process ---

class HierarchicalProcessTaskItem:
    """
    CrewAI Task definition for the MANAGER agent to orchestrate the processing
    of a single C++ to Godot conversion task item within a hierarchical crew.
    """
    def create_task(self,
                    manager_agent: Agent, # Task assigned to the manager
                    task_item_details: Dict[str, Any], # Details of the specific item
                    package_context: str, # General context for the package
                    dependent_tasks: Optional[List[Task]] = None) -> Task: # For potential dependencies *between* items
        """
        Creates the CrewAI Task for processing a single conversion task item.

        Args:
            manager_agent (Agent): The Manager agent overseeing the hierarchical process.
            task_item_details (Dict[str, Any]): The dictionary representing the single task item
                                                (e.g., from MappingTask Pydantic model).
            package_context (str): Shared context relevant to the entire package (e.g., C++ code).
                                   Specific code snippets might be included here or passed via inputs.
            dependent_tasks (Optional[List[Task]]): Tasks for previous items in the package, if needed for context.

        Returns:
            Task: The CrewAI Task object assigned to the manager.
        """
        task_id = task_item_details.get('task_id', task_item_details.get('task_title', 'unknown_task'))
        target_file = task_item_details.get('output_godot_file', 'unknown_file')
        logger.info(f"Creating HierarchicalProcessTaskItem for Manager: Task ID '{task_id}', Target '{target_file}'")

        # Construct description guiding the MANAGER
        description = (
            f"**Orchestrate the processing of Task Item '{task_id}' targeting '{target_file}'.**\n\n"
            f"**Task Item Details:**\n```json\n{json.dumps(task_item_details, indent=2)}\n```\n\n"
            f"**Your Orchestration Steps (Delegate to appropriate agents):**\n"
            f"1.  **Generate Code:** Delegate to the 'CodeGeneratorAgent'. Provide it with the task item details and relevant package context ({package_context}). The agent should return only the generated {config.TARGET_LANGUAGE} code string.\n"
            f"2.  **Determine Output Format & Search Block:** Analyze the generated code and task item details. Decide if the output is 'FULL_FILE' or 'CODE_BLOCK'. If 'CODE_BLOCK', extract the exact original code block (`search_block`) from the existing Godot code (found within the package context or potentially read via a tool if necessary) that the generated code should replace. *Self-correction: The CodeGeneratorAgent might be able to provide this info directly based on its input task, simplifying this step.*\n"
            f"3.  **Validate Syntax:** Delegate to the 'SyntaxValidationAgent'. Provide the generated code. The agent will use the 'Godot Syntax Validator' tool and return a success message or error details.\n"
            f"4.  **Refine Code (If Validation Failed):** If step 3 reported errors, delegate to the 'CodeRefinementAgent'. Provide the original generated code and the validation errors. The agent should return the corrected code or indicate failure.\n"
            f"5.  **Re-Validate (If Refined):** If code was refined in step 4, delegate to the 'SyntaxValidationAgent' again with the *refined* code.\n"
            f"6.  **File Operation:** If syntax validation is successful (either initially or after refinement), delegate to the 'FileManagerAgent'. Provide:\n"
            f"    - The final, validated {config.TARGET_LANGUAGE} code.\n"
            f"    - The target file path: `{target_file}`.\n"
            f"    - The determined `output_format` ('FULL_FILE' or 'CODE_BLOCK').\n"
            f"    - The `search_block` (if `output_format` is 'CODE_BLOCK').\n"
            f"    Use the 'File Writer' tool for 'FULL_FILE' or the 'File Content Replacer' tool for 'CODE_BLOCK'. The agent will return the tool's execution status message.\n"
            f"7.  **Consolidate Result:** Compile a final JSON result for this task item, including `task_id`, final `status` ('completed' or 'failed'), `target_godot_file`, `target_element`, and any relevant `error_message` from failed steps (generation, validation, refinement, file op)."
        )

        return Task(
            description=description,
            expected_output=(
                "A JSON object summarizing the final outcome of processing this single task item, conforming to the TaskItemProcessingResult structure. "
                "This includes 'task_id', 'status' ('completed' or 'failed'), 'target_godot_file', 'target_element', and 'error_message' if applicable."
            ),
            agent=manager_agent, # This task is assigned to the manager
            context=dependent_tasks, # Pass outputs of previous tasks if needed
            # output_json=True # Expecting JSON output
            # output_pydantic=TaskItemProcessingResult # Validate output structure
            # async_execution=True # Consider for parallel processing of items if manager supports it well
        )


class AnalyzePackageFailuresTask:
    """
    CrewAI Task definition for the RemappingAdvisorAgent to analyze failures
    for an entire package and provide remapping advice. Runs after all items
    in a package have been processed.
    """
    def create_task(self,
                    advisor_agent: Agent,
                    all_item_results_context: List[Task]) -> Task: # Context comes from previous item tasks
        """
        Creates the CrewAI Task for analyzing package failures.

        Args:
            advisor_agent (Agent): The RemappingAdvisorAgent instance.
            all_item_results_context (List[Task]): A list containing the Task objects for all
                                                   processed items in the package. Their outputs
                                                   (TaskItemProcessingResult JSON) will form the context.

        Returns:
            Task: The CrewAI Task object assigned to the RemappingAdvisorAgent.
        """
        logger.info(f"Creating AnalyzePackageFailuresTask for agent: {advisor_agent.role}")

        description = (
            "Analyze the results of all processed task items for this work package, provided in the context. "
            "The context contains a list of JSON outputs (TaskItemProcessingResult) from the processing of each individual task item.\n\n"
            "1.  Identify all task items with a final status of 'failed'.\n"
            "2.  Extract the relevant details for these failed tasks (task_id, error messages, etc.).\n"
            "3.  Use the 'Remapping Logic Analyzer' tool, passing the extracted list of failed task details to it.\n"
            "4.  Based *only* on the output received from the 'Remapping Logic Analyzer' tool, formulate your final response.\n"
            "5.  Structure your final output as a JSON object conforming to the RemappingAdvice model, containing `recommend_remapping` (boolean), `reason` (string), and `feedback` (string)."
        )

        return Task(
            description=description,
            expected_output=(
                "A JSON object conforming to the RemappingAdvice model, indicating whether remapping is recommended, "
                "the reason, and detailed feedback if applicable."
            ),
            agent=advisor_agent,
            context=all_item_results_context, # Depends on all previous item processing tasks
            output_pydantic=RemappingAdvice # Validate output structure
        )
