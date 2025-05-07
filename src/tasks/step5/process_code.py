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

# NOTE: This model might be simplified or deprecated if CodeGeneratorAgent's output changes.

class FileOutputParameters(BaseModel):
    """Represents the parameters needed for a file operation (write/replace)."""
    output_format: Literal['FULL_FILE', 'CODE_BLOCK'] = Field(..., description="Indicates if the code is for a full file or a replacement block.")
    search_block: Optional[str] = Field(None, description="The exact code block to search for if output_format is 'CODE_BLOCK'.")


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
