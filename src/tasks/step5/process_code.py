# src/tasks/process_code.py
import json
from crewai import Task, Agent
from src.logger_setup import get_logger
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import Optional

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


def create_analyze_package_failures_task(
                    advisor_agent: Agent,
                    item_processing_results: List[Dict[str, Any]],
                    instructions: Optional[str] = None) -> Task:
    """
    Creates the CrewAI Task for analyzing package failures.

    Args:
        advisor_agent (Agent): The RemappingAdvisorAgent instance.
        item_processing_results (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                        is a TaskItemProcessingResult.
        instructions (Optional[str]): General instructions to prepend to the task description.

    Returns:
        Task: The CrewAI Task object assigned to the RemappingAdvisorAgent.
    """
    logger.info(f"Creating AnalyzePackageFailuresTask for agent: {advisor_agent.role} with {len(item_processing_results)} item results.")

    # Prepare the full description, prepending instructions if available
    full_description = ""
    if instructions:
        full_description += f"**General Instructions:**\n{instructions}\n\n---\n\n"

    # Embed the item results directly into the description for the agent to parse
    # The agent's goal will need to be updated to expect this format.
    results_json_str = json.dumps(item_processing_results, indent=2)
    full_description += (
        f"Analyze the following results of all processed task items for this work package:\n\n"
        f"**Processed Task Item Results:**\n```json\n{results_json_str}\n```\n\n"
        f"**Your Analysis Steps:**\n"
        f"1.  From the 'Processed Task Item Results' above, identify all task items with a final status of 'failed'.\n"
        f"2.  Extract the relevant details for these failed tasks (task_id, error_message, target_godot_file, target_element etc.).\n"
        f"3.  Use your 'Remapping Logic Analyzer' tool. Pass the extracted list of failed task details (as a list of dictionaries) to this tool.\n"
        f"4.  Based *only* on the output received from the 'Remapping Logic Analyzer' tool, formulate your final response.\n"
        f"5.  Structure your final output as a JSON object conforming to the RemappingAdvice model, containing `recommend_remapping` (boolean), `reason` (string), and `feedback` (string)."
    )

    return Task(
        description=full_description,
        expected_output=(
            "A JSON object conforming to the RemappingAdvice model, indicating whether remapping is recommended, "
            "the reason, and detailed feedback if applicable."
        ),
        agent=advisor_agent,
        output_pydantic=RemappingAdvice
    )
