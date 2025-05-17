# src/models/process_code_models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

class ProcessCodeItemFlowInput(BaseModel):
    """Input for the ProcessCodeItemFlow."""
    package_id: str
    task_item_details: Dict[str, Any] # From Step 4 mapping
    item_context_str: str # Pre-built context for this item
    godot_project_path: str
    general_instructions: Optional[str] = None

class ProcessCodeItemFlowState(BaseModel):
    """Internal state tracking for the ProcessCodeItemFlow."""
    task_id: str = "default_task_id" # Added default
    target_godot_file: str = "default_target_file.gd" # Added default
    target_element: Optional[str] = None
    
    generated_code: Optional[str] = None
    initial_write_successful: bool = False
    initial_validation_passed: bool = False
    initial_validation_errors: Optional[str] = None
    
    refined_code: Optional[str] = None
    refinement_attempted: bool = False
    rewrite_successful: bool = False
    final_validation_passed: bool = False
    final_validation_errors: Optional[str] = None
    
    current_status: Literal["pending", "generating_code", "writing_initial_file", "validating_initial", 
                            "refining_code", "writing_refined_file", "validating_final", 
                            "completed", "failed"] = "pending"
    error_log: list[str] = Field(default_factory=list)

class ProcessCodeItemFlowOutput(BaseModel):
    """
    Output of the ProcessCodeItemFlow, compatible with TaskItemProcessingResult.
    """
    task_id: str = Field(..., description="The ID of the original task item.")
    status: str = Field(..., description="Final status ('completed', 'failed').")
    target_godot_file: Optional[str] = Field(None, description="Target file path.")
    target_element: Optional[str] = Field(None, description="Target element.")
    error_message: Optional[str] = Field(None, description="Consolidated error message if failed.")
    # Potentially add more detailed output if needed for Step 5's package-level analysis
    # For now, aligns with TaskItemProcessingResult
