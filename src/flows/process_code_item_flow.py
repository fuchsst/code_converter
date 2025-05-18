# src/flows/process_code_item_flow.py
from crewai import Crew, Process, Task
from crewai.tasks.task_output import TaskOutput
from crewai.flow.flow import Flow # Correct import for Flow base class

from typing import Dict, Any, Optional

from src.logger_setup import get_logger
import src.config as config
from src.core.context_manager import ContextManager
from src.core.state_manager import StateManager

# Models for this flow
from src.models.process_code_models import (
    ProcessCodeItemFlowInput,
    ProcessCodeItemFlowState, # May not be directly used by crewai.flow.Flow state, but good for reference
    ProcessCodeItemFlowOutput
)

# Agents for Step 5
from src.agents.step5.code_generator import get_code_generator_agent
from src.agents.step5.code_refiner import get_code_refinement_agent
# We might need a simple agent to wrap tool calls if tasks strictly require an agent
# For now, tools might be called by agents designed for generation/refinement or by the flow directly if possible.

# Tools
from src.tools.crewai_tools import (
    FileWriterTool,
    GodotProjectValidatorTool,
    GodotFileReaderTool
)
import json
import re

logger = get_logger(__name__)

class ProcessCodeItemFlow(Flow[ProcessCodeItemFlowState]): # Using ProcessCodeItemFlowState for structured state
    """
    Flow for processing a single task item from the Step 4 mapping.
    Handles code generation, file writing, validation, and refinement.
    """

    def __init__(
        self,
        flow_input: ProcessCodeItemFlowInput,
        state_manager: StateManager, # For logging or complex state interactions if needed beyond flow's state
        context_manager: ContextManager, # For providing broad context if needed by agents
        llm_instances: Dict[str, Any], # Dict of LLM instances for agents
    ):
        super().__init__() # Initialize Flow base class
        self.flow_input = flow_input
        self.state_manager = state_manager
        self.context_manager = context_manager
        self.llm_instances = llm_instances

        # Initialize state (using Pydantic model for structured state)
        # self.state is already an instance of ProcessCodeItemFlowState (with defaults)
        # after super().__init__() is called. We update its attributes here.
        self.state.task_id = self.flow_input.task_item_details.get('task_id', 'unknown_task_item')
        self.state.target_godot_file = self.flow_input.task_item_details.get('output_godot_file', 'unknown_file.gd')
        self.state.target_element = self.flow_input.task_item_details.get('target_element')
        # Other fields in ProcessCodeItemFlowState (like error_log, current_status)
        # will retain their defaults from the Pydantic model.
        
        logger.info(f"ProcessCodeItemFlow initialized for task_id: {self.state.task_id}, target: {self.state.target_godot_file}")

        # --- Instantiate Tools ---
        self.file_reader_tool = GodotFileReaderTool()
        self.file_writer_tool = FileWriterTool()
        self.project_validator_tool = GodotProjectValidatorTool()
        logger.debug("Instantiated tools for ProcessCodeItemFlow.")

        # --- Instantiate Agents ---
        # Ensure LLMs are provided
        generator_llm = self.llm_instances.get('GENERATOR_REFINER_MODEL')
        refiner_llm = self.llm_instances.get('GENERATOR_REFINER_MODEL') # Can be the same or different

        if not generator_llm or not refiner_llm:
            missing = [name for name, llm in [("GENERATOR_REFINER_MODEL", generator_llm), ("REFINER_MODEL (or GENERATOR_REFINER_MODEL)", refiner_llm)] if not llm]
            msg = f"Missing critical LLM instances for ProcessCodeItemFlow: {', '.join(missing)}"
            logger.error(msg)
            raise ValueError(msg)

        self.code_generator_agent = get_code_generator_agent(generator_llm, tools=[self.file_reader_tool])
        self.code_refiner_agent = get_code_refinement_agent(refiner_llm, tools=[self.file_reader_tool])
        logger.debug("Instantiated agents for ProcessCodeItemFlow.")

    def _extract_fenced_code(self, text: str) -> Optional[str]:
        """Extracts code from the first well-formed fenced code block."""
        match = re.search(r"```(?:[a-zA-Z0-9_]+)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_simple = re.search(r"```(.*?)```", text, re.DOTALL)
        if match_simple:
            return match_simple.group(1).strip()
        return None

    def run(self) -> ProcessCodeItemFlowOutput:
        """
        Executes the code processing flow for a single task item.
        Orchestrates code generation, file writing, validation, and refinement.
        Returns a ProcessCodeItemFlowOutput object.
        """
        logger.info(f"[{self.state.task_id}] Starting ProcessCodeItemFlow run for target: {self.state.target_godot_file}")

        # --- Step 1: Generate Code ---
        self.state.current_status = "generating_code"
        logger.info(f"[{self.state.task_id}] Step 1: Generating code.")
        code_gen_task_desc = (
            f"{self.flow_input.general_instructions or ''}\n\n"
            f"**Task: Generate FULL Godot Code for item '{self.state.task_id}'**\n"
            f"Target File: `{self.state.target_godot_file}`\n"
            f"Task Item Details (from mapping): ```json\n{json.dumps(self.flow_input.task_item_details, indent=2)}\n```\n"
            f"Context (C++ source, existing Godot code if any, project structure etc.):\n"
            f"--- START OF PROVIDED CONTEXT ---\n{self.flow_input.item_context_str}\n--- END OF PROVIDED CONTEXT ---\n"
            f"Your goal is to generate the **complete and final {config.TARGET_LANGUAGE} code content** for the file `{self.state.target_godot_file}`. "
            f"If the file `{self.state.target_godot_file}` already exists (check context or use File Reader tool to read its current content), "
            f"you MUST incorporate your changes into the existing content to produce the new full file content. "
            f"If it's a new file, generate the complete content from scratch. "
            f"Adhere to SOLID principles and Godot best practices. "
            f"Output ONLY the raw, complete code string for the entire file. Do not use markdown fences."
        )
        code_gen_task = Task(
            description=code_gen_task_desc,
            expected_output=f"The raw, complete {config.TARGET_LANGUAGE} code string for the entire file '{self.state.target_godot_file}'.",
            agent=self.code_generator_agent
        )
        
        gen_crew = Crew(agents=[self.code_generator_agent], tasks=[code_gen_task], process=Process.sequential, memory=False)
        try:
            gen_output: TaskOutput = gen_crew.kickoff()
            raw_gen_code = gen_output.raw_output if isinstance(gen_output, TaskOutput) else str(gen_output)
            
            extracted_code = self._extract_fenced_code(raw_gen_code)
            self.state.generated_code = extracted_code if extracted_code is not None else raw_gen_code.strip()

            if not self.state.generated_code:
                self.state.error_log.append("S1: Code generation resulted in an empty string.")
                raise ValueError("Code generation failed or produced empty output.")
            logger.info(f"[{self.state.task_id}] S1: Code generation successful (length: {len(self.state.generated_code)}).")
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S1: Code generation failed: {e}", exc_info=True)
            self.state.error_log.append(f"S1 Error: {e}")
            self.state.current_status = "failed"
            return self._prepare_output()

        # --- Step 2: Write Initial Code to File ---
        self.state.current_status = "writing_initial_file"
        logger.info(f"[{self.state.task_id}] Step 2: Writing initial code to {self.state.target_godot_file}.")
        try:
            write_result_str = self.file_writer_tool._run(
                file_path=self.state.target_godot_file, 
                content=self.state.generated_code
            )
            if "success" not in write_result_str.lower():
                self.state.error_log.append(f"S2: Initial file write failed. Tool Msg: {write_result_str}")
                raise ValueError(f"Initial file write failed: {write_result_str}")
            self.state.initial_write_successful = True
            logger.info(f"[{self.state.task_id}] S2: Initial file write successful.")
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S2: File write exception: {e}", exc_info=True)
            self.state.error_log.append(f"S2 Error: {e}")
            self.state.current_status = "failed"
            return self._prepare_output()

        # --- Step 3: Validate Project (Initial) ---
        self.state.current_status = "validating_initial"
        logger.info(f"[{self.state.task_id}] Step 3: Initial project validation.")
        try:
            validation_result_str = self.project_validator_tool._run(
                godot_project_path=self.flow_input.godot_project_path,
                target_file_path=self.state.target_godot_file 
            )
            if "Project validation successful" in validation_result_str:
                self.state.initial_validation_passed = True
                logger.info(f"[{self.state.task_id}] S3: Initial validation passed.")
                self.state.current_status = "completed"
                return self._prepare_output() 
            else:
                self.state.initial_validation_errors = validation_result_str
                self.state.error_log.append(f"S3: Initial validation failed. Details: {validation_result_str}")
                logger.warning(f"[{self.state.task_id}] S3: Initial validation failed. Errors: {validation_result_str}")
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S3: Validation exception: {e}", exc_info=True)
            self.state.error_log.append(f"S3 Error: {e}")
            self.state.initial_validation_errors = f"Validation tool exception: {e}"
            
        # --- Step 4: Refine Code (if initial validation failed) ---
        self.state.current_status = "refining_code"
        self.state.refinement_attempted = True
        logger.info(f"[{self.state.task_id}] Step 4: Refining code due to validation errors.")
        
        refinement_context = ( 
            f"{self.flow_input.general_instructions or ''}\n\n"
            f"**Original Task Item Details (for context):**\n```json\n{json.dumps(self.flow_input.task_item_details, indent=2)}\n```\n\n"
            f"**Validation Errors for `{self.state.target_godot_file}`:**\n```\n{self.state.initial_validation_errors}\n```\n\n"
            f"Your goal is to fix the validation errors in the file `{self.state.target_godot_file}`. "
            f"Use the 'File Reader' tool to read the current content of `{self.state.target_godot_file}`. "
            f"Analyze the errors and the file content, then generate a corrected version of the **entire file content**. "
            f"Output ONLY the raw, complete, corrected code string for the entire file. Do not use markdown fences."
        )
        refinement_task = Task(
            description=refinement_context,
            expected_output=f"The raw, complete, corrected {config.TARGET_LANGUAGE} code string for the entire file '{self.state.target_godot_file}'.",
            agent=self.code_refiner_agent
        )
        ref_crew = Crew(agents=[self.code_refiner_agent], tasks=[refinement_task], process=Process.sequential, memory=False)
        try:
            ref_output: TaskOutput = ref_crew.kickoff()
            raw_refined_code = ref_output.raw_output if isinstance(ref_output, TaskOutput) else str(ref_output)
            
            extracted_refined_code = self._extract_fenced_code(raw_refined_code)
            self.state.refined_code = extracted_refined_code if extracted_refined_code is not None else raw_refined_code.strip()

            if not self.state.refined_code:
                self.state.error_log.append("S4: Code refinement resulted in an empty string.")
                raise ValueError("Code refinement failed or produced empty output.")
            logger.info(f"[{self.state.task_id}] S4: Code refinement successful (length: {len(self.state.refined_code)}).")
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S4: Code refinement failed: {e}", exc_info=True)
            self.state.error_log.append(f"S4 Error: {e}")
            self.state.current_status = "failed" 
            return self._prepare_output()

        # --- Step 5: Write Refined Code to File ---
        self.state.current_status = "writing_refined_file"
        logger.info(f"[{self.state.task_id}] Step 5: Writing refined code to {self.state.target_godot_file}.")
        try:
            rewrite_result_str = self.file_writer_tool._run(
                file_path=self.state.target_godot_file,
                content=self.state.refined_code
            )
            if "success" not in rewrite_result_str.lower():
                self.state.error_log.append(f"S5: Refined file write failed. Tool Msg: {rewrite_result_str}")
                raise ValueError(f"Refined file write failed: {rewrite_result_str}")
            self.state.rewrite_successful = True
            logger.info(f"[{self.state.task_id}] S5: Refined file write successful.")
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S5: Refined file write exception: {e}", exc_info=True)
            self.state.error_log.append(f"S5 Error: {e}")
            self.state.current_status = "failed"
            return self._prepare_output()

        # --- Step 6: Re-Validate Project ---
        self.state.current_status = "validating_final"
        logger.info(f"[{self.state.task_id}] Step 6: Final project validation after refinement.")
        try:
            final_validation_result_str = self.project_validator_tool._run(
                godot_project_path=self.flow_input.godot_project_path,
                target_file_path=self.state.target_godot_file
            )
            if "Project validation successful" in final_validation_result_str:
                self.state.final_validation_passed = True
                logger.info(f"[{self.state.task_id}] S6: Final validation passed.")
                self.state.current_status = "completed"
            else:
                self.state.final_validation_errors = final_validation_result_str
                self.state.error_log.append(f"S6: Final validation failed. Details: {final_validation_result_str}")
                logger.warning(f"[{self.state.task_id}] S6: Final validation failed. Errors: {final_validation_result_str}")
                self.state.current_status = "failed" 
        except Exception as e:
            logger.error(f"[{self.state.task_id}] S6: Final validation exception: {e}", exc_info=True)
            self.state.error_log.append(f"S6 Error: {e}")
            self.state.final_validation_errors = f"Final validation tool exception: {e}"
            self.state.current_status = "failed"
            
        return self._prepare_output()

    def _prepare_output(self) -> ProcessCodeItemFlowOutput:
        """Consolidates state into the final ProcessCodeItemFlowOutput."""
        final_error_message = "; ".join(self.state.error_log) if self.state.error_log else None
        
        if self.state.current_status == "failed" and not final_error_message:
            final_error_message = "Processing failed for an unspecified reason during the flow."
        elif self.state.current_status == "completed" and final_error_message:
            logger.warning(f"[{self.state.task_id}] Status is 'completed' but errors were logged: {final_error_message}. Overriding status to 'failed'.")
            self.state.current_status = "failed"

        flow_output = ProcessCodeItemFlowOutput(
            task_id=self.state.task_id,
            status=self.state.current_status,
            target_godot_file=self.state.target_godot_file,
            target_element=self.state.target_element,
            error_message=final_error_message
        )
        logger.info(f"[{self.state.task_id}] ProcessCodeItemFlow finished with status: {flow_output.status}. Errors (if any): {flow_output.error_message}")
        return flow_output

    # Placeholder for the full sequence of tasks (generation, writing, validation, refinement)
    # This will replace the simplified 'run' method above.
    def run_full_sequence(self) -> ProcessCodeItemFlowOutput:
        # TODO: Implement the full multi-step logic here using a sequence of CrewAI tasks
        # 1. Generate Code Task
        # 2. Write File Task (using FileWriterTool, possibly via a simple agent/task)
        # 3. Validate Project Task (using GodotProjectValidatorTool)
        # 4. Conditional Refinement Task (if validation fails)
        # 5. Conditional Re-Write Task
        # 6. Conditional Re-Validate Task
        # Update self.state throughout the process.
        pass
