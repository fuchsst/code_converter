# src/core/executors/step3_structure_definer.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, read_file_content # Added read_file_content
from src.agents.structure_definer import StructureDefinerAgent
from src.tasks.define_structure import DefineStructureTask, GodotStructureOutput # Added Pydantic model
from crewai import Crew, Process
from crewai import LLM as CrewAI_LLM # Alias default LLM
from src.llms.google_genai_llm import GoogleGenAI_LLM
from src.logger_setup import get_logger
from src.utils.json_utils import parse_json_from_string # Added JSON parser utility
import src.config as global_config

logger = get_logger(__name__)

class Step3Executor(StepExecutor):
    """Executes Step 3: Godot Structure Definition."""

    def execute(self, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        """
        Runs the Godot structure definition for specified or all eligible packages.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process.
                                                If None, processes all eligible packages.
            force (bool): If True, forces reprocessing of packages even if already defined or failed.
            **kwargs: Additional arguments (e.g., instruction_dir).

        Returns:
            bool: True if the structure definition was successful for the processed packages in this run, False otherwise.
        """
        logger.info(f"--- Starting Step 3 Execution: Define Structure (Requested: {package_ids or 'All Eligible'}, Force: {force}) ---")

        # --- Fetch and Validate Processing Order ---
        processing_order = self.state_manager.get_package_processing_order()
        all_packages = self.state_manager.get_all_packages()

        if not all_packages:
            logger.warning("No packages found in state. Cannot proceed with Step 3.")
            self.state_manager.update_workflow_status('step3_complete')
            return True

        if processing_order is None:
            logger.error("Critical: Package processing order is missing from state. Cannot ensure correct execution order.")
            self.state_manager.update_workflow_status('failed_step3', "Processing order missing.")
            return False
        if not isinstance(processing_order, list):
            logger.error(f"Critical: Package processing order is not a list (type: {type(processing_order)}). Invalid state.")
            self.state_manager.update_workflow_status('failed_step3', "Invalid processing order format.")
            return False

        # --- Identify Eligible Packages for This Run ---
        packages_to_process_this_run = []
        potential_target_package_ids = set()
        target_status = 'identified' # Step 3 processes packages identified in Step 2
        failed_status_prefix = 'failed_structure' # Status prefix indicating failure in *this* step
        completed_status = 'structure_defined' # Status indicating success in *this* step

        # First pass: identify all packages that could potentially be processed by this step
        for pkg_id, pkg_data in all_packages.items():
            current_status = pkg_data.get('status')
            is_target = (current_status == target_status)
            is_failed_this_step = (current_status and current_status.startswith(failed_status_prefix))
            is_already_completed = (current_status == completed_status)

            # Check if it matches specific request if provided
            matches_specific_request = (not package_ids or pkg_id in package_ids)

            if matches_specific_request:
                if is_target:
                    potential_target_package_ids.add(pkg_id)
                elif force and (is_failed_this_step or is_already_completed):
                    logger.info(f"Force=True: Package '{pkg_id}' (status: {current_status}) will be re-processed for Step 3.")
                    potential_target_package_ids.add(pkg_id)

        # Second pass: build the list for *this run* based on order and current status/force flag
        processed_in_order = set()
        for pkg_id in processing_order:
            if pkg_id not in all_packages:
                logger.error(f"Critical: Package '{pkg_id}' from processing order not found in state's work_packages. Inconsistent state.")
                self.state_manager.update_workflow_status('failed_step3', f"Inconsistency: Package {pkg_id} in order but not in state.")
                return False

            processed_in_order.add(pkg_id) # Track packages covered by the order

            # Check if this package is among the potential targets for this run
            if pkg_id in potential_target_package_ids:
                pkg_data = all_packages[pkg_id]
                current_status = pkg_data.get('status')
                is_target = (current_status == target_status)
                is_failed_this_step = (current_status and current_status.startswith(failed_status_prefix))
                is_already_completed = (current_status == completed_status)

                if is_target:
                    packages_to_process_this_run.append(pkg_id)
                elif force and (is_failed_this_step or is_already_completed):
                    # Reset status to target status before processing if forced
                    self.state_manager.update_package_state(pkg_id, target_status, error=None) # Clear previous error/status
                    packages_to_process_this_run.append(pkg_id)
                # else: package is potential target but not in correct state for this run (e.g., already completed without force)

        # Final consistency check: ensure all potential target packages were covered by the processing order
        missing_from_order = potential_target_package_ids - processed_in_order
        if missing_from_order:
            logger.error(f"Critical: Potential target packages are missing from the processing order: {missing_from_order}. Inconsistent state.")
            self.state_manager.update_workflow_status('failed_step3', f"Inconsistency: Potential targets missing from order: {missing_from_order}")
            return False

        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 3 run (might be already processed, failed without force, or filtered).")
            # Check if the overall step should be marked complete based on *potential* targets
            all_potential_targets_done_or_failed = True
            final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
            for pkg_id in potential_target_package_ids:
                status = final_packages_state.get(pkg_id, {}).get('status')
                # Consider completed or any failed state *for this step* as terminal for this check
                if not (status == completed_status or (status and status.startswith(failed_status_prefix))):
                    all_potential_targets_done_or_failed = False
                    logger.debug(f"Package {pkg_id} is still pending Step 3 completion (status: {status}).")
                    break
            if all_potential_targets_done_or_failed and potential_target_package_ids: # Only complete if there were potential targets
                logger.info("All potential target packages for Step 3 are now processed or failed.")
                # Avoid overwriting a global failed state if some packages failed earlier
                current_global_status = self.state_manager.get_state().get('workflow_status')
                if not (current_global_status and 'failed' in current_global_status):
                    self.state_manager.update_workflow_status('step3_complete')
            # Even if nothing to process now, the run itself didn't fail.
            return True # Indicate this specific invocation had nothing to fail on

        logger.info(f"Packages to process in this Step 3 run (in order, Force={force}): {packages_to_process_this_run}")
        # Set running status only if we are actually processing packages
        self.state_manager.update_workflow_status('running_step3')
        overall_success_this_run = True # Tracks success *of this specific run*
        analysis_dir = Path(self.config.get("ANALYSIS_OUTPUT_DIR", "analysis_output")).resolve() # Use Path

        # --- Get the LLM config and instantiate the LLM object ---
        mapper_llm_config = self._get_llm_config('mapper')
        mapper_llm_instance = None
        if mapper_llm_config:
            model_identifier = mapper_llm_config.get("model", "")
            try:
                if model_identifier.startswith("gemini/"):
                    # Use GoogleGenAI_LLM for Gemini, request JSON output via schema
                    mapper_llm_config['timeout'] = global_config.GEMINI_TIMEOUT # Add timeout
                    mapper_llm_config['response_schema'] = GodotStructureOutput # Pass the Pydantic model
                    mapper_llm_config['response_mime_type'] = "application/json" # Set mime type
                    mapper_llm_instance = GoogleGenAI_LLM(**mapper_llm_config)
                    logger.info(f"Instantiated custom GoogleGenAI_LLM for role 'mapper': {model_identifier} with timeout {global_config.GEMINI_TIMEOUT}s, JSON output, and GodotStructureOutput schema")
                else:
                    # Use default CrewAI LLM for other models
                    # Filter config if necessary for CrewAI_LLM
                    mapper_llm_instance = CrewAI_LLM(**mapper_llm_config)
                    logger.info(f"Instantiated default crewai.LLM for role 'mapper': {model_identifier}")
            except Exception as e:
                logger.error(f"Failed to instantiate LLM for role 'mapper' ({model_identifier}): {e}", exc_info=True)
                mapper_llm_instance = None # Ensure it's None on error
        else:
            logger.error("Mapper LLM configuration ('mapper') not found.")

        if not mapper_llm_instance:
             logger.error("Mapper LLM instance could not be created. Cannot execute Step 3.")
             self.state_manager.update_workflow_status('failed_step3', "Mapper LLM not configured or failed to instantiate.")
             return False # Return False as this run failed

        # --- Load Instruction Context (if configured) ---
        instruction_context_str = ""
        instruction_dir_path_str = self.config.get("INSTRUCTION_DIR")
        if instruction_dir_path_str:
            instruction_dir = Path(instruction_dir_path_str).resolve()
            if instruction_dir.is_dir():
                logger.info(f"Reading instruction files from: {instruction_dir}")
                instruction_parts = []
                try:
                    for instruction_file in instruction_dir.iterdir():
                        if instruction_file.is_file():
                            # Use the imported read_file_content utility
                            content = read_file_content(str(instruction_file), remove_comments_blank_lines=False) # Keep formatting
                            if content:
                                instruction_parts.append(f"// --- Instruction File: {instruction_file.name} ---\n{content}")
                            else:
                                logger.warning(f"Could not read instruction file or it was empty: {instruction_file}")
                    if instruction_parts:
                        instruction_context_str = "\n\n".join(instruction_parts)
                        logger.info(f"Loaded instruction context ({len(instruction_context_str)} chars).")
                    else:
                        logger.info("Instruction directory exists but contains no readable files.")
                except Exception as e:
                    logger.error(f"Error reading instruction files from {instruction_dir}: {e}", exc_info=True)
            else:
                logger.warning(f"Instruction directory specified but not found or not a directory: {instruction_dir}")
        else:
            logger.info("No INSTRUCTION_DIR configured. Skipping instruction context loading.")

        # --- Process packages iteratively, saving state after each ---
        for pkg_id in packages_to_process_this_run:
            logger.info(f"Processing Step 3 for package: {pkg_id}")
            # State is saved immediately when status is updated
            self.state_manager.update_package_state(pkg_id, status='running_structure')

            try:
                # Re-fetch package info inside the loop in case state was updated externally?
                # Probably not necessary if Orchestrator manages runs sequentially.
                pkg_info = self.state_manager.get_package_info(pkg_id) # Use already fetched data if possible? No, get fresh state.
                if not pkg_info:
                     # This should not happen if validation passed, but check defensively
                     logger.error(f"Critical state inconsistency: Package info for {pkg_id} disappeared during execution.")
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state mid-execution.")

                primary_files = pkg_info.get('files', [])
                # Determine dependencies using the existing method on ContextManager
                if hasattr(self.context_manager, '_get_dependencies_for_package'):
                     dependency_files = self.context_manager._get_dependencies_for_package(primary_files) # Pass the list of files
                else:
                     # This case should ideally not happen if ContextManager is correctly initialized
                     logger.error(f"Critical: Method '_get_dependencies_for_package' not found on ContextManager. Cannot determine dependencies for {pkg_id}.")
                     dependency_files = [] # Proceed without dependencies

                # --- Fetch additional context ---
                all_package_summaries = self.context_manager.get_all_package_summaries()
                existing_structure = self.context_manager.get_existing_structure(pkg_id)
                all_godot_files = self.context_manager.get_all_existing_godot_files()

                # --- Assemble context ---
                context = self.context_manager.get_context_for_step(
                    step_name="STRUCTURE_DEFINITION",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    work_package_id=pkg_id,
                    work_package_description=pkg_info.get('description', ''),
                    work_package_files=primary_files,
                    instruction_context=instruction_context_str,
                    all_package_summaries=all_package_summaries,
                    existing_package_structure=existing_structure,
                    all_existing_godot_files=all_godot_files
                )

                if not context:
                     raise ValueError("Failed to assemble context for Step 3.")

                # Instantiate Agent and Task
                agent = StructureDefinerAgent().get_agent(llm_instance=mapper_llm_instance)

                # Create task - conditionally use output_json based on LLM type/config
                task_creator = DefineStructureTask()
                if isinstance(mapper_llm_instance, GoogleGenAI_LLM) and mapper_llm_instance.response_schema:
                    task = task_creator.create_task(agent, context) # Let LLM/SDK handle JSON via schema
                else:
                    task = task_creator.create_task(agent, context, output_json=GodotStructureOutput) # Use CrewAI's mechanism

                # Create and run Crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=mapper_llm_instance,
                    process=Process.sequential,
                    verbose=True
                )
                logger.info(f"Kicking off Crew for Step 3 (Package: {pkg_id})...")
                result = crew.kickoff()
                logger.info(f"Step 3 Crew finished for package: {pkg_id}")

                parsed_result_dict = None # Use a different name to avoid confusion with Pydantic model
                raw_output_for_debug = None

                # --- Result Parsing Logic ---
                # Check if CrewAI already parsed it (e.g., if output_json was used with non-Gemini)
                if isinstance(result, GodotStructureOutput): # Check for Pydantic model first
                    logger.info(f"Crew returned validated Pydantic object for {pkg_id}.")
                    # Convert Pydantic model to dict for saving
                    parsed_result_dict = result.model_dump()
                elif isinstance(result, str):
                    logger.warning(f"Crew returned a string for {pkg_id}. Attempting to parse as JSON.")
                    raw_output_for_debug = result
                    parsed_dict_attempt = parse_json_from_string(result)
                    if parsed_dict_attempt:
                        logger.info(f"Successfully parsed JSON string fallback for {pkg_id}.")
                        # Optional: Could try to re-validate with Pydantic here if needed
                        try:
                            # Validate the parsed dict against the Pydantic model
                            GodotStructureOutput(**parsed_dict_attempt)
                            logger.info("Parsed JSON conforms to Pydantic model.")
                            parsed_result_dict = parsed_dict_attempt # Assign if valid
                        except Exception as pydantic_err:
                            logger.error(f"Parsed JSON does NOT conform to Pydantic model: {pydantic_err}")
                            parsed_result_dict = None # Invalidate if structure is wrong
                    else:
                        logger.error(f"Failed to parse string output as JSON for {pkg_id}.")
                else:
                    # Handle unexpected types (like raw CrewOutput if parsing fails internally)
                    if hasattr(result, 'raw') and isinstance(result.raw, str):
                         logger.warning(f"Crew returned unexpected type ({type(result)}), but found raw string. Attempting parse.")
                         raw_output_for_debug = result.raw
                         parsed_dict_attempt = parse_json_from_string(result.raw)
                         if parsed_dict_attempt:
                              logger.info(f"Successfully parsed JSON from raw string fallback for {pkg_id}.")
                              # Optional: Re-validate here too
                              try:
                                  GodotStructureOutput(**parsed_dict_attempt)
                                  logger.info("Parsed JSON from raw string conforms to Pydantic model.")
                                  parsed_result_dict = parsed_dict_attempt # Assign if valid
                              except Exception as pydantic_err:
                                  logger.error(f"Parsed JSON from raw string does NOT conform to Pydantic model: {pydantic_err}")
                                  parsed_result_dict = None # Invalidate
                         else:
                              logger.error(f"Failed to parse raw string output as JSON for {pkg_id}.")
                    else:
                         logger.error(f"Step 3 Crew did not return a Pydantic object or a parsable JSON string. Type: {type(result)}")
                         raw_output_for_debug = str(result) # Log the string representation

                if not parsed_result_dict:
                     logger.debug(f"Raw output for {pkg_id} (if available): {raw_output_for_debug}") # Log raw output on failure
                     raise ValueError("Step 3 Crew did not produce a valid structure output dictionary.")

                # --- Save the artifact as JSON ---
                artifact_filename = f"package_{pkg_id}_structure.json"
                # Use analysis_dir Path object correctly
                artifact_path = analysis_dir / artifact_filename
                try:
                    analysis_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists using Path object
                    with open(artifact_path, 'w', encoding='utf-8') as f:
                        # Save the dictionary version
                        json.dump(parsed_result_dict, f, indent=2) # Save the parsed dict
                    logger.info(f"Saved structure definition artifact: {artifact_path}")
                    # --- Update state with success ---
                    self.state_manager.update_package_state(
                        pkg_id,
                        status=completed_status, # Use 'structure_defined'
                        artifacts={'structure_json': artifact_filename},
                        error=None # Clear previous error on success
                    )
                except IOError as e:
                    # Raise specific error for saving failure
                    raise IOError(f"Failed to save structure artifact {artifact_path}: {e}")
                except TypeError as e:
                     # Raise specific error for JSON serialization failure (shouldn't happen with parsed dict)
                     raise TypeError(f"Failed to serialize structure result to JSON: {e}")

            except Exception as e:
                logger.error(f"An error occurred during Step 3 processing for package {pkg_id}: {e}", exc_info=True)
                # --- Update state with failure ---
                self.state_manager.update_package_state(
                    pkg_id,
                    status=failed_status_prefix, # Use 'failed_structure'
                    error=str(e)
                )
                overall_success_this_run = False
            # No finally block needed here as state is saved within update_package_state

        # --- Final Workflow Status Check ---
        # Check if all packages that were *potentially* eligible are now done or failed
        all_potential_targets_done_or_failed = True
        final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
        for pkg_id in potential_target_package_ids:
             status = final_packages_state.get(pkg_id, {}).get('status')
             # Use correct completed/failed status checks for this step
             if not (status == completed_status or (status and status.startswith(failed_status_prefix))):
                  all_potential_targets_done_or_failed = False
                  logger.debug(f"Package {pkg_id} is still pending Step 3 completion (status: {status}).")
                  break

        if all_potential_targets_done_or_failed and potential_target_package_ids: # Only complete if there were potential targets
             logger.info("All potential target packages for Step 3 are now processed or failed.")
             # Avoid overwriting a global failed state if some packages failed earlier
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  self.state_manager.update_workflow_status('step3_complete')
        elif not overall_success_this_run:
             # If any package failed *in this specific run*, ensure the global status reflects a failure in Step 3
             # Avoid overwriting if it's already failed at an earlier step (e.g., failed_step2)
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  # Use specific failure status for this step
                  self.state_manager.update_workflow_status('failed_step3', "One or more packages failed during structure definition in the latest run.")
        # Else: Keep 'running_step3' or previous status if some packages are still pending

        logger.info(f"--- Finished Step 3 Execution Run (Success This Run: {overall_success_this_run}) ---")
        # Return the success status of *this specific run*
        return overall_success_this_run
