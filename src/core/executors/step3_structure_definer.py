# src/core/executors/step3_structure_definer.py
# Standard library imports
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

# CrewAI imports
from crewai import Crew, Process, Task, Agent
from crewai.tasks.task_output import TaskOutput # Import for callback type hint

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, read_file_content # Keep read_file_content if needed elsewhere

# Import agents for Step 3
from src.agents.step3.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step3.global_context_analyst import get_global_context_analyst_agent
from src.agents.step3.structure_designer import get_structure_designer_agent
from src.agents.step3.json_output_formatter import get_json_output_formatter_agent

# Import Task definition for Step 3
from src.tasks.define_structure import create_hierarchical_define_structure_task, GodotStructureOutput # Import Pydantic model too

# Import utilities
from src.utils.json_utils import parse_json_from_string
from src.utils.formatting_utils import format_structure_to_markdown # Keep for context assembly if needed
from src.logger_setup import get_logger
import src.config as global_config

# Import specific LLM classes if needed for instantiation check
from src.llms.google_genai_llm import GoogleGenAI_LLM
from crewai import LLM as CrewAI_LLM

logger = get_logger(__name__)

class Step3Executor(StepExecutor):
    """Executes Step 3: Godot Structure Definition using a hierarchical multi-agent crew."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config_dict: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]],
                 tools: Dict[Type, Any]): # Tools likely not needed here
        super().__init__(state_manager, context_manager, config_dict, llm_configs, tools)

        # --- Get LLM Instances ---
        # Define grouped roles used by Step 3 agents + manager
        analyzer_config = self._get_llm_config('analyzer')
        designer_config = self._get_llm_config('designer_planner')
        utility_config = self._get_llm_config('utility')
        manager_config = self._get_llm_config('manager')

        # Get LLM instances using grouped roles
        self.cpp_analyst_llm = self._get_llm_instance_by_role('analyzer', analyzer_config)
        self.global_analyst_llm = self._get_llm_instance_by_role('analyzer', analyzer_config)
        self.designer_llm = self._get_llm_instance_by_role('designer_planner', designer_config)
        self.formatter_llm = self._get_llm_instance_by_role('utility', utility_config) # Formatter uses utility
        self.manager_llm = self._get_llm_instance_by_role('manager', manager_config)

        if not self.manager_llm: # Manager is crucial
             logger.error("Missing critical LLM configuration for Step 3 Manager role. Cannot proceed.")
             raise ValueError("Missing critical LLM configuration for Step 3 Manager role.")

        # --- Instantiate Agents ---
        # Use the specific LLM instances fetched above
        self.cpp_analyst_agent = get_cpp_code_analyst_agent(self.cpp_analyst_llm)
        self.global_analyst_agent = get_global_context_analyst_agent(self.global_analyst_llm)
        self.designer_agent = get_structure_designer_agent(self.designer_llm)
        self.formatter_agent = get_json_output_formatter_agent(self.formatter_llm)

        # Log if any LLM instance is missing (should ideally be caught by verification below)
        if not self.cpp_analyst_llm: logger.warning("Step3 Cpp Analyst LLM instance is missing.")
        if not self.global_analyst_llm: logger.warning("Step3 Global Analyst LLM instance is missing.")
        if not self.designer_llm: logger.warning("Step3 Structure Designer LLM instance is missing.")
        if not self.formatter_llm: logger.warning("Step3 Formatter LLM instance is missing.")

        logger.info("Instantiated CrewAI agents for Step 3.")

        # Verify all agents have an LLM assigned
        agent_list = [self.cpp_analyst_agent, self.global_analyst_agent, self.designer_agent, self.formatter_agent]
        for agent in agent_list:
             if not agent.llm:
                  logger.error(f"Agent '{agent.role}' in Step3Executor ended up with no assigned LLM.")
                  raise ValueError(f"Agent '{agent.role}' missing LLM instance.")
             else:
                  logger.debug(f"Agent '{agent.role}' successfully assigned LLM: {type(agent.llm)}")

        # --- Instantiate Task Creator ---
        # Task creator function is imported directly, no need to instantiate a class
        logger.info("Task creator function available for Step 3.")


    def _get_llm_instance_by_role(self, role: str, fallback_config: Optional[Dict] = None) -> Optional[Any]:
         """
         Helper to get and instantiate an LLM object for a specific role,
         potentially falling back to another role's config.
         Returns an instantiated LLM object or None.
         """
         llm_config = self._get_llm_config(role) # Fetches config dict from Orchestrator's map
         if not llm_config and fallback_config:
             logger.warning(f"LLM config for role '{role}' not found. Attempting fallback.")
             llm_config = fallback_config # Use the fallback config dict
         elif not llm_config and not fallback_config:
              logger.warning(f"LLM config for role '{role}' not found and no fallback provided. Cannot instantiate LLM.")
              return None

         if not llm_config or not isinstance(llm_config, dict):
              logger.error(f"Invalid or missing LLM config dictionary for role '{role}' (after fallback). Type: {type(llm_config)}")
              return None

         # Now, instantiate based on the final llm_config dictionary
         try:
             model_identifier = llm_config.get("model", "[Unknown Model]")
             logger.debug(f"Attempting to instantiate LLM for role '{role}' using model '{model_identifier}' with config: {llm_config}")

             # Explicitly check for Gemini model to use the custom class
             if model_identifier.startswith(("gemini/", "google/")):
                 # from src.llms.google_genai_llm import GoogleGenAI_LLM # Already imported at top
                 llm_config_copy = llm_config.copy()
                 llm_config_copy.setdefault('timeout', global_config.GEMINI_TIMEOUT)
                 # Add specific response schema/mime type only if needed for this role
                 if role == 'formatter': # Only formatter needs strict JSON output via schema
                     llm_config_copy['response_schema'] = GodotStructureOutput
                     llm_config_copy['response_mime_type'] = "application/json"
                 llm_instance = GoogleGenAI_LLM(**llm_config_copy)
                 logger.info(f"Successfully instantiated GoogleGenAI_LLM for role '{role}': {model_identifier}")
             else:
                 # Use default CrewAI LLM for other models
                 llm_instance = CrewAI_LLM(**llm_config)
                 logger.info(f"Successfully instantiated default crewai.LLM for role '{role}': {model_identifier}")

             return llm_instance # Return the instantiated object

         except ImportError as ie:
              logger.error(f"Import error during LLM instantiation for role '{role}': {ie}. Check dependencies.")
              return None
         except TypeError as te:
              logger.error(f"Type error during LLM instantiation for role '{role}': {te}. Check config parameters for model '{model_identifier}'. Config: {llm_config}")
              return None
         except Exception as e:
             logger.error(f"Unexpected error during LLM instantiation for role '{role}' with model '{model_identifier}': {e}", exc_info=True)
             return None


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
        logger.info(f"--- Starting Step 3 Execution (Hierarchical): Define Structure (Requested: {package_ids or 'All Eligible'}, Force: {force}) ---")

        # --- Fetch and Validate Processing Order (Logic Remains Same) ---
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

        # --- Load Instruction Context (Logic Remains Same) ---
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
                            content = read_file_content(str(instruction_file), remove_comments_blank_lines=False)
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
            self.state_manager.update_package_state(pkg_id, status='running_structure')

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     logger.error(f"Critical state inconsistency: Package info for {pkg_id} disappeared during execution.")
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state mid-execution.")

                primary_files = pkg_info.get('files', [])
                dependency_files = self.context_manager._get_dependencies_for_package(primary_files) if hasattr(self.context_manager, '_get_dependencies_for_package') else []

                # --- Fetch additional context (Modify to use new ContextManager method) ---
                all_package_summaries = self.context_manager.get_all_package_summaries()
                existing_structure = self.context_manager.get_existing_structure(pkg_id)
                # !! IMPORTANT: Replace this with call to get files from state !!
                # This requires ContextManager to have access to StateManager
                # Assuming ContextManager has a method like get_globally_defined_godot_files()
                if hasattr(self.context_manager, 'get_globally_defined_godot_files_from_state'):
                     all_defined_godot_files = self.context_manager.get_globally_defined_godot_files_from_state()
                else:
                     logger.error("ContextManager is missing the required method 'get_globally_defined_godot_files_from_state'. Cannot get global file list.")
                     all_defined_godot_files = [] # Proceed with empty list, but log error

                # --- Assemble context ---
                context = self.context_manager.get_context_for_step(
                    step_name="STRUCTURE_DEFINITION",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    work_package_id=pkg_id,
                    work_package_description=pkg_info.get('description', ''),
                    work_package_files=primary_files, # Keep for reference if needed by LLM
                    instruction_context=instruction_context_str,
                    all_package_summaries=all_package_summaries,
                    existing_package_structure=existing_structure,
                    globally_defined_godot_files=all_defined_godot_files # Pass the list from state
                )

                if not context:
                     raise ValueError("Failed to assemble context for Step 3.")

                # --- Create Hierarchical Task ---
                task = create_hierarchical_define_structure_task(
                    manager_agent=None, # Manager is implicit
                    context=context,
                    package_id=pkg_id
                )

                # --- Create and run Hierarchical Crew ---
                crew = Crew(
                    agents=[ # List ALL worker agents for Step 3
                        self.cpp_analyst_agent,
                        self.global_analyst_agent,
                        self.designer_agent,
                        self.formatter_agent
                    ],
                    tasks=[task], # Single task for the manager
                    process=Process.hierarchical,
                    manager_llm=self.manager_llm,
                    memory=True, # Enable memory for context flow within the crew
                    verbose=True,
                    task_callback=self._log_step3_task_completion # Add callback
                )
                logger.info(f"Kicking off Hierarchical Crew for Step 3 (Package: {pkg_id})...")
                result = crew.kickoff() # Expecting final JSON string from FormatterAgent
                logger.info(f"Step 3 Hierarchical Crew finished for package: {pkg_id}")
                logger.debug(f"Crew Result Raw Output:\n{result}")

                parsed_result_dict = None # Use a different name to avoid confusion with Pydantic model
                raw_output_for_debug = None

                # --- Result Parsing Logic ---
                # Check if CrewAI already parsed it (e.g., if output_json was used with non-Gemini)
                if isinstance(result, GodotStructureOutput): # Check for Pydantic model first
                    logger.info(f"Crew returned validated Pydantic object for {pkg_id}.")
                    # Convert Pydantic model to dict for saving
                    parsed_result_dict = result.model_dump()
                elif isinstance(result, str): # Fallback parsing
                    logger.warning(f"Crew returned a string for {pkg_id}. Attempting to parse as JSON.")
                    raw_output_for_debug = result
                    parsed_dict_attempt = parse_json_from_string(result)
                    if parsed_dict_attempt and isinstance(parsed_dict_attempt, dict):
                        try:
                            GodotStructureOutput(**parsed_dict_attempt) # Validate structure
                            logger.info("Parsed JSON string conforms to GodotStructureOutput model.")
                            parsed_result_dict = parsed_dict_attempt
                        except Exception as pydantic_err:
                            logger.error(f"Parsed JSON string does NOT conform to GodotStructureOutput model: {pydantic_err}")
                    else:
                        logger.error(f"Failed to parse string output as JSON dictionary for {pkg_id}.")
                elif isinstance(result, dict): # Handle if kickoff returns dict directly
                     try:
                          GodotStructureOutput(**result)
                          logger.info("Crew output dict conforms to GodotStructureOutput model.")
                          parsed_result_dict = result
                     except Exception as pydantic_err:
                          logger.error(f"Crew output dict does NOT conform to GodotStructureOutput model: {pydantic_err}")
                else:
                    logger.error(f"Step 3 Crew returned unexpected type: {type(result)}")
                    raw_output_for_debug = str(result)

                if not parsed_result_dict:
                     logger.debug(f"Raw output for {pkg_id} (if available): {raw_output_for_debug}")
                     raise ValueError("Step 3 Hierarchical Crew failed to produce valid structure output dictionary.")

                # --- Extract defined file paths ---
                defined_files = []
                for scene in parsed_result_dict.get("scenes", []):
                    if isinstance(scene, dict) and "path" in scene: defined_files.append(scene["path"])
                for script in parsed_result_dict.get("scripts", []):
                    if isinstance(script, dict) and "path" in script: defined_files.append(script["path"])
                for resource in parsed_result_dict.get("resources", []):
                    if isinstance(resource, dict) and "path" in resource: defined_files.append(resource["path"])
                for m_script in parsed_result_dict.get("migration_scripts", []):
                    if isinstance(m_script, dict) and "path" in m_script: defined_files.append(m_script["path"])
                defined_files = sorted(list(set(defined_files))) # Unique and sorted
                logger.info(f"Extracted {len(defined_files)} defined output files for package {pkg_id}.")

                # --- Save the artifact as JSON using StateManager ---
                artifact_filename = f"package_{pkg_id}_structure.json"
                save_ok = self.state_manager.save_artifact(artifact_filename, parsed_result_dict, is_json=True)
                if not save_ok:
                     # Log error but attempt to update state anyway? Or raise? Let's raise.
                     raise IOError(f"Failed to save structure artifact via StateManager: {artifact_filename}")

                # --- Update state with success and defined files ---
                artifacts_to_update = {
                    'structure_json': artifact_filename,
                    'defined_godot_files': defined_files # Store the extracted list
                }
                self.state_manager.update_package_state(
                    pkg_id,
                    status=completed_status, # 'structure_defined'
                    artifacts=artifacts_to_update,
                    error=None # Clear previous error on success
                )

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
        return overall_success_this_run

    # --- Callback Method ---
    def _log_step3_task_completion(self, task_output: TaskOutput):
        """Logs information about each completed task within the Step 3 crew."""
        try:
            agent_role = task_output.agent.role if task_output.agent else "Unknown Agent"
            task_desc_snippet = task_output.task.description[:100] + "..." if task_output.task else "Unknown Task"
            output_snippet = task_output.raw_output[:150].replace('\n', ' ') + "..." if task_output.raw_output else "No output"

            logger.info(f"[Step 3 Crew Callback] Task Completed:")
            logger.info(f"  - Agent: {agent_role}")
            logger.info(f"  - Task Desc (Start): {task_desc_snippet}")
            logger.info(f"  - Output (Start): {output_snippet}")
        except Exception as e:
            logger.error(f"[Step 3 Crew Callback] Error processing task output: {e}", exc_info=True)
            try:
                 logger.debug(f"Raw task_output object: {task_output}")
            except Exception:
                 logger.debug("Could not log raw task_output object.")
