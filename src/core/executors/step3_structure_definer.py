# src/core/executors/step3_structure_definer.py
# Standard library imports
import os # Import os to read environment variables
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

# CrewAI imports
from crewai import Crew, Process, Task, Agent
from crewai.tasks.task_output import TaskOutput # Import for callback type hint
from crewai.crews.crew_output import CrewOutput # Import for handling crew result type

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens, read_file_content # Keep read_file_content if needed elsewhere

# Import agents for Step 3
from src.agents.step3.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step3.global_context_analyst import get_global_context_analyst_agent
from src.agents.step3.structure_designer import get_structure_designer_agent
from src.agents.step3.json_output_formatter import get_json_output_formatter_agent

# Import Task definition for Step 3
from src.tasks.step3.define_structure import create_hierarchical_define_structure_task, GodotStructureOutput # Import Pydantic model too

# Import utilities
from src.utils.json_utils import parse_json_from_string
from src.logger_setup import get_logger
import src.config as config


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
        logger.info("Step3Executor initialized. LLMs and agents will be created dynamically.")

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

        # --- Instantiate LLMs and Agents using the new helper ---
        cpp_analyst_llm = self._create_llm_instance('ANALYZER_MODEL')
        global_analyst_llm = self._create_llm_instance('ANALYZER_MODEL')
        designer_llm = self._create_llm_instance('DESIGNER_PLANNER_MODEL')
        formatter_llm = self._create_llm_instance('UTILITY_MODEL', response_schema_class=GodotStructureOutput)
        manager_llm = self._create_llm_instance('MANAGER_MODEL')

        # Check if all required LLMs were instantiated successfully
        required_llms = {
            'Manager': manager_llm,
            'CPP Analyst': cpp_analyst_llm,
            'Global Analyst': global_analyst_llm,
            'Designer': designer_llm,
            'Formatter': formatter_llm
        }
        missing_llms = [name for name, llm in required_llms.items() if not llm]

        if missing_llms:
             logger.error(f"Missing critical LLM instances for Step 3: {', '.join(missing_llms)}. Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step3', f"Missing LLM config for: {', '.join(missing_llms)}")
             return False

        cpp_analyst_agent = get_cpp_code_analyst_agent(cpp_analyst_llm)
        global_analyst_agent = get_global_context_analyst_agent(global_analyst_llm)
        designer_agent = get_structure_designer_agent(designer_llm)
        formatter_agent = get_json_output_formatter_agent(formatter_llm)
        logger.info("Instantiated LLMs and Agents for Step 3 execution.")

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
        running_status = 'running_structure' # Status indicating success in *this* step

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
                is_running = (current_status == running_status)

                if is_target:
                    packages_to_process_this_run.append(pkg_id)
                elif force and (is_failed_this_step or is_already_completed or is_running):
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

        # --- Process packages iteratively, saving state after each ---
        for pkg_id in packages_to_process_this_run:
            logger.info(f"Processing Step 3 for package: {pkg_id}")

            # --- Pre-run status updates ---
            pkg_info_at_start_of_execute = all_packages.get(pkg_id, {}) # Use the initial snapshot
            original_status_for_logic = pkg_info_at_start_of_execute.get('status')

            if force and (original_status_for_logic.startswith(failed_status_prefix) or \
                           original_status_for_logic == completed_status):
                logger.info(f"Forcing package {pkg_id} (original status: {original_status_for_logic}): Resetting status to '{target_status}' and clearing error.")
                self.state_manager.update_package_state(pkg_id, target_status, error=None)
            elif not force and original_status_for_logic.startswith(failed_status_prefix): # Auto-retry failed
                logger.info(f"Auto-retrying failed package {pkg_id} (original status: {original_status_for_logic}): Clearing error.")
                # Keep original 'failed_structure...' status but clear the error message for a fresh attempt.
                # The 'running_structure' status will be set next.
                self.state_manager.update_package_state(pkg_id, original_status_for_logic, error=None)
            elif original_status_for_logic == running_status: # Resuming a 'running' package
                logger.info(f"Resuming package {pkg_id} (original status: {original_status_for_logic}).")
                # Ensure error is cleared if it was somehow set during a crash
                self.state_manager.update_package_state(pkg_id, original_status_for_logic, error=None)
            
            self.state_manager.update_package_state(pkg_id, status='running_structure') # Set to current running status

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     logger.error(f"Critical state inconsistency: Package info for {pkg_id} disappeared during execution.")
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state mid-execution.")

                # --- Assemble Context using ContextManager methods ---
                context_parts = []

                # 1. General Instructions
                instr_context = self.context_manager.get_instruction_context()

                # 2. Package Description
                pkg_desc = pkg_info.get('description', 'N/A')
                context_parts.append(f"**Current Package ({pkg_id}) Description:**\n{pkg_desc}")

                # 3. Source Files List & Roles
                source_files = self.context_manager.get_source_file_list(pkg_id)
                if source_files:
                    source_list_str = "\n".join([f"- `{f['file_path']}`: {f['role']}" for f in source_files])
                    context_parts.append(f"**Source Files & Roles for {pkg_id}:**\n{source_list_str}")

                # 4. Source Code Content (respecting limits)
                # Calculate remaining token budget for source code
                temp_context_so_far = "\n\n".join(context_parts)
                tokens_so_far = count_tokens(temp_context_so_far) + count_tokens(instr_context) # Include instruction tokens
                max_source_tokens = (config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER) - tokens_so_far - 1000 # Extra buffer
                if max_source_tokens > 0:
                    source_code = self.context_manager.get_work_package_source_code_content(pkg_id, max_tokens=max_source_tokens)
                    if source_code:
                        context_parts.append(f"**Source Code for {pkg_id}:**\n{source_code}")
                    else:
                        logger.warning(f"Could not retrieve source code for {pkg_id} within token limits.")
                else:
                    logger.warning(f"Not enough token budget remaining for source code of {pkg_id}.")


                # 5. Existing Structure (if any) - Load artifact directly
                structure_artifact_name = pkg_info.get('artifacts', {}).get('structure_json')
                existing_structure_json = None
                if structure_artifact_name:
                     existing_structure_json = self.state_manager.load_artifact(structure_artifact_name, expect_json=True)
                     if existing_structure_json:
                          context_parts.append(f"**Existing Structure Definition for {pkg_id} (for refinement):**\n```json\n{json.dumps(existing_structure_json, indent=2)}\n```")

                # 6. Globally Defined Godot Files (from overall mappings artifact)
                overall_mappings = self.state_manager.load_artifact("mappings.json", expect_json=True)
                all_defined_godot_files = []
                if overall_mappings and isinstance(overall_mappings.get("all_output_files"), list):
                     all_defined_godot_files = overall_mappings["all_output_files"]
                     # Filter out files potentially defined *by this package* if refining
                     if existing_structure_json:
                          current_pkg_files = set()
                          for scene in existing_structure_json.get("scenes", []): current_pkg_files.add(scene.get("path"))
                          for script in existing_structure_json.get("scripts", []): current_pkg_files.add(script.get("path"))
                          for res in existing_structure_json.get("resources", []): current_pkg_files.add(res.get("path"))
                          all_defined_godot_files = [f for f in all_defined_godot_files if f not in current_pkg_files]

                if all_defined_godot_files:
                     files_list_str = "\n".join([f"- `{f}`" for f in sorted(all_defined_godot_files)])
                     context_parts.append(f"**Globally Defined Godot Files (excluding current package if refining):**\n{files_list_str}")


                # Combine all context parts intended for the main prompt
                context_str = "\n\n---\n\n".join(context_parts)
                final_tokens = count_tokens(context_str) + count_tokens(instr_context) # Add instruction tokens for logging
                logger.info(f"Assembled context for Step 3 - {pkg_id} ({final_tokens} tokens).")
                if final_tokens >= (config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER):
                     logger.warning(f"Context for {pkg_id} might be near or exceeding token limits!")

                if not context_str: # Should not happen if basic info exists, but check
                    raise ValueError(f"Failed to assemble any context for Step 3 - {pkg_id}.")


                # --- Create Hierarchical Task ---
                task = create_hierarchical_define_structure_task(
                    manager_agent=None, # Manager is implicit
                    context=context_str, # Pass the assembled context string
                    package_id=pkg_id,
                    instructions=instr_context # Pass the fetched instructions
                )

                # --- Create and run Hierarchical Crew ---
                # Determine Embedder Configuration (moved closer to Crew creation)
                embedder_config = None
                vertex_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                vertex_location = os.getenv("GOOGLE_CLOUD_LOCATION")
                if vertex_project_id and vertex_location:
                    logger.info("Configuring embedder for Vertex AI.")
                    embedder_config = {"provider": "vertexai", "config": {"project_id": vertex_project_id, "location": vertex_location}}
                elif config.GEMINI_API_KEY:
                    logger.info("Configuring embedder for Google Generative AI.")
                    embedder_config = {"provider": "google", "config": {"api_key": config.GEMINI_API_KEY}}
                else:
                    logger.warning("Neither Vertex AI env vars nor GEMINI_API_KEY found. Attempting Vertex AI with ADC.")
                    embedder_config = {"provider": "vertexai", "config": {}}
                logger.info(f"Final Embedder Configuration for Crew: {embedder_config}")

                crew = Crew(
                    agents=[ # List ALL worker agents for Step 3
                        cpp_analyst_agent,
                        global_analyst_agent,
                        designer_agent,
                        formatter_agent
                    ],
                    tasks=[task], # Single task for the manager
                    process=Process.hierarchical,
                    manager_llm=manager_llm,
                    memory=True,
                    embedder=embedder_config, # Use the dynamically determined config
                    verbose=True,
                    task_callback=self._log_step3_task_completion # Add callback
                )
                logger.info(f"Kicking off Hierarchical Crew for Step 3 (Package: {pkg_id})")
                result = crew.kickoff() # Expecting final JSON string from FormatterAgent or CrewOutput
                logger.info(f"Step 3 Hierarchical Crew finished for package: {pkg_id}")
                logger.debug(f"Crew Result Raw Output (Type: {type(result)}):\n{result}")

                parsed_result_dict = None
                raw_output_for_parsing = None # Store the string to be parsed

                # --- Result Extraction and Parsing Logic ---
                if isinstance(result, CrewOutput):
                    logger.info(f"Crew returned CrewOutput object for {pkg_id}. Extracting raw output.")
                    # CrewOutput often stores the final raw output in the 'raw' attribute
                    # or potentially in the last task's output. Let's prioritize 'raw'.
                    if hasattr(result, 'raw') and isinstance(result.raw, str):
                        raw_output_for_parsing = result.raw
                        logger.debug(f"Extracted raw output from CrewOutput.raw")
                    elif result.tasks_output and isinstance(result.tasks_output, list) and len(result.tasks_output) > 0:
                         # Fallback: try the raw output of the last task
                         last_task_output = result.tasks_output[-1]
                         if isinstance(last_task_output, TaskOutput) and isinstance(last_task_output.raw_output, str):
                              raw_output_for_parsing = last_task_output.raw_output
                              logger.warning("Extracted raw output from last task in CrewOutput.tasks_output (fallback).")
                         else:
                              logger.error("Could not extract raw string output from CrewOutput's last task.")
                              raw_output_for_parsing = str(result) # Use string representation as last resort for debugging
                    else:
                         logger.error("CrewOutput object did not contain expected 'raw' string or usable 'tasks_output'.")
                         raw_output_for_parsing = str(result) # Use string representation as last resort for debugging

                elif isinstance(result, GodotStructureOutput): # Check for Pydantic model (e.g., if output_pydantic used)
                    logger.info(f"Crew returned validated Pydantic object for {pkg_id}.")
                    parsed_result_dict = result.model_dump() # Already validated and parsed
                elif isinstance(result, str): # Direct string output
                    logger.info(f"Crew returned a string directly for {pkg_id}.")
                    raw_output_for_parsing = result
                elif isinstance(result, dict): # Direct dict output (less common for final, but handle)
                     logger.warning(f"Crew returned a dictionary directly for {pkg_id}. Attempting validation.")
                     try:
                          GodotStructureOutput(**result)
                          logger.info("Crew output dict conforms to GodotStructureOutput model.")
                          parsed_result_dict = result # Already a dict, validated
                     except Exception as pydantic_err:
                          logger.error(f"Crew output dict does NOT conform to GodotStructureOutput model: {pydantic_err}")
                          raw_output_for_parsing = json.dumps(result) # Convert to string for debug log
                else:
                    logger.error(f"Step 3 Crew returned unexpected type: {type(result)}")
                    raw_output_for_parsing = str(result) # Use string representation for debug log

                # --- Parse the extracted raw string if necessary ---
                if raw_output_for_parsing is not None and parsed_result_dict is None:
                    logger.info(f"Attempting to parse raw output string for {pkg_id} using consolidated parser.")
                    logger.debug(f"Raw string being sent to parse_json_from_string:\n{raw_output_for_parsing}")
                    parsed_dict_attempt = parse_json_from_string(raw_output_for_parsing)

                    if parsed_dict_attempt and isinstance(parsed_dict_attempt, dict):
                        try:
                            # Validate the parsed dictionary against the Pydantic model
                            GodotStructureOutput(**parsed_dict_attempt)
                            logger.info("Parsed JSON dictionary conforms to GodotStructureOutput model.")
                            parsed_result_dict = parsed_dict_attempt # Assign the validated dict
                        except Exception as pydantic_err:
                            logger.error(f"Parsed JSON dictionary does NOT conform to GodotStructureOutput model: {pydantic_err}")
                            logger.debug(f"Invalid dictionary structure received (from raw string):\n{json.dumps(parsed_dict_attempt, indent=2)}")
                            # Keep parsed_result_dict as None
                    elif parsed_dict_attempt: # Parsed, but not a dictionary
                         logger.error(f"Consolidated parser returned a result, but it's not a dictionary (type: {type(parsed_dict_attempt)}). Expected dict for GodotStructureOutput.")
                         logger.debug(f"Non-dictionary result: {parsed_dict_attempt}")
                    else: # Parsing failed completely
                        logger.error(f"Consolidated parser failed to extract/parse JSON dictionary from the raw output for {pkg_id}.")
                        logger.debug(f"Raw string content that failed parsing:\n{raw_output_for_parsing}")
                        # Keep parsed_result_dict as None

                # --- Final Check ---
                if not parsed_result_dict:
                     # Log the raw output again if parsing/validation failed
                     logger.debug(f"Final raw output for {pkg_id} before failing (parsing/validation unsuccessful): {raw_output_for_parsing}")
                     raise ValueError("Step 3 Hierarchical Crew failed to produce a valid, parsable, and schema-compliant structure output dictionary.")

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
    def _log_step3_task_completion(self, task_output: Any): # Use Any for initial check
        """Logs information about each completed task within the Step 3 crew."""
        try:
            # --- Safely Access Attributes ---
            agent_role = "Unknown Agent"
            task_desc_snippet = "Unknown Task"
            output_snippet = "No output"

            if isinstance(task_output, TaskOutput):
                # Access agent safely
                if hasattr(task_output, 'agent') and isinstance(task_output.agent, Agent):
                    agent_role = task_output.agent.role
                elif hasattr(task_output, 'agent'):
                     logger.warning(f"[Step 3 Crew Callback] task_output.agent is not an Agent object (Type: {type(task_output.agent)}).")
                     agent_role = f"Agent (Type: {type(task_output.agent).__name__})" # Log type if not Agent
                else:
                     logger.warning("[Step 3 Crew Callback] task_output object missing 'agent' attribute.")

                # Access task description safely
                if hasattr(task_output, 'task') and task_output.task and hasattr(task_output.task, 'description'):
                    task_desc_snippet = task_output.task.description[:100] + "..."
                elif hasattr(task_output, 'description') and isinstance(task_output.description, str): # Sometimes description is direct attribute
                     task_desc_snippet = task_output.description[:100] + "..."
                else:
                     logger.warning("[Step 3 Crew Callback] Could not determine task description from task_output.")

                # Access raw_output safely
                if hasattr(task_output, 'raw_output') and isinstance(task_output.raw_output, str):
                    output_snippet = task_output.raw_output[:150].replace('\n', ' ') + "..."
                elif hasattr(task_output, 'output') and isinstance(task_output.output, str): # Check 'output' as fallback
                     output_snippet = task_output.output[:150].replace('\n', ' ') + "..."
                else:
                     output_snippet=str(task_output)[:150].replace('\n', ' ') + "..." # Fallback to string representation
                     logger.warning(f"[Step 3 Crew Callback] Could not determine raw output string from task_output.")

            else:
                # Log if the input is not a TaskOutput object at all
                logger.warning(f"[Step 3 Crew Callback] Received unexpected type for task_output: {type(task_output)}. Cannot extract details.")
                # Try to get a string representation for logging
                try:
                    output_snippet = str(task_output)[:150].replace('\n', ' ') + "..."
                except Exception:
                    output_snippet = "[Could not get string representation]"


            # --- Log Information ---
            logger.info(f"[Step 3 Crew Callback] Task Update:")
            logger.info(f"  - Agent: {agent_role}")
            logger.info(f"  - Task Desc (Start): {task_desc_snippet}")
            logger.info(f"  - Output (Start): {output_snippet}")

        except Exception as e:
            logger.error(f"[Step 3 Crew Callback] Error processing task output: {e}", exc_info=True)
            try:
                 # Try logging the raw object again for debugging if an error occurs
                 logger.debug(f"Raw task_output object during error: {task_output}")
            except Exception:
                 logger.debug("Could not log raw task_output object during error.")
