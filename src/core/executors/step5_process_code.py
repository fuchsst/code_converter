# src/core/executors/step5_process_code.py
import os
import json
from typing import Any, Dict, List, Optional, Type
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from ..remapping_logic import RemappingLogic
from ..tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator
from src.agents.code_processor import CodeProcessorAgent
from src.tasks.process_code import ProcessCodeTask
from crewai import Crew, Process, LLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

class Step5Executor(StepExecutor):
    """Executes Step 5: Iterative Conversion & Refinement."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]], # Corrected param name from base class update
                 tools: Dict[Type, Any],
                 remapping_logic: RemappingLogic):
        super().__init__(state_manager, context_manager, config, llm_configs, tools) # Use llm_configs
        self.remapping_logic = remapping_logic
        # Use GODOT_PROJECT_DIR as the target directory for output
        self.target_dir = os.path.abspath(config.get("GODOT_PROJECT_DIR", "data/godot_project"))
        self.analysis_dir = os.path.abspath(config.get("ANALYSIS_OUTPUT_DIR", "analysis_output"))
        self.task_item_max_retries = config.get("TASK_ITEM_MAX_RETRIES", 2) # Agent internal retries handled by CrewAI/LLM

    def execute(self, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        """
        Runs the iterative code processing for specified or all eligible packages.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process.
                                                If None, processes all eligible packages.
            force (bool): If True, forces reprocessing of packages even if already processed or failed.
            **kwargs: Not used directly but passed down.

        Returns:
            bool: True if processing was successful for the processed packages in this run, False otherwise.
        """
        logger.info(f"--- Starting Step 5 Execution: Process Code (Packages: {package_ids or 'All Eligible'}, Force={force}) ---")

        # --- Identify Eligible Packages ---
        target_status = 'mapping_defined'
        failed_status_prefixes = ['failed_processing', 'failed_remapping'] # Status prefixes indicating failure in this step
        processed_status = 'processed'

        # Use the updated _get_eligible_packages which considers the force flag for failed statuses
        eligible_packages_for_run = self._get_eligible_packages(
            target_status=target_status,
            specific_ids=package_ids,
            force=force # Pass force flag here
        )

        # --- Handle Force for 'processed' status ---
        # _get_eligible_packages currently only forces failed states. We need to handle forcing 'processed' state here.
        packages_to_process_this_run = []
        potential_target_package_ids = set() # Track all packages that *could* be processed

        all_packages = self.state_manager.get_all_packages()
        if not all_packages:
             logger.warning("No packages found in state. Cannot proceed with Step 5.")
             self.state_manager.update_workflow_status('step5_complete') # Or skipped status
             return True

        # Determine potential targets based on initial state and force flag
        for pkg_id, pkg_data in all_packages.items():
             current_status = pkg_data.get('status')
             is_target = (current_status == target_status)
             is_failed_this_step = any(current_status.startswith(prefix) for prefix in failed_status_prefixes)
             is_processed = (current_status == processed_status)

             matches_specific_request = (not package_ids or pkg_id in package_ids)

             if matches_specific_request and (is_target or (force and (is_failed_this_step or is_processed))):
                  potential_target_package_ids.add(pkg_id)

        # Filter the list from _get_eligible_packages and reset status if needed
        for pkg_id in eligible_packages_for_run:
             if pkg_id in potential_target_package_ids: # Ensure it's a potential target
                  pkg_data = all_packages[pkg_id]
                  current_status = pkg_data.get('status')
                  is_failed_this_step = any(current_status.startswith(prefix) for prefix in failed_status_prefixes)
                  is_processed = (current_status == processed_status)

                  if current_status == target_status:
                       packages_to_process_this_run.append(pkg_id)
                  elif force and (is_failed_this_step or is_processed):
                       logger.info(f"Force=True: Adding previously failed/processed package '{pkg_id}' (status: {current_status}) to process list for Step 5.")
                       # Reset status to target status before processing
                       self.state_manager.update_package_state(pkg_id, target_status, error=None) # Clear previous error
                       packages_to_process_this_run.append(pkg_id)

        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 5 run.")
            # Check if the overall step should be marked complete based on *potential* targets
            all_potential_targets_done_or_failed = True
            final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
            for pkg_id in potential_target_package_ids:
                 status = final_packages_state.get(pkg_id, {}).get('status')
                 # Terminal states for Step 5 are 'processed' or failed states for this step
                 if not (status == processed_status or any(status.startswith(prefix) for prefix in failed_status_prefixes)):
                      all_potential_targets_done_or_failed = False
                      break
            if all_potential_targets_done_or_failed and potential_target_package_ids:
                 logger.info("All potential target packages for Step 5 are now processed or failed.")
                 current_global_status = self.state_manager.get_state().get('workflow_status')
                 if not (current_global_status and 'failed' in current_global_status):
                      self.state_manager.update_workflow_status('step5_complete') # Or 'completed'
            return True # Indicate this specific invocation had nothing to fail on

        logger.info(f"Packages to process in this Step 5 run (Force={force}): {packages_to_process_this_run}")
        self.state_manager.update_workflow_status('running_step5')
        overall_success_this_run = True # Tracks success *of this specific run*

        # Get required tools using interfaces
        file_writer = self._get_tool(IFileWriter)
        file_replacer = self._get_tool(IFileReplacer)
        file_reader = self._get_tool(IFileReader) # Needed for reading target file for validation
        syntax_validator = self._get_tool(ISyntaxValidator)

        if not all([file_writer, file_replacer, file_reader, syntax_validator]):
            logger.error("One or more required tools (FileWriter, FileReplacer, FileReader, SyntaxValidator) are missing. Cannot execute Step 5.")
            self.state_manager.update_workflow_status('failed_step5', "Required tools missing.")
            return False

        # Get the LLM config and instantiate the LLM object
        generator_llm_config = self._get_llm_config('generator')
        generator_llm_instance = None
        if generator_llm_config:
            try:
                generator_llm_instance = LLM(**generator_llm_config)
                logger.info(f"Instantiated crewai.LLM for role 'generator': {generator_llm_config.get('model')}")
            except Exception as e:
                logger.error(f"Failed to instantiate crewai.LLM for role 'generator': {e}", exc_info=True)
        else:
            logger.error("Generator LLM configuration ('generator') not found.")

        if not generator_llm_instance:
             logger.error("Generator LLM instance could not be created. Cannot execute Step 5.")
             self.state_manager.update_workflow_status('failed_step5', "Generator LLM not configured or failed to instantiate.")
             return False

        for pkg_id in packages_to_process_this_run:
            logger.info(f"Processing Step 5 for package: {pkg_id}")
            self.state_manager.update_package_state(pkg_id, status='running_processing')
            package_success = True # Track success per package

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                tasks_artifact = pkg_info.get('artifacts', {}).get('tasks_json')
                if not tasks_artifact:
                    # If forcing, maybe try the non-suffixed version?
                    if force:
                         tasks_artifact = f"package_{pkg_id}_tasks.json"
                         logger.warning(f"Remapped tasks artifact not found for forced run of {pkg_id}, trying default: {tasks_artifact}")
                         # Check if default exists before raising error
                         tasks_json_path_check = os.path.join(self.analysis_dir, tasks_artifact)
                         if not os.path.exists(tasks_json_path_check):
                              raise FileNotFoundError(f"Task list artifact missing for package {pkg_id} (and default not found for force run).")
                    else:
                         raise FileNotFoundError(f"Task list artifact missing for package {pkg_id}.")


                tasks_json_path = os.path.join(self.analysis_dir, tasks_artifact)
                if not os.path.exists(tasks_json_path):
                     raise FileNotFoundError(f"Task list file not found: {tasks_json_path}")

                with open(tasks_json_path, 'r', encoding='utf-8') as f:
                    task_list = json.load(f) # This is the list of tasks

                if not isinstance(task_list, list):
                     raise TypeError(f"Task list loaded from {tasks_json_path} is not a list.")

                # --- Task-by-Task Loop ---
                task_results = [] # Store results for each task item
                failed_task_details_for_remapping = [] # Store details only for remapping check

                for i, task_item in enumerate(task_list):
                    task_id = task_item.get('task_id', f'task_{i+1}') # Use index if ID missing
                    logger.info(f"--- Processing Task Item {i+1}/{len(task_list)} (ID: {task_id}) for Package {pkg_id} ---")
                    task_report: Dict[str, Any] = {} # Initialize report for this task

                    try:
                        # --- Context Assembly for Single Task ---
                        target_godot_file_rel = task_item.get('target_godot_file')
                        existing_godot_content = None
                        if target_godot_file_rel:
                            target_godot_file_abs = os.path.join(self.target_dir, target_godot_file_rel)
                            if os.path.exists(target_godot_file_abs):
                                logger.debug(f"Reading existing target Godot file for task context: {target_godot_file_abs}")
                                # Use the IFileReader tool wrapper
                                read_result = file_reader.read(target_godot_file_abs)
                                if read_result.get('status') == 'success':
                                    existing_godot_content = read_result.get('content')
                                else:
                                    logger.warning(f"Failed to read existing target file {target_godot_file_abs} via tool: {read_result.get('message')}. Context may be incomplete.")

                        primary_files = pkg_info.get('files', [])
                        # Determine dependencies using the existing method on ContextManager
                        dependency_files = []
                        if hasattr(self.context_manager, '_get_dependencies_for_package'):
                            dependency_files = self.context_manager._get_dependencies_for_package(primary_files)
                        else:
                            # This case should ideally not happen if ContextManager is correctly initialized
                            logger.error(f"Critical: Method '_get_dependencies_for_package' not found on ContextManager. Cannot determine dependencies for {pkg_id}.")

                        task_context = self.context_manager.get_context_for_step(
                            step_name=f"CODE_PROCESSING_TASK_{task_id}",
                            primary_relative_paths=primary_files,
                            dependency_relative_paths=dependency_files,
                            work_package_id=pkg_id,
                            task_item_details=task_item,
                            existing_target_godot_file_content={target_godot_file_rel: existing_godot_content} if target_godot_file_rel and existing_godot_content else None
                        )

                        if not task_context:
                            raise ValueError("Failed to assemble context for task.")

                        # --- Agent Invocation ---
                        agent = CodeProcessorAgent().get_agent(llm_instance=generator_llm_instance)
                        task_item_json_str = json.dumps(task_item)
                        process_task = ProcessCodeTask().create_task(agent, task_context, task_item_json_str)

                        # Create and run Crew
                        crew = Crew(
                            agents=[agent],
                            tasks=[process_task],
                            llm=generator_llm_instance,
                            process=Process.sequential,
                            verbose=False
                        )
                        logger.info(f"Kicking off Crew for Task Item {task_id}...")
                        agent_report = crew.kickoff() # Agent handles internal retries
                        logger.info(f"Crew finished for Task Item {task_id}.")

                        # --- Process Agent Report ---
                        if not isinstance(agent_report, dict):
                            logger.error(f"Unexpected report type from Crew: {type(agent_report)}. Expected dict.")
                            if isinstance(agent_report, str):
                                try: agent_report = json.loads(agent_report)
                                except Exception: raise ValueError("Agent report was not a valid JSON object string.")
                            else: raise ValueError("Agent report was not a JSON object or valid JSON string.")

                        # Initialize task_report with agent's output
                        task_report = agent_report.copy()
                        if 'task_id' not in task_report: task_report['task_id'] = task_id # Ensure task_id is present

                        # --- File Operation (using Tool Wrappers) ---
                        file_op_status = 'skipped'
                        file_op_message = None
                        generated_code = task_report.get('generated_code')
                        output_format = task_report.get('output_format')
                        target_file_rel = task_report.get('target_godot_file')
                        target_file_abs = os.path.join(self.target_dir, target_file_rel) if target_file_rel else None

                        if task_report.get('status') == 'completed' and target_file_abs and generated_code is not None:
                            if output_format == 'FULL_FILE':
                                write_result = file_writer.write(target_file_abs, generated_code)
                                file_op_status = write_result.get('status', 'failure')
                                file_op_message = write_result.get('message')
                            elif output_format == 'CODE_BLOCK':
                                search_block = task_report.get('search_block')
                                if search_block:
                                    # Construct diff string for the tool wrapper
                                    diff_str = f"<<<<<<< SEARCH\n{search_block}\n=======\n{generated_code}\n>>>>>>> REPLACE"
                                    replace_result = file_replacer.replace(target_file_abs, diff_str)
                                    file_op_status = replace_result.get('status', 'failure')
                                    file_op_message = replace_result.get('message')
                                else:
                                    file_op_status = 'failure'
                                    file_op_message = "Agent reported CODE_BLOCK but search_block was missing."
                                    logger.error(f"Task {task_id}: Agent requested CODE_BLOCK but missing search_block.")
                            else:
                                file_op_status = 'failure'
                                file_op_message = f"Unknown output_format: {output_format}"
                                logger.error(f"Task {task_id}: Unknown output_format '{output_format}'.")
                        elif task_report.get('status') != 'completed':
                             file_op_message = "Skipped due to agent failure."
                        else:
                             file_op_message = "Skipped due to missing target file or generated code."


                        task_report['file_operation_status'] = file_op_status
                        task_report['file_operation_message'] = file_op_message

                        # --- Post-Operation Validation ---
                        orch_validation_status = 'skipped'
                        orch_validation_errors = None
                        if file_op_status == 'success' and target_file_abs:
                            # Read the potentially modified content
                            read_result_val = file_reader.read(target_file_abs)
                            if read_result_val.get('status') == 'success':
                                modified_content = read_result_val.get('content')
                                if modified_content is not None:
                                     val_result = syntax_validator.validate(modified_content)
                                     orch_validation_status = val_result.get('status', 'failure')
                                     orch_validation_errors = val_result.get('errors')
                                     if orch_validation_status == 'failure':
                                          logger.warning(f"Orchestrator validation failed for task {task_id}. Errors: {orch_validation_errors}")
                                else:
                                     orch_validation_status = 'failure'
                                     orch_validation_errors = "Failed to read content after successful write/replace."
                                     logger.error(f"Task {task_id}: Could not read content from {target_file_abs} after successful file op.")
                            else:
                                 orch_validation_status = 'failure'
                                 orch_validation_errors = f"Failed to read file for validation: {read_result_val.get('message')}"
                                 logger.error(f"Task {task_id}: Failed to read {target_file_abs} for validation: {read_result_val.get('message')}")

                        task_report['orchestrator_validation_status'] = orch_validation_status
                        task_report['orchestrator_validation_errors'] = orch_validation_errors

                        # --- Final Task Status Determination ---
                        if task_report.get('status') == 'completed' and file_op_status == 'success' and orch_validation_status != 'failure':
                            task_report['status'] = 'completed' # Confirm completion
                            logger.info(f"Task Item {task_id} completed successfully.")
                        else:
                            task_report['status'] = 'failed' # Mark as failed if agent failed, file op failed, or validation failed
                            package_success = False
                            failed_task_details_for_remapping.append(task_report) # Add to list for remapping check
                            logger.error(f"Task Item {task_id} failed. Agent Status: {agent_report.get('status')}, File Op: {file_op_status}, Validation: {orch_validation_status}. Error: {task_report.get('error_message') or file_op_message or orch_validation_errors}")

                    except Exception as task_err:
                        logger.error(f"Critical error processing task item {task_id} for package {pkg_id}: {task_err}", exc_info=True)
                        # Ensure a basic failure report is created
                        task_report = {
                            'task_id': task_id,
                            'status': 'failed',
                            'error_message': f"Executor error: {task_err}",
                            'target_godot_file': task_item.get('target_godot_file'),
                            'target_element': task_item.get('target_element'),
                            'file_operation_status': 'skipped',
                            'orchestrator_validation_status': 'skipped'
                        }
                        package_success = False
                        failed_task_details_for_remapping.append(task_report)
                    finally:
                        task_results.append(task_report) # Append the final report for this task

                # --- Post-Loop Processing for Package ---
                package_report_filename = f"package_{pkg_id}_task_results.json"
                package_report_path = os.path.join(self.analysis_dir, package_report_filename)
                try:
                    os.makedirs(self.analysis_dir, exist_ok=True)
                    with open(package_report_path, 'w', encoding='utf-8') as f:
                        json.dump(task_results, f, indent=2)
                    logger.info(f"Saved consolidated task results report: {package_report_path}")
                    # Update status only after saving report, keep artifacts key consistent
                    self.state_manager.update_package_state(pkg_id, status='processing_report_generated', artifacts={'task_results_report': package_report_filename})
                except IOError as e:
                    logger.error(f"Failed to save consolidated task results report {package_report_path}: {e}")
                    # Continue processing status update despite save failure, maybe set a specific error status?
                    # For now, we proceed to remapping check even if report save failed.

                # --- Remapping Check ---
                if not package_success:
                    overall_success_this_run = False # Mark overall run success as False if any package fails
                    current_remapping_attempts = pkg_info.get('remapping_attempts', 0)
                    max_remap = self.config.get("MAX_REMAPPING_ATTEMPTS", 1)

                    if current_remapping_attempts >= max_remap:
                         logger.warning(f"Max remapping attempts ({max_remap}) reached for package {pkg_id}. Marking as failed.")
                         self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remapping attempts reached.")
                    elif self.remapping_logic.should_remap_package(failed_task_details_for_remapping):
                        logger.info(f"Detected pattern of failures suggesting mapping issues for package {pkg_id}. Triggering remapping (Attempt {current_remapping_attempts + 1}).")
                        # Set status to 'needs_remapping'. The Orchestrator's resume logic will handle calling Step 4 again.
                        self.state_manager.update_package_state(pkg_id, status='needs_remapping', error="Code application/validation failures suggest mapping issues.")
                        # We consider triggering remapping a "successful" outcome for this Step 5 run,
                        # as it prevents marking the whole pipeline as failed immediately.
                        # The remapping attempt itself might fail later.
                        # Keep overall_success_this_run as True if remapping is triggered
                    else:
                        logger.info(f"Failures detected for package {pkg_id}, but remapping condition not met or limit reached.")
                        self.state_manager.update_package_state(pkg_id, status='failed_processing', error="File operations or validation failed for some tasks.")
                else:
                     logger.info(f"All tasks processed successfully for package {pkg_id}.")
                     self.state_manager.update_package_state(pkg_id, status='processed')

            except Exception as e: # Catch errors in the main package loop (e.g., loading task list)
                logger.error(f"A critical error occurred during Step 5 setup/loop for package {pkg_id}: {e}", exc_info=True)
                self.state_manager.update_package_state(pkg_id, status='failed_processing', error=f"Critical executor error: {e}")
                overall_success_this_run = False

        # --- Final Workflow Status Check ---
        all_potential_targets_done_or_failed = True
        final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
        for pkg_id in potential_target_package_ids: # Check against potential targets determined at start
             status = final_packages_state.get(pkg_id, {}).get('status')
             # Terminal states for Step 5 are 'processed' or failed states for this step or remapping triggered
             if not (status == processed_status or status == 'needs_remapping' or any(status.startswith(prefix) for prefix in failed_status_prefixes)):
                  all_potential_targets_done_or_failed = False
                  logger.debug(f"Package {pkg_id} is still pending Step 5 completion (status: {status}).")
                  break

        if all_potential_targets_done_or_failed and potential_target_package_ids:
             # Check if any package still needs remapping
             needs_remapping_pending = any(p.get('status') == 'needs_remapping' for p in final_packages_state.values() if p.get('id') in potential_target_package_ids) # Check only potential targets
             if not needs_remapping_pending:
                  logger.info("All potential target packages for Step 5 are now processed or failed.")
                  current_global_status = self.state_manager.get_state().get('workflow_status')
                  if not (current_global_status and 'failed' in current_global_status):
                       self.state_manager.update_workflow_status('step5_complete') # Or 'completed'
             else:
                  logger.info("Step 5 finished processing available packages, but some require remapping.")
                  # Keep status as 'running_step5' or let orchestrator handle 'needs_remapping' status
        elif not overall_success_this_run:
             # If any package failed *in this specific run* and didn't trigger remapping
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  self.state_manager.update_workflow_status('failed_step5', "One or more packages failed during code processing in the latest run.")

        logger.info(f"--- Finished Step 5 Execution Run (Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run
