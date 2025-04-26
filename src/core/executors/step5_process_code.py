# src/core/executors/step5_process_code.py
import os
import json
from typing import Any, Dict, List, Optional, Type
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from ..remapping_logic import RemappingLogic
# Standard library imports
import os
import json
from typing import Any, Dict, List, Optional, Type

# CrewAI imports
from crewai import Crew, Process, Task, Agent

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from ..remapping_logic import RemappingLogic
from ..tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator # Keep interfaces for tool wrappers if needed

# Import NEW agents
from src.agents.step5.code_generator import get_code_generator_agent
from src.agents.step5.syntax_validator import get_syntax_validation_agent
from src.agents.step5.code_refiner import get_code_refinement_agent
from src.agents.step5.file_manager import get_file_manager_agent
from src.agents.step5.remapping_advisor import get_remapping_advisor_agent

# Import NEW CrewAI tools (wrappers)
from src.tools.crewai_tools import (
    FileWriterTool,
    FileReplacerTool,
    GodotSyntaxValidatorTool,
    RemappingLogicTool
)

# Import NEW Task definitions
from src.tasks.step5.process_code import (
    HierarchicalProcessTaskItem,
    AnalyzePackageFailuresTask,
    RemappingAdvice, # Import Pydantic model for parsing result
    TaskItemProcessingResult # Import Pydantic model for parsing result
)

# Import utilities
from src.utils.json_utils import parse_json_from_string # Helper for parsing final JSON output
from src.logger_setup import get_logger
# Import TaskOutput for type hinting the callback parameter
from crewai.tasks.task_output import TaskOutput

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
        super().__init__(state_manager, context_manager, config, llm_configs, tools)
        self.remapping_logic = remapping_logic
        self.target_dir = os.path.abspath(config.get("GODOT_PROJECT_DIR", "data/godot_project"))
        self.analysis_dir = os.path.abspath(config.get("ANALYSIS_OUTPUT_DIR", "analysis_output"))
        # task_item_max_retries is now handled internally by CrewAI agent/task config if needed

        # --- Instantiate CrewAI Tools ---
        # These tools wrap the actual implementation logic
        self.file_writer_tool = FileWriterTool()
        self.file_replacer_tool = FileReplacerTool()
        self.syntax_validator_tool = GodotSyntaxValidatorTool()
        self.remapping_tool = RemappingLogicTool()
        logger.info("Instantiated CrewAI tools for Step 5.")


        # --- Instantiate Task Creators ---
        self.item_task_creator = HierarchicalProcessTaskItem()
        self.analysis_task_creator = AnalyzePackageFailuresTask()
        logger.info("Instantiated Task creators for Step 5.")

        # --- Get LLM Instances (Consider moving agent creation here if LLMs are ready) ---
        # Get LLM instances using specific roles defined in Orchestrator's map
        # The _get_llm_instance_by_role method handles fetching the correct config based on the role name.
        self.generator_llm_instance = self._get_llm_instance_by_role('generator')
        self.validator_llm_instance = self._get_llm_instance_by_role('validator')
        self.refiner_llm_instance = self._get_llm_instance_by_role('refiner', fallback_llm=self.generator_llm_instance)
        self.file_manager_llm_instance = self._get_llm_instance_by_role('file_manager')
        self.remapping_llm_instance = self._get_llm_instance_by_role('remapping_advisor')
        self.manager_llm_instance = self._get_llm_instance_by_role('manager')

        # Check crucial instances AFTER attempting instantiation
        if not self.generator_llm_instance or not self.manager_llm_instance:
             # Manager and Generator are crucial (Refiner can fallback)
             logger.error("Missing critical LLM instances (Generator or Manager). Step 5 cannot proceed.")
             raise ValueError("Missing critical LLM configurations for Step 5.")

        # --- Instantiate Agents (Pass LLMs and Tools) ---
        # Use the specific LLM instances fetched above
        self.code_generator_agent = get_code_generator_agent(self.generator_llm_instance)
        self.syntax_validator_agent = get_syntax_validation_agent(self.validator_llm_instance, tools=[self.syntax_validator_tool])
        self.code_refinement_agent = get_code_refinement_agent(self.refiner_llm_instance)
        self.file_manager_agent = get_file_manager_agent(self.file_manager_llm_instance, tools=[self.file_writer_tool, self.file_replacer_tool])
        self.remapping_advisor_agent = get_remapping_advisor_agent(self.remapping_llm_instance, tools=[self.remapping_tool])

        # Log if any LLM instance is missing (should be caught by verification below)
        if not self.generator_llm_instance: logger.warning("Step5 Generator LLM instance is missing.")
        if not self.validator_llm_instance: logger.warning("Step5 Validator LLM instance is missing.")
        if not self.refiner_llm_instance: logger.warning("Step5 Refiner LLM instance is missing.")
        if not self.file_manager_llm_instance: logger.warning("Step5 File Manager LLM instance is missing.")
        if not self.remapping_llm_instance: logger.warning("Step5 Remapping Advisor LLM instance is missing.")

        logger.info("Instantiated CrewAI agents for Step 5.")

    def _get_llm_instance_by_role(self, role: str, fallback_llm: Optional[Any] = None) -> Optional[Any]:
        """
        Helper to get and instantiate an LLM object for a specific role,
        potentially falling back to another role's instance.
        Returns an instantiated LLM object or None.
        """
        llm_config = self._get_llm_config(role) # Fetches config dict from Orchestrator's map

        # Use fallback LLM *instance* if config is missing for the current role
        if not llm_config:
            logger.warning(f"LLM config for role '{role}' not found.")
            if fallback_llm:
                logger.warning(f"Using fallback LLM instance for role '{role}'. Type: {type(fallback_llm)}")
                return fallback_llm # Return the already instantiated fallback LLM
            else:
                logger.error(f"No config for role '{role}' and no fallback LLM provided.")
                return None

        if not isinstance(llm_config, dict):
             logger.error(f"Invalid LLM config dictionary type for role '{role}'. Expected dict, got {type(llm_config)}")
             return None

        # Instantiate based on the llm_config dictionary
        try:
             model_identifier = llm_config.get("model")
             if not model_identifier:
                 logger.error(f"Model identifier missing in LLM config for role '{role}'. Config: {llm_config}")
                 return None

             logger.debug(f"Attempting to instantiate LLM for role '{role}' using model '{model_identifier}' with config: {llm_config}")

             # Import necessary classes locally
             from src.llms.google_genai_llm import GoogleGenAI_LLM
             from crewai import LLM as CrewAI_LLM
             import src.config as global_config

             # Prepare arguments explicitly
             common_args = {
                 "model": model_identifier,
                 "temperature": llm_config.get("temperature"),
                 "top_p": llm_config.get("top_p"),
                 "top_k": llm_config.get("top_k"),
                 # Add other common parameters accepted by both constructors if any
             }
             # Filter out None values
             common_args = {k: v for k, v in common_args.items() if v is not None}

             if model_identifier.startswith(("gemini/", "google/")):
                 gemini_args = common_args.copy()
                 gemini_args['timeout'] = llm_config.get('timeout', global_config.GEMINI_TIMEOUT)
                 if llm_config.get('api_key'):
                     gemini_args['api_key'] = llm_config['api_key']
                 # Add schema/mime_type if needed for specific roles (e.g., formatter)
                 # if role == 'formatter':
                 #     gemini_args['response_schema'] = GodotStructureOutput # Example
                 #     gemini_args['response_mime_type'] = "application/json" # Example

                 llm_instance = GoogleGenAI_LLM(**gemini_args)
                 logger.info(f"Successfully instantiated GoogleGenAI_LLM for role '{role}': {model_identifier}")
             else:
                 # For default CrewAI LLM, pass only common args.
                 # It might rely on environment variables for API keys (e.g., OPENAI_API_KEY)
                 llm_instance = CrewAI_LLM(**common_args)
                 logger.info(f"Successfully instantiated default crewai.LLM for role '{role}': {model_identifier}")

             return llm_instance

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
        overall_success_this_run = True # Tracks success *of this specific execution run*

        # Ensure critical LLMs (manager, generator) are available
        if not self.manager_llm_instance or not self.generator_llm_instance:
             logger.error("Manager or Generator LLM instance is missing. Cannot execute Step 5.")
             self.state_manager.update_workflow_status('failed_step5', "Critical LLM instances missing.")
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

                # --- Assemble Context for the Entire Package ---
                # This context will be shared across tasks within the crew for this package
                primary_files = pkg_info.get('files', [])
                dependency_files = self.context_manager._get_dependencies_for_package(primary_files) if hasattr(self.context_manager, '_get_dependencies_for_package') else []
                # TODO: Decide if existing Godot file content needs to be pre-loaded into context
                # or if the FileManagerAgent should read it if needed for replacement search blocks.
                # Pre-loading might hit token limits for large packages/files.
                # Let's assume for now the manager/generator agent will handle extracting search blocks
                # from context if the existing code is provided there.
                package_context_str = self.context_manager.get_context_for_step(
                    step_name=f"PACKAGE_PROCESSING_{pkg_id}",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    work_package_id=pkg_id
                    # Add other relevant package-level info if needed
                )
                if not package_context_str:
                     logger.warning(f"Failed to assemble base context for package {pkg_id}. Crew might lack information.")
                     # Decide if this is critical enough to skip the package
                     # continue

                # --- Create CrewAI Tasks for the Package ---
                crew_tasks = self._create_crewai_tasks_for_package(task_list, package_context_str)
                if not crew_tasks:
                     logger.error(f"Failed to create CrewAI tasks for package {pkg_id}. Skipping package.")
                     self.state_manager.update_package_state(pkg_id, status='failed_processing', error="Failed to create internal tasks.")
                     overall_success_this_run = False
                     continue

                # --- Create and Run Hierarchical Crew ---
                package_crew = Crew(
                    agents=[ # List ALL worker agents
                        self.code_generator_agent,
                        self.syntax_validator_agent,
                        self.code_refinement_agent,
                        self.file_manager_agent,
                        self.remapping_advisor_agent
                    ],
                    tasks=crew_tasks, # Tasks for the manager to orchestrate
                    process=Process.hierarchical,
                    # Assign only the essential manager_llm for hierarchical process
                    manager_llm=self.manager_llm_instance,
                    # Removed explicit chat_llm, planning_llm, function_calling_llm
                    # manager_agent=self.manager_agent, # Or provide a custom manager agent instance
                    memory=True, # Enable memory for context persistence within the package run
                    # planning=True, # Consider enabling Crew planning feature
                    verbose=True,
                    task_callback=self._log_step5_task_completion # Add the callback
                    # step_callback=self.log_step, # Optional callbacks for detailed logging
                )

                logger.info(f"Kicking off Hierarchical Crew for Package {pkg_id} with {len(task_list)} items...")
                # Pass package-level inputs if needed by tasks (e.g., package ID)
                crew_inputs = {'package_id': pkg_id}
                crew_result = package_crew.kickoff(inputs=crew_inputs)
                logger.info(f"Hierarchical Crew finished for Package {pkg_id}.")
                logger.debug(f"Crew Result Raw Output:\n{crew_result}") # Log raw output for debugging

                # --- Process Crew Result ---
                # The final result should ideally be the output of the AnalyzePackageFailuresTask
                final_output_str = crew_result # Crew.kickoff() often returns the last task's output directly
                parsed_advice = None
                package_processing_errors = [] # Collect errors from individual task items if possible

                # Attempt to parse the final output as RemappingAdvice JSON
                if isinstance(final_output_str, str):
                    parsed_json = parse_json_from_string(final_output_str)
                    if parsed_json and isinstance(parsed_json, dict):
                         try:
                              parsed_advice = RemappingAdvice(**parsed_json)
                              logger.info(f"Successfully parsed RemappingAdvice from crew output for {pkg_id}.")
                         except Exception as pydantic_error:
                              logger.error(f"Failed to validate final crew output as RemappingAdvice for {pkg_id}: {pydantic_error}")
                              logger.debug(f"Final output string was: {final_output_str}")
                    else:
                         logger.error(f"Final crew output for {pkg_id} was not a parseable JSON object.")
                         logger.debug(f"Final output string was: {final_output_str}")
                elif isinstance(final_output_str, dict): # If kickoff returns a dict
                     try:
                          parsed_advice = RemappingAdvice(**final_output_str)
                          logger.info(f"Successfully parsed RemappingAdvice from crew output dict for {pkg_id}.")
                     except Exception as pydantic_error:
                          logger.error(f"Failed to validate final crew output dict as RemappingAdvice for {pkg_id}: {pydantic_error}")
                          logger.debug(f"Final output dict was: {final_output_str}")
                else:
                     logger.error(f"Unexpected final crew output type for {pkg_id}: {type(final_output_str)}")


                # Determine overall package success - needs refinement.
                # Option 1: Rely solely on RemappingAdvisor output (if it ran successfully).
                # Option 2: Inspect individual task results within the crew's execution report (if accessible).
                # For now, let's assume failure if we couldn't get valid remapping advice,
                # or if the advice explicitly mentions failures. A more robust check might be needed.
                if parsed_advice is None:
                     package_success = False
                     logger.error(f"Package {pkg_id} processing failed: Could not determine remapping advice from crew output.")
                     # Try to find specific errors in the crew's task outputs if available
                     # This part depends on how CrewAI exposes task results in hierarchical mode.
                     # Placeholder: Check crew.usage_metrics or task outputs if they exist
                     # if hasattr(package_crew, 'tasks') and package_crew.tasks:
                     #      for task_run in package_crew.tasks:
                     #           if hasattr(task_run, 'output') and 'failed' in str(task_run.output).lower():
                     #                package_processing_errors.append(f"Task '{task_run.description[:50]}...' potentially failed.")

                else:
                     # If advice was parsed, assume the crew completed. Success depends on whether remapping is needed.
                     # If remapping is recommended, the package technically "failed" this processing attempt.
                     package_success = not parsed_advice.recommend_remapping

                # Store detailed results (raw output or parsed task results) as an artifact
                package_report_filename = f"package_{pkg_id}_crew_results.json"
                package_report_path = os.path.join(self.analysis_dir, package_report_filename)
                try:
                    os.makedirs(self.analysis_dir, exist_ok=True)
                    # Save the raw result or structure if available
                    report_content = crew_result if isinstance(crew_result, (dict, list)) else {'raw_output': str(crew_result)}
                    # TODO: Ideally, save structured results of *all* tasks if accessible from crew object
                    with open(package_report_path, 'w', encoding='utf-8') as f:
                        json.dump(report_content, f, indent=2)
                    logger.info(f"Saved crew results report: {package_report_path}")
                    self.state_manager.update_package_state(pkg_id, status='processing_report_generated', artifacts={'crew_results_report': package_report_filename})
                except IOError as e:
                    logger.error(f"Failed to save crew results report {package_report_path}: {e}")

                # --- Remapping Check based on Crew Result ---
                if not package_success:
                    overall_success_this_run = False # Mark overall run as failed if any package fails *and* doesn't trigger remapping
                    error_reason = "Package processing failed within the crew."
                    if parsed_advice and parsed_advice.reason:
                         error_reason = parsed_advice.reason # Use reason from advisor if available

                    current_remapping_attempts = pkg_info.get('remapping_attempts', 0)
                    max_remap = self.config.get("MAX_REMAPPING_ATTEMPTS", 1)

                    # Check if max attempts reached
                    if current_remapping_attempts >= max_remap:
                        logger.warning(f"Max remapping attempts ({max_remap}) reached for package {pkg_id}. Marking as failed_remapping.")
                        self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remapping attempts reached. Last error: {error_reason}")
                    # Check if remapping is recommended by the advisor tool (via parsed_advice)
                    elif parsed_advice and parsed_advice.recommend_remapping:
                        logger.info(f"Remapping recommended for package {pkg_id} by RemappingAdvisorAgent (Attempt {current_remapping_attempts + 1}). Reason: {parsed_advice.reason}")
                        # Set status to 'needs_remapping'. Orchestrator handles calling Step 4 again.
                        self.state_manager.update_package_state(pkg_id, status='needs_remapping', error=f"Remapping recommended: {parsed_advice.reason}", increment_remap_attempt=True)
                        # Don't mark overall_success_this_run as False if remapping is triggered
                    else:
                        # Failures occurred, but remapping not recommended or advice parsing failed
                        logger.info(f"Failures detected for package {pkg_id}, but remapping not recommended or advisor failed.")
                        final_error = error_reason if error_reason else "Package processing failed, but remapping condition not met or advisor failed."
                        if package_processing_errors:
                             final_error += f" Specific errors: {'; '.join(package_processing_errors)}"
                        self.state_manager.update_package_state(pkg_id, status='failed_processing', error=final_error)
                else:
                    # Package processed successfully according to crew result analysis
                    logger.info(f"Hierarchical crew processed package {pkg_id} successfully.")
                    self.state_manager.update_package_state(pkg_id, status='processed')

            except Exception as e: # Catch errors during package setup or crew execution kickoff
                logger.error(f"A critical error occurred during Step 5 hierarchical processing for package {pkg_id}: {e}", exc_info=True)
                self.state_manager.update_package_state(pkg_id, status='failed_processing', error=f"Critical executor error during crew setup/kickoff: {e}")
                overall_success_this_run = False

        # --- Final Workflow Status Check (Remains largely the same) ---
        # ... (keep existing logic for checking if all potential targets are done/failed/remapping) ...
        # This logic correctly checks for terminal states including 'needs_remapping'
        # --- Start of existing final check logic ---
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
             needs_remapping_pending = any(p.get('status') == 'needs_remapping' for p_id, p in final_packages_state.items() if p_id in potential_target_package_ids) # Check only potential targets
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
        # --- End of existing final check logic ---


        logger.info(f"--- Finished Step 5 Execution Run (Overall Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run

    # --- Helper Method to Create Tasks ---
    def _create_crewai_tasks_for_package(self, task_items: List[Dict[str, Any]], package_context_str: str) -> List[Task]:
        """Creates the list of CrewAI tasks for a work package."""
        crew_tasks: List[Task] = []
        item_processing_tasks: List[Task] = [] # Keep track of item tasks for final analysis context

        # Ensure task creators are initialized (should be done in __init__)
        if not hasattr(self, 'item_task_creator') or not hasattr(self, 'analysis_task_creator'):
             logger.error("Task creators not initialized in Step5Executor.")
             return []

        for task_item in task_items:
            # Create task for the manager to process this item
            # Note: manager_agent is implicit in hierarchical process, task guides it.
            # Dependencies between items are complex; let manager handle sequence for now.
            item_task = self.item_task_creator.create_task(
                manager_agent=None, # Manager is implicit
                task_item_details=task_item,
                package_context=package_context_str,
                dependent_tasks=None # Let manager handle sequence/context flow
            )
            crew_tasks.append(item_task)
            item_processing_tasks.append(item_task) # Add to list for final task context

        # Add the final analysis task that depends on all item tasks
        if item_processing_tasks: # Only add if there were items to process
             analysis_task = self.analysis_task_creator.create_task(
                 advisor_agent=self.remapping_advisor_agent,
                 all_item_results_context=item_processing_tasks # Pass the list of item tasks
             )
             crew_tasks.append(analysis_task)
        else:
             logger.warning("No task items found for package, skipping analysis task creation.")

        logger.info(f"Created {len(crew_tasks)} CrewAI tasks for the package ({len(item_processing_tasks)} item tasks, {1 if item_processing_tasks else 0} analysis task).")
        return crew_tasks

    # --- Callback Method ---
    def _log_step5_task_completion(self, task_output: TaskOutput):
        """Logs information about each completed task within the Step 5 crew."""
        try:
            agent_role = task_output.agent.role if task_output.agent else "Unknown Agent"
            task_desc_snippet = task_output.task.description[:100] + "..." if task_output.task else "Unknown Task"
            output_snippet = task_output.raw_output[:150].replace('\n', ' ') + "..." if task_output.raw_output else "No output"

            logger.info(f"[Step 5 Crew Callback] Task Completed:")
            logger.info(f"  - Agent: {agent_role}")
            logger.info(f"  - Task Desc (Start): {task_desc_snippet}")
            logger.info(f"  - Output (Start): {output_snippet}")
            # Avoid state updates here, focus on logging
        except Exception as e:
            logger.error(f"[Step 5 Crew Callback] Error processing task output: {e}", exc_info=True)
            # Log the raw object if possible for debugging
            try:
                 logger.debug(f"Raw task_output object: {task_output}")
            except Exception:
                 logger.debug("Could not log raw task_output object.")
