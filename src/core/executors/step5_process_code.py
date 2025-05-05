# src/core/executors/step5_process_code.py
import os
import json
from typing import Any, Dict, List, Optional, Type
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens, read_godot_file_content
from ...tools.remapping_logic import RemappingLogic
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
from ...tools.remapping_logic import RemappingLogic
from ..tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator # Keep interfaces for tool wrappers if needed

# Import NEW agents
from src.agents.step5.code_generator import get_code_generator_agent
from src.agents.step5.syntax_validator import get_project_validation_agent
from src.agents.step5.code_refiner import get_code_refinement_agent
from src.agents.step5.file_manager import get_file_manager_agent
from src.agents.step5.remapping_advisor import get_remapping_advisor_agent

# Import NEW CrewAI tools (wrappers)
from src.tools.crewai_tools import (
    FileWriterTool,
    FileReplacerTool,
    GodotProjectValidatorTool,
    RemappingLogicTool,
    FileReaderTool
)

# Import NEW Task definitions
from src.tasks.step5.process_code import (
    create_hierarchical_process_taskitem_task,
    create_analyze_package_failures_task,
    RemappingAdvice, # Import Pydantic model for parsing result
    TaskItemProcessingResult # Import Pydantic model for parsing result
)

# Import utilities
from src.utils.json_utils import parse_json_from_string # Helper for parsing final JSON output
from src.logger_setup import get_logger
import src.config as global_config # Use alias
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
        logger.info("Step5Executor initialized.")


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

        # --- Instantiate LLMs, Tools, and Agents ---
        # Tools need to be instantiated here as they might have dependencies or state
        file_writer_tool = FileWriterTool()
        file_replacer_tool = FileReplacerTool()
        project_validator_tool = GodotProjectValidatorTool()
        file_reader_tool = FileReaderTool()
        remapping_tool = RemappingLogicTool()
        logger.info("Instantiated CrewAI tools for Step 5.")

        # Instantiate LLMs using the helper method
        generator_llm_instance = self._create_llm_instance('GENERATOR_REFINER_MODEL')
        validator_llm_instance = self._create_llm_instance('UTILITY_MODEL')
        refiner_llm_instance = self._create_llm_instance('GENERATOR_REFINER_MODEL')
        file_manager_llm_instance = self._create_llm_instance('UTILITY_MODEL')
        remapping_llm_instance = self._create_llm_instance('ANALYZER_MODEL', response_schema_class=RemappingAdvice)
        manager_llm_instance = self._create_llm_instance('MANAGER_MODEL')

        # Check crucial instances
        required_llms = {
            'Generator/Refiner': generator_llm_instance, # Check the primary generator instance
            'Manager': manager_llm_instance,
            'Validator': validator_llm_instance,
            'File Manager': file_manager_llm_instance,
            'Remapping Advisor': remapping_llm_instance
        }
        missing_llms = [name for name, llm in required_llms.items() if not llm]
        if missing_llms:
             logger.error(f"Missing critical LLM instances for Step 5: {', '.join(missing_llms)}. Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step5', f"Missing LLM config for: {', '.join(missing_llms)}")
             return False
        # Ensure refiner has a valid instance (either its own or the fallback)
        if not refiner_llm_instance:
             logger.error("Critical error: Refiner LLM could not be instantiated (including fallback). Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step5', "Refiner LLM instance missing.")
             return False


        # Instantiate Agents
        code_generator_agent = get_code_generator_agent(generator_llm_instance, tools=[file_reader_tool])
        project_validator_agent = get_project_validation_agent(validator_llm_instance, tools=[project_validator_tool])
        code_refinement_agent = get_code_refinement_agent(refiner_llm_instance, tools=[file_reader_tool])
        file_manager_agent = get_file_manager_agent(file_manager_llm_instance, tools=[file_writer_tool, file_replacer_tool])
        # Pass the instantiated tool to the agent
        remapping_advisor_agent = get_remapping_advisor_agent(remapping_llm_instance, tools=[remapping_tool])
        logger.info("Instantiated LLMs, Tools, and Agents for Step 5 execution.")


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
        packages_to_process_this_run = []
        potential_target_package_ids = set() # Track all packages that *could* be processed

        all_packages = self.state_manager.get_all_packages()
        if not all_packages:
             logger.warning("No packages found in state. Cannot proceed with Step 5.")
             self.state_manager.update_workflow_status('step5_complete') # Or skipped status
             return True

        # Define all states that imply Step 4 (mapping) is done
        step4_completed_or_later_states = {
            target_status, # 'mapping_defined'
            'running_processing',
            processed_status, # 'processed'
            'needs_remapping',
        } | set(failed_status_prefixes) # Add 'failed_processing', 'failed_remapping'

        # Determine potential targets based on initial state and force flag
        for pkg_id, pkg_data in all_packages.items():
             current_status = pkg_data.get('status')
             is_target = (current_status == target_status)
             # Check if the status is one indicating Step 4 was done
             is_step4_done_or_later = current_status in step4_completed_or_later_states

             matches_specific_request = (not package_ids or pkg_id in package_ids)

             if matches_specific_request:
                  # Eligible if: It's the target status OR (force=True AND Step 4 was completed or a later state was reached)
                  if is_target or (force and is_step4_done_or_later):
                       potential_target_package_ids.add(pkg_id)

        # Build the list for *this run* based on potential targets and force flag
        for pkg_id in potential_target_package_ids:
             pkg_data = all_packages[pkg_id]
             current_status = pkg_data.get('status')
             is_target = (current_status == target_status)
             is_running_this_step = (current_status == 'running_processing') # Check if interrupted during this step
             is_step4_done_or_later = current_status in step4_completed_or_later_states

             if is_target or is_running_this_step:
                  # If it's the normal target status OR was interrupted during this step's processing, add it
                  if is_running_this_step:
                       logger.info(f"Resuming processing for package '{pkg_id}' found in state 'running_processing'.")
                  packages_to_process_this_run.append(pkg_id)
             elif force and is_step4_done_or_later:
                  # If force=True and it's in *any* state after mapping_defined (and not target/running), add it and reset status
                  logger.info(f"Force=True: Adding package '{pkg_id}' (status: {current_status}) to process list for Step 5.")
                  # Reset status to target status before processing
                  self.state_manager.update_package_state(pkg_id, target_status, error=None) # Clear previous error/status
                  packages_to_process_this_run.append(pkg_id)
             # Else: It was a potential target but not in the right state for this run (e.g., completed/failed without force)

        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 5 run.")
            # Check if the overall step should be marked complete based on *potential* targets
            all_potential_targets_done_or_failed = True
            final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
            for pkg_id in potential_target_package_ids:
                 status = final_packages_state.get(pkg_id, {}).get('status')
                 # Terminal states for Step 5 are 'processed' or failed states for this step or remapping triggered
                 if not (status == processed_status or status == 'needs_remapping' or any(status.startswith(prefix) for prefix in failed_status_prefixes)):
                      all_potential_targets_done_or_failed = False
                      break
            if all_potential_targets_done_or_failed and potential_target_package_ids:
                 logger.info("All potential target packages for Step 5 are now processed or failed.")
                 current_global_status = self.state_manager.get_state().get('workflow_status')
                 if not (current_global_status and 'failed' in current_global_status):
                      self.state_manager.update_workflow_status('step5_complete') # Or 'completed'
            return True # Indicate this specific invocation had nothing to fail on

        logger.info(f"Eligible packages identified for this Step 5 run (Force={force}): {packages_to_process_this_run}")

        # --- Fetch Processing Order ---
        processing_order = self.state_manager.get_package_processing_order()
        if processing_order is None:
            logger.error("Critical: Package processing order is missing from state. Cannot ensure correct execution order for Step 5.")
            self.state_manager.update_workflow_status('failed_step5', "Processing order missing.")
            return False
        if not isinstance(processing_order, list):
             logger.error(f"Critical: Package processing order is not a list (type: {type(processing_order)}). Invalid state.")
             self.state_manager.update_workflow_status('failed_step5', "Invalid processing order format.")
             return False

        # --- Process packages iteratively IN ORDER ---
        self.state_manager.update_workflow_status('running_step5')
        overall_success_this_run = True # Tracks success *of this specific execution run*
        processed_a_package = False # Track if any package was actually processed in this run

        for pkg_id in processing_order:
            # Process only if the package is in the eligible list for this run
            if pkg_id not in packages_to_process_this_run:
                continue

            processed_a_package = True # Mark that we are processing at least one
            logger.info(f"Processing Step 5 for package: {pkg_id} (in defined order)")
            self.state_manager.update_package_state(pkg_id, status='running_processing')
            package_success = True # Track success per package

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                # --- Load Mapping Artifact ---
                mapping_artifact_name = pkg_info.get('artifacts', {}).get('mapping_json')
                if not mapping_artifact_name:
                     # Attempt to load default if forcing a re-run from a failed state?
                     if force and pkg_info.get('status', '').startswith('failed_'):
                          mapping_artifact_name = f"package_{pkg_id}_mapping.json"
                          logger.warning(f"Mapping artifact not explicitly found for forced run of {pkg_id}, trying default: {mapping_artifact_name}")
                     else:
                          raise FileNotFoundError(f"Mapping JSON artifact name missing in state for package {pkg_id}.")

                mapping_data = self.state_manager.load_artifact(mapping_artifact_name, expect_json=True)
                if not mapping_data or not isinstance(mapping_data, dict):
                     raise FileNotFoundError(f"Failed to load or parse mapping JSON artifact: {mapping_artifact_name}")

                task_groups = mapping_data.get("task_groups", [])
                if not isinstance(task_groups, list):
                     raise TypeError(f"Task groups in {mapping_artifact_name} is not a list.")

                # Flatten task items from groups
                task_items = []
                for group in task_groups:
                    if isinstance(group, dict) and isinstance(group.get("tasks"), list):
                        task_items.extend(group.get("tasks", []))
                    else:
                        logger.warning(f"Invalid task group format in {mapping_artifact_name}: {group}")

                if not task_items:
                     logger.warning(f"No task items found in mapping artifact for package {pkg_id}. Skipping processing.")
                     self.state_manager.update_package_state(pkg_id, status='processed', error="No tasks found in mapping file.")
                     continue # Skip to next package

                # --- Assemble Base Context for the Package ---
                context_parts = []
                # 1. Instructions
                instr_context = self.context_manager.get_instruction_context()

                # 2. Package Description
                pkg_desc = pkg_info.get('description', 'N/A')
                context_parts.append(f"**Current Package ({pkg_id}) Description:**\n{pkg_desc}")
                # 3. Source Files List
                source_files = self.context_manager.get_source_file_list(pkg_id)
                if source_files:
                    source_list_str = "\n".join([f"- `{f['file_path']}`: {f['role']}" for f in source_files])
                    context_parts.append(f"**Source Files & Roles for {pkg_id}:**\n{source_list_str}")
                # 4. Target Files List
                target_files = self.context_manager.get_target_file_list(pkg_id)
                if target_files:
                     target_list_str = "\n".join([f"- `{f['path']}` (Exists: {f['exists']}): {f['purpose']}" for f in target_files])
                     context_parts.append(f"**Target Files & Status for {pkg_id}:**\n{target_list_str}")
                # 5. Mapping Strategy (Optional)
                mapping_strategy = mapping_data.get("mapping_strategy")
                if mapping_strategy:
                     context_parts.append(f"**Overall Mapping Strategy for {pkg_id}:**\n{mapping_strategy}")
                # 6. Source Code Content (Add this to the base context for reference by tasks)
                # Calculate token budget carefully
                temp_context_so_far = "\n\n".join(context_parts)
                tokens_so_far = count_tokens(temp_context_so_far) + count_tokens(instr_context)
                # Allocate remaining budget, maybe slightly less than 100% to be safe
                max_source_tokens = int((global_config.MAX_CONTEXT_TOKENS - global_config.PROMPT_TOKEN_BUFFER) * 0.8) - tokens_so_far
                if max_source_tokens > 0:
                    source_code = self.context_manager.get_work_package_source_code_content(pkg_id, max_tokens=max_source_tokens)
                    if source_code:
                        context_parts.append(f"**Source Code for {pkg_id}:**\n{source_code}")
                    else:
                        logger.warning(f"Could not retrieve source code for {pkg_id} within token limits for Step 5 base context.")
                else:
                    logger.warning(f"Not enough token budget remaining for source code of {pkg_id} in Step 5 base context.")

                # Combine base context parts
                package_context_str = "\n\n---\n\n".join(context_parts)
                base_context_tokens = count_tokens(package_context_str) + count_tokens(instr_context)
                logger.info(f"Assembled base context for Step 5 - {pkg_id} ({base_context_tokens} tokens).")


                # --- Create CrewAI Tasks for the Package ---
                # Pass agents instantiated above
                crew_tasks = self._create_crewai_tasks_for_package(
                    task_items=task_items,
                    package_context_str=package_context_str,
                    instructions=instr_context,
                    remapping_advisor_agent=remapping_advisor_agent
                )
                if not crew_tasks:
                     logger.error(f"Failed to create CrewAI tasks for package {pkg_id}. Skipping package.")
                     self.state_manager.update_package_state(pkg_id, status='failed_processing', error="Failed to create internal tasks.")
                     overall_success_this_run = False
                     continue

                # --- Create and Run Hierarchical Crew ---
                package_crew = Crew(
                    agents=[ # List ALL worker agents
                        code_generator_agent,
                        project_validator_agent,
                        code_refinement_agent,
                        file_manager_agent,
                        remapping_advisor_agent
                    ],
                    tasks=crew_tasks, # Tasks for the manager to orchestrate
                    process=Process.hierarchical,
                    # Assign only the essential manager_llm for hierarchical process
                    manager_llm=manager_llm_instance,
                    memory=True, # Enable memory for context persistence within the package run
                    verbose=True,
                    task_callback=self._log_step5_task_completion # Add the callback
                )

                logger.info(f"Kicking off Hierarchical Crew for Package {pkg_id} with {len(task_items)} items...")
                crew_result = package_crew.kickoff()
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


                # Determine overall package success
                if parsed_advice is None:
                     package_success = False
                     logger.error(f"Package {pkg_id} processing failed: Could not determine remapping advice from crew output.")
                     # TODO: Potentially inspect individual task results if accessible
                else:
                     package_success = not parsed_advice.recommend_remapping

                # Store detailed results (raw output or parsed task results) as an artifact
                package_report_filename = f"package_{pkg_id}_crew_results.json"
                package_report_path = os.path.join(self.analysis_dir, package_report_filename)
                try:
                    os.makedirs(self.analysis_dir, exist_ok=True)
                    report_content = crew_result if isinstance(crew_result, (dict, list)) else {'raw_output': str(crew_result)}
                    with open(package_report_path, 'w', encoding='utf-8') as f:
                        json.dump(report_content, f, indent=2)
                    logger.info(f"Saved crew results report: {package_report_path}")
                    # Update state with the report artifact path
                    current_artifacts = pkg_info.get('artifacts', {})
                    current_artifacts['crew_results_report'] = package_report_filename
                    self.state_manager.update_package_state(pkg_id, artifacts=current_artifacts) # Only update artifacts here
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

                    if current_remapping_attempts >= max_remap:
                        logger.warning(f"Max remapping attempts ({max_remap}) reached for package {pkg_id}. Marking as failed_remapping.")
                        self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remapping attempts reached. Last error: {error_reason}")
                    elif parsed_advice and parsed_advice.recommend_remapping:
                        logger.info(f"Remapping recommended for package {pkg_id} by RemappingAdvisorAgent (Attempt {current_remapping_attempts + 1}). Reason: {parsed_advice.reason}")
                        self.state_manager.update_package_state(pkg_id, status='needs_remapping', error=f"Remapping recommended: {parsed_advice.reason}", increment_remap_attempt=True)
                    else:
                        logger.info(f"Failures detected for package {pkg_id}, but remapping not recommended or advisor failed.")
                        final_error = error_reason if error_reason else "Package processing failed, but remapping condition not met or advisor failed."
                        # TODO: Extract specific errors from report_content if possible
                        self.state_manager.update_package_state(pkg_id, status='failed_processing', error=final_error)
                else:
                    logger.info(f"Hierarchical crew processed package {pkg_id} successfully.")
                    self.state_manager.update_package_state(pkg_id, status='processed')

            except Exception as e: # Catch errors during package setup or crew execution kickoff
                logger.error(f"A critical error occurred during Step 5 hierarchical processing for package {pkg_id}: {e}", exc_info=True)
                self.state_manager.update_package_state(pkg_id, status='failed_processing', error=f"Critical executor error during crew setup/kickoff: {e}")
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
             needs_remapping_pending = any(p.get('status') == 'needs_remapping' for p_id, p in final_packages_state.items() if p_id in potential_target_package_ids) # Check only potential targets
             if not needs_remapping_pending:
                  logger.info("All potential target packages for Step 5 are now processed or failed.")
                  current_global_status = self.state_manager.get_state().get('workflow_status')
                  if not (current_global_status and 'failed' in current_global_status):
                       self.state_manager.update_workflow_status('step5_complete') # Or 'completed'
             else:
                  logger.info("Step 5 finished processing available packages, but some require remapping.")
        elif not overall_success_this_run:
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  self.state_manager.update_workflow_status('failed_step5', "One or more packages failed during code processing in the latest run.")


        logger.info(f"--- Finished Step 5 Execution Run (Overall Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run

    # --- Helper Method to Create Tasks ---
    def _create_crewai_tasks_for_package(self,
                                         task_items: List[Dict[str, Any]],
                                         package_context_str: str,
                                         instructions: Optional[str],
                                         remapping_advisor_agent: Agent) -> List[Task]:
        """Creates the list of CrewAI tasks for a work package."""
        crew_tasks: List[Task] = []
        item_processing_tasks: List[Task] = [] # Keep track of item tasks for final analysis context
        godot_project_path = self.config.get("GODOT_PROJECT_DIR") # Get base Godot dir (absolute path)
        if not godot_project_path:
             raise Exception("GODOT_PROJECT_DIR not configured. Cannot proceed with task creation.")

        for task_item in task_items:
            item_context_parts = [
                package_context_str,
                f"\n\n**Godot Project Path:** `{godot_project_path}`" # Add project path to context
            ]

            # Attempt to read target file content for this specific item
            target_godot_file_res_path = task_item.get('output_godot_file')
            target_file_content = None
            if target_godot_file_res_path and target_godot_file_res_path.startswith("res://"):
                relative_path = target_godot_file_res_path[len("res://"):]
                absolute_path = os.path.join(godot_project_path, relative_path)
                # Use the imported read_godot_file_content function
                target_file_content = read_godot_file_content(absolute_path)
                if target_file_content is not None:
                    logger.debug(f"Read content for target file: {target_godot_file_res_path}")
                    item_context_parts.append(
                        f"\n\n**Target File Content (`{target_godot_file_res_path}`):**\n"
                        f"```gdscript\n{target_file_content}\n```"
                    )
                else:
                    logger.warning(f"Could not read content for target file: {target_godot_file_res_path} (Path: {absolute_path})")
            elif target_godot_file_res_path:
                 logger.warning(f"Target file path '{target_godot_file_res_path}' does not start with 'res://' or GODOT_PROJECT_DIR is missing. Cannot read content.")

            # Combine context for this item
            item_specific_context_str = "\n".join(item_context_parts)

            # Create task for the manager to process this item using item-specific context
            item_task = create_hierarchical_process_taskitem_task(
                manager_agent=None, # Manager is implicit
                task_item_details=task_item,
                package_context=item_specific_context_str,
                instructions=instructions,
                dependent_tasks=None # Let manager handle sequence/context flow
            )
            crew_tasks.append(item_task)
            item_processing_tasks.append(item_task) # Add to list for final task context

        # Add the final analysis task that depends on all item tasks
        if item_processing_tasks: # Only add if there were items to process
             analysis_task = create_analyze_package_failures_task(
                 advisor_agent=remapping_advisor_agent,
                 all_item_results_context=item_processing_tasks,
                 instructions=instructions
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
            # Robustly determine agent role
            agent_role = "Unknown Agent"
            if hasattr(task_output, 'agent') and task_output.agent:
                # Check if it's actually an Agent object before accessing role
                if isinstance(task_output.agent, Agent):
                    agent_role = task_output.agent.role
                else:
                    # Log if it's not an Agent object (e.g., could be a string ID)
                    logger.warning(f"[Step 5 Crew Callback] task_output.agent is not an Agent object (Type: {type(task_output.agent)}). Using placeholder role.")
                    agent_role = f"Agent ({type(task_output.agent).__name__})"

            # Robustly determine task description
            task_desc_snippet = "Unknown Task"
            if hasattr(task_output, 'task') and task_output.task and hasattr(task_output.task, 'description'):
                 task_desc_snippet = task_output.task.description[:100] + "..."
            elif hasattr(task_output, 'description') and isinstance(task_output.description, str): # Fallback if task object isn't populated but description is
                 task_desc_snippet = task_output.description[:100] + "..."

            # Robustly determine output snippet
            output_snippet = "No output"
            if hasattr(task_output, 'raw_output') and task_output.raw_output is not None:
                 output_snippet = str(task_output.raw_output)[:150].replace('\n', ' ') + "..."
            elif hasattr(task_output, 'output') and task_output.output is not None: # Fallback
                 output_snippet = str(task_output.output)[:150].replace('\n', ' ') + "..."


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
