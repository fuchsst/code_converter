# src/core/executors/step4_mapping_definer.py
# Standard library imports
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type # Added Type

# CrewAI imports
from crewai import Crew, Process, Task, Agent

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager

# Import NEW agents for Step 4
from src.agents.step4.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step4.godot_structure_analyst import get_godot_structure_analyst_agent
from src.agents.step4.conversion_strategist import get_conversation_strategist_agent
from src.agents.step4.task_decomposer import get_task_decomposer_agent
from src.agents.step4.json_output_formatter import get_json_output_fomratter_agent

# Import NEW Task definition for Step 4
from src.tasks.step4.define_mapping import HierarchicalDefineMappingTask, MappingOutput # Import Pydantic model too

# Import utilities
from src.utils.parser_utils import parse_json_from_string # Keep for parsing final output if needed
from src.utils.formatting_utils import format_structure_to_markdown
from src.logger_setup import get_logger
import src.config as config
# Import TaskOutput for type hinting the callback parameter
from crewai.tasks.task_output import TaskOutput

logger = get_logger(__name__)

class Step4Executor(StepExecutor):
    """Executes Step 4: C++ to Godot Mapping Definition using a hierarchical multi-agent crew."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config_dict: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]],
                 tools: Dict[Type, Any]): # Tools might not be needed here unless agents use them
        super().__init__(state_manager, context_manager, config_dict, llm_configs, tools)

        # --- Instantiate Task Creator ---
        self.mapping_task_creator = HierarchicalDefineMappingTask()
        logger.info("Instantiated Task creator for Step 4.")

        # --- Get LLM Instances ---
        # Define grouped roles used by Step 4 agents + manager
        analyzer_config = self._get_llm_config('analyzer')
        designer_planner_config = self._get_llm_config('designer_planner')
        utility_config = self._get_llm_config('utility')
        manager_config = self._get_llm_config('manager')

        # Get LLM instances using grouped roles
        self.cpp_analyst_llm = self._get_llm_instance_by_role('analyzer', analyzer_config)
        self.godot_analyst_llm = self._get_llm_instance_by_role('analyzer', analyzer_config)
        self.strategist_llm = self._get_llm_instance_by_role('designer_planner', designer_planner_config)
        self.decomposer_llm = self._get_llm_instance_by_role('designer_planner', designer_planner_config)
        self.formatter_llm = self._get_llm_instance_by_role('utility', utility_config)
        self.manager_llm = self._get_llm_instance_by_role('manager', manager_config)

        if not self.manager_llm: # Manager is crucial
             logger.error("Missing critical LLM configuration for Step 4 Manager role. Cannot proceed.")
             raise ValueError("Missing critical LLM configuration for Step 4 Manager role.")

        # --- Instantiate Agents ---
        # Use the specific LLM instances fetched above
        self.cpp_analyst_agent = get_cpp_code_analyst_agent(self.cpp_analyst_llm)
        self.godot_analyst_agent = get_godot_structure_analyst_agent(self.godot_analyst_llm)
        self.strategist_agent = get_conversation_strategist_agent(self.strategist_llm)
        self.decomposer_agent = get_task_decomposer_agent(self.decomposer_llm)
        self.formatter_agent = get_json_output_fomratter_agent(self.formatter_llm)

        # Log if any LLM instance is missing (should be caught by verification below)
        if not self.cpp_analyst_llm: logger.warning("Step4 Cpp Analyst LLM instance is missing.")
        if not self.godot_analyst_llm: logger.warning("Step4 Godot Analyst LLM instance is missing.")
        if not self.strategist_llm: logger.warning("Step4 Strategist LLM instance is missing.")
        if not self.decomposer_llm: logger.warning("Step4 Decomposer LLM instance is missing.")
        if not self.formatter_llm: logger.warning("Step4 Formatter LLM instance is missing.")

        logger.info("Instantiated CrewAI agents for Step 4.")

        # Verify all agents have an LLM assigned
        agent_list = [self.cpp_analyst_agent, self.godot_analyst_agent, self.strategist_agent, self.decomposer_agent, self.formatter_agent]
        for agent in agent_list:
             if not agent.llm:
                  logger.error(f"Agent '{agent.role}' in Step4Executor ended up with no assigned LLM. Check LLM configuration and instantiation.")
                  raise ValueError(f"Agent '{agent.role}' missing LLM instance.")
             else:
                  logger.debug(f"Agent '{agent.role}' successfully assigned LLM: {type(agent.llm)}")


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
                 from src.llms.google_genai_llm import GoogleGenAI_LLM # Local import
                 # Ensure necessary params are present, add defaults if needed
                 llm_config_copy = llm_config.copy() # Avoid modifying the original dict
                 llm_config_copy.setdefault('timeout', config.GEMINI_TIMEOUT)
                 llm_instance = GoogleGenAI_LLM(**llm_config_copy)
                 logger.info(f"Successfully instantiated GoogleGenAI_LLM for role '{role}': {model_identifier}")
             else:
                 # Use default CrewAI LLM for other models (e.g., OpenAI configured via env vars)
                 # CrewAI's default LLM loader might handle API keys from environment variables.
                 from crewai import LLM as CrewAI_LLM # Local import
                 # Pass only the necessary args expected by CrewAI_LLM if needed,
                 # or let it use its defaults / environment variables.
                 # For simplicity, assume CrewAI_LLM handles config dict well or uses env vars.
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
        Runs the C++ to Godot mapping definition for specified or all eligible packages.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process.
                                                If None, processes all eligible packages.
            force (bool): If True, forces reprocessing of packages even if already mapped or failed.
            **kwargs: Accepts 'feedback_override' dict mapping package_id to feedback string
                      for remapping scenarios.

        Returns:
            bool: True if mapping definition was successful for the processed packages in this run, False otherwise.
        """
        feedback_override = kwargs.get('feedback_override', {})
        logger.info(f"--- Starting Step 4 Execution (Hierarchical): Define Mapping (Packages: {package_ids or 'All Eligible'}, Force={force}) ---")

        # --- Identify Eligible Packages (Logic remains the same) ---
        packages_to_process_this_run = []
        potential_target_package_ids = set()
        target_status = 'structure_defined' # Step 4 follows Step 3
        failed_status_prefix_mapping = 'failed_mapping'
        failed_status_prefix_remapping = 'failed_remapping'
        needs_remapping_status = 'needs_remapping' # Status set by Step 5
        completed_status = 'mapping_defined' # Success status for this step

        all_packages = self.state_manager.get_all_packages()
        if not all_packages:
             logger.warning("No packages found in state. Cannot proceed with Step 4.")
             self.state_manager.update_workflow_status('step4_complete') # Or skipped status
             return True

        # First pass: identify all packages that could potentially be processed
        for pkg_id, pkg_data in all_packages.items():
             current_status = pkg_data.get('status')
             is_target = (current_status == target_status)
             is_failed_mapping = (current_status and current_status.startswith(failed_status_prefix_mapping))
             is_failed_remapping = (current_status and current_status.startswith(failed_status_prefix_remapping))
             is_needs_remapping = (current_status == needs_remapping_status)
             is_already_completed = (current_status == completed_status)

             matches_specific_request = (not package_ids or pkg_id in package_ids)

             if matches_specific_request:
                 # Eligible if: Target status OR Needs remapping OR (Force AND (Failed OR Already Completed))
                 if is_target or is_needs_remapping or (force and (is_failed_mapping or is_failed_remapping or is_already_completed)):
                     potential_target_package_ids.add(pkg_id)

        # Second pass: build the list for *this run*
        for pkg_id in potential_target_package_ids:
             pkg_data = all_packages[pkg_id]
             current_status = pkg_data.get('status')
             is_target = (current_status == target_status)
             is_failed_mapping = (current_status and current_status.startswith(failed_status_prefix_mapping))
             is_failed_remapping = (current_status and current_status.startswith(failed_status_prefix_remapping))
             is_needs_remapping = (current_status == needs_remapping_status)
             is_already_completed = (current_status == completed_status)

             if is_target or is_needs_remapping:
                  packages_to_process_this_run.append(pkg_id)
             elif force and (is_failed_mapping or is_failed_remapping or is_already_completed):
                  logger.info(f"Force=True: Adding package '{pkg_id}' (status: {current_status}) to process list for Step 4.")
                  # Reset status to target status before processing
                  self.state_manager.update_package_state(pkg_id, target_status, error=None) # Clear previous error/status
                  packages_to_process_this_run.append(pkg_id)

        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 4 run.")
            # Check if the overall step should be marked complete based on *potential* targets
            all_potential_targets_done_or_failed = True
            final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
            for pkg_id in potential_target_package_ids:
                 status = final_packages_state.get(pkg_id, {}).get('status')
                 if not (status == completed_status or (status and (status.startswith(failed_status_prefix_mapping) or status.startswith(failed_status_prefix_remapping)))):
                      all_potential_targets_done_or_failed = False
                      break
            if all_potential_targets_done_or_failed and potential_target_package_ids:
                 logger.info("All potential target packages for Step 4 are now processed or failed.")
                 current_global_status = self.state_manager.get_state().get('workflow_status')
                 if not (current_global_status and 'failed' in current_global_status):
                      self.state_manager.update_workflow_status('step4_complete')
            return True # Indicate this specific invocation had nothing to fail on

        logger.info(f"Packages to process in this Step 4 run (Force={force}): {packages_to_process_this_run}")
        self.state_manager.update_workflow_status('running_step4')
        overall_success_this_run = True # Tracks success *of this specific run*
        analysis_dir = Path(self.config.get("ANALYSIS_OUTPUT_DIR", "analysis_output")).resolve() # Keep for structure path
        godot_project_dir_str = self.config.get("GODOT_PROJECT_DIR")
        if not godot_project_dir_str:
             logger.error("GODOT_PROJECT_DIR not found in config. Cannot proceed with Step 4.")
             self.state_manager.update_workflow_status('failed_step4', "GODOT_PROJECT_DIR missing in config.")
             return False
        godot_project_dir = Path(godot_project_dir_str).resolve()

        # Ensure Manager LLM is available (checked in __init__, but double-check)
        if not self.manager_llm:
             logger.error("Manager LLM instance is missing. Cannot execute Step 4.")
             self.state_manager.update_workflow_status('failed_step4', "Manager LLM instance missing.")
             return False

        # --- Load Global Packages Summary (packages.json content) ---
        # Use StateManager to load this artifact
        packages_json_content = self.state_manager.load_artifact("packages.json", expect_json=True)
        if packages_json_content is None:
            logger.error("Failed to load global packages summary (packages.json) via StateManager. Cannot proceed.")
            self.state_manager.update_workflow_status('failed_step4', "Failed to load packages.json")
            return False
        elif not isinstance(packages_json_content, dict):
             logger.error(f"Loaded packages.json is not a dictionary (type: {type(packages_json_content)}). Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step4', "Invalid packages.json format")
             return False
        else:
             logger.info("Loaded global packages summary (packages.json) via StateManager.")


        # --- Get List of Existing Godot Output Files ---
        existing_godot_outputs = self.context_manager.get_existing_godot_output_files(str(godot_project_dir))


        # --- Initialize Overall Mapping Summary ---
        overall_mapping_summary: Dict[str, Any] = {
            "package_summaries": {},
            "all_output_files": set() # Use set for efficient unique collection
        }
        # Load existing overall mappings using StateManager if it exists
        overall_mapping_filename = "mappings.json"
        if not force: # Don't load if forcing a full rebuild
            existing_overall_summary = self.state_manager.load_artifact(overall_mapping_filename, expect_json=True)
            if existing_overall_summary is not None:
                if isinstance(existing_overall_summary, dict) and \
                   "package_summaries" in existing_overall_summary and \
                   "all_output_files" in existing_overall_summary:
                    overall_mapping_summary = existing_overall_summary
                    # Convert list back to set for processing
                    overall_mapping_summary["all_output_files"] = set(overall_mapping_summary.get("all_output_files", []))
                    logger.info(f"Loaded existing overall mapping summary ({overall_mapping_filename}) via StateManager.")
                else:
                    logger.warning(f"Existing overall mapping file {overall_mapping_filename} has invalid format. Starting fresh.")
            else:
                 logger.info(f"No existing overall mapping summary ({overall_mapping_filename}) found or failed to load. Starting fresh.")


        # --- Process Packages ---
        for pkg_id in packages_to_process_this_run:
            logger.info(f"Processing Step 4 for package: {pkg_id}")
            pkg_info_before_run = self.state_manager.get_package_info(pkg_id) # Get state before update
            is_remapping_run = (pkg_info_before_run.get('status') == needs_remapping_status or
                                (force and (pkg_info_before_run.get('status', '').startswith(failed_status_prefix_mapping) or
                                            pkg_info_before_run.get('status', '').startswith(failed_status_prefix_remapping) or
                                            pkg_info_before_run.get('status') == completed_status))) # Also remapping if forcing completed

            current_status_log = 'running_remapping' if is_remapping_run else 'running_mapping'
            self.state_manager.update_package_state(pkg_id, status=current_status_log)

            # --- Define Artifact Names (Remains similar) ---
            suffix = "_remapped" if is_remapping_run and pkg_info_before_run.get('status') != target_status else ""
            json_artifact_filename = f"package_{pkg_id}_mapping{suffix}.json"

            # --- Load Existing Data for Context (Remains similar) ---
            existing_mapping_json = None
            referenced_godot_content = {}
            original_json_artifact_filename = f"package_{pkg_id}_mapping.json"
            loaded_existing_mapping = self.state_manager.load_artifact(original_json_artifact_filename, expect_json=True)
            if loaded_existing_mapping and isinstance(loaded_existing_mapping, dict):
                logger.info(f"Found and loaded existing mapping file for refinement context: {original_json_artifact_filename}")
                existing_mapping_json = loaded_existing_mapping
                # Read referenced Godot files (Logic remains similar, ensure read_godot_file_content is robust)
                if "task_groups" in existing_mapping_json:
                    referenced_files_set: Set[str] = set()
                    for group in existing_mapping_json.get("task_groups", []):
                         if isinstance(group, dict):
                              for task in group.get("tasks", []):
                                   if isinstance(task, dict) and "output_godot_file" in task:
                                        referenced_files_set.add(task["output_godot_file"])
                    logger.info(f"Found {len(referenced_files_set)} unique Godot files referenced in existing mapping.")
                    for res_path in referenced_files_set:
                         if res_path.startswith("res://"):
                              relative_to_res = res_path[len("res://"):]
                              # Ensure read_godot_file_content exists and handles errors
                              if hasattr(self.context_manager, 'read_godot_file_content'):
                                   content = self.context_manager.read_godot_file_content(str(godot_project_dir), relative_to_res)
                                   if content is not None:
                                        referenced_godot_content[res_path] = content
                                   else: logger.warning(f"Failed to read content for referenced Godot file: {res_path}")
                              else: logger.error("ContextManager missing read_godot_file_content method.")
                         else: logger.warning(f"Referenced file path '{res_path}' does not start with 'res://'. Skipping.")
            elif loaded_existing_mapping:
                 logger.error(f"Loaded existing mapping file {original_json_artifact_filename} but it's not a dictionary. Ignoring.")

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info: raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                # --- Load Structure Definition ---
                structure_artifact_filename = pkg_info.get('artifacts', {}).get('structure_json')
                if not structure_artifact_filename: raise FileNotFoundError(f"Structure definition JSON artifact missing for package {pkg_id}.")
                structure_json_content = self.state_manager.load_artifact(structure_artifact_filename, expect_json=True)
                if not structure_json_content or not isinstance(structure_json_content, dict):
                     raise FileNotFoundError(f"Failed to load or parse structure JSON artifact: {structure_artifact_filename}")
                structure_notes = structure_json_content.get("notes", "") # Extract notes

                # --- Assemble Comprehensive Context ---
                primary_files = pkg_info.get('files', [])
                dependency_files = self.context_manager._get_dependencies_for_package(primary_files) if hasattr(self.context_manager, '_get_dependencies_for_package') else []

                context_kwargs = {
                    "primary_relative_paths": primary_files,
                    "dependency_relative_paths": dependency_files, # Request full content via ContextManager
                    "work_package_id": pkg_id,
                    "work_package_description": pkg_info.get('description', ''),
                    "proposed_godot_structure_json": structure_json_content, # Pass raw structure dict
                    "proposed_godot_structure_notes": structure_notes, # Pass notes separately
                    "global_packages_summary": packages_json_content,
                    "existing_godot_outputs": existing_godot_outputs,
                    "existing_mapping_json": existing_mapping_json,
                    "referenced_godot_content": referenced_godot_content # Pass dict of {path: content}
                }
                if pkg_id in feedback_override:
                     context_kwargs["previous_mapping_feedback"] = feedback_override[pkg_id]
                     logger.info(f"Adding remapping feedback for package {pkg_id}.")

                step_name_log = "MAPPING_DEFINITION_REFINEMENT" if existing_mapping_json else "MAPPING_DEFINITION"
                # ContextManager needs to handle fetching full content based on paths
                context = self.context_manager.get_context_for_step(step_name_log, **context_kwargs)
                if not context: raise ValueError("Failed to assemble context for Step 4.")

                # --- Create Hierarchical Task ---
                # Manager agent is implicit in hierarchical process
                task = self.mapping_task_creator.create_task(
                    manager_agent=None, # Manager is implicit
                    context=context,
                    package_id=pkg_id
                )

                # --- Create and run Hierarchical Crew ---
                crew = Crew(
                    agents=[ # List ALL worker agents for Step 4
                        self.cpp_analyst_agent,
                        self.godot_analyst_agent,
                        self.strategist_agent,
                        self.decomposer_agent,
                        self.formatter_agent
                    ],
                    tasks=[task], # Single task for the manager
                    process=Process.hierarchical,
                    llm=self.manager_llm,
                    manager_llm=self.manager_llm, 
                    planning_llm=self.manager_llm,
                    planning=True,
                    memory=True,
                    verbose=True,
                    task_callback=self._log_step4_task_completion # Add the callback
                )
                logger.info(f"Kicking off Hierarchical Crew for Step 4 (Package: {pkg_id}, Remapping: {is_remapping_run})...")
                result = crew.kickoff() # Expecting final JSON string from FormatterAgent
                logger.info(f"Step 4 Hierarchical Crew finished for package: {pkg_id}")
                logger.debug(f"Crew Result Raw Output:\n{result}")

                # --- Result Parsing & Validation ---
                mapping_data_dict = None
                if isinstance(result, str):
                    # Attempt to parse the final string output as JSON
                    parsed_dict_attempt = parse_json_from_string(result)
                    if parsed_dict_attempt:
                        try:
                            MappingOutput(**parsed_dict_attempt) # Validate structure
                            logger.info("Final crew output conforms to MappingOutput model.")
                            mapping_data_dict = parsed_dict_attempt
                        except Exception as pydantic_err:
                            logger.error(f"Final crew output does NOT conform to MappingOutput model: {pydantic_err}", exc_info=True)
                            logger.debug(f"Invalid JSON received: {result}")
                    else:
                        logger.error(f"Failed to parse final crew output string as JSON for {pkg_id}. Raw output:\n---\n{result}\n---")
                # Handle CrewOutput object if returned (less likely for final task?)
                elif isinstance(result, dict): # If kickoff somehow returns a dict
                     try:
                          MappingOutput(**result)
                          logger.info("Crew output dict conforms to MappingOutput model.")
                          mapping_data_dict = result
                     except Exception as pydantic_err:
                          logger.error(f"Crew output dict does NOT conform to MappingOutput model: {pydantic_err}", exc_info=True)
                else:
                    logger.error(f"Unexpected final crew output type for {pkg_id}: {type(result)}")

                if not mapping_data_dict:
                    raise ValueError(f"Step 4 Hierarchical Crew failed to produce valid mapping data for {pkg_id}.")

                # --- Save JSON artifact using StateManager ---
                save_json_ok = self.state_manager.save_artifact(json_artifact_filename, mapping_data_dict, is_json=True)
                if not save_json_ok:
                     raise IOError(f"Failed to save structured mapping artifact via StateManager: {json_artifact_filename}")

                # --- Update Overall Mapping Summary (Logic remains similar) ---
                overall_mapping_summary["package_summaries"][pkg_id] = {"mapping_file": json_artifact_filename}
                if "task_groups" in mapping_data_dict:
                    for group in mapping_data_dict.get("task_groups", []):
                        if isinstance(group, dict):
                            for task_item in group.get("tasks", []):
                                if isinstance(task_item, dict) and "output_godot_file" in task_item:
                                    overall_mapping_summary["all_output_files"].add(task_item["output_godot_file"])
                logger.debug(f"Updated overall mapping summary for package {pkg_id}.")

                # --- Update state (Logic remains similar) ---
                artifacts_to_update = {'mapping_json': json_artifact_filename}
                self.state_manager.update_package_state(
                    pkg_id,
                    status=completed_status, # 'mapping_defined'
                    artifacts=artifacts_to_update,
                    increment_remap_attempt=is_remapping_run
                )
                logger.info(f"Package {pkg_id} successfully (re)mapped. Status set to '{completed_status}'.")

                # --- Save Overall Mapping Summary (Logic remains similar) ---
                try:
                    overall_mapping_to_save = overall_mapping_summary.copy()
                    overall_mapping_to_save["all_output_files"] = sorted(list(overall_mapping_summary["all_output_files"]))
                    save_overall_ok = self.state_manager.save_artifact(overall_mapping_filename, overall_mapping_to_save, is_json=True)
                    if save_overall_ok: logger.info(f"Saved intermediate overall mapping summary ({overall_mapping_filename}) after processing {pkg_id}.")
                    else: logger.error(f"Failed to save intermediate overall mapping summary ({overall_mapping_filename}) after processing {pkg_id}.")
                except Exception as e_save:
                    logger.error(f"Unexpected error saving intermediate overall mapping summary after {pkg_id}: {e_save}", exc_info=True)

            except Exception as e:
                logger.error(f"An error occurred during Step 4 hierarchical processing for package {pkg_id}: {e}", exc_info=True)
                fail_status = failed_status_prefix_remapping if is_remapping_run else failed_status_prefix_mapping
                self.state_manager.update_package_state(pkg_id, status=fail_status, error=str(e))
                overall_success_this_run = False

        # --- Final Workflow Status Check ---
        all_potential_targets_done_or_failed = True
        final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
        for pkg_id in potential_target_package_ids:
             status:str = final_packages_state.get(pkg_id, {}).get('status')
             if not (status == completed_status or (status and (status.startswith(failed_status_prefix_mapping) or status.startswith(failed_status_prefix_remapping)))):
                  all_potential_targets_done_or_failed = False
                  logger.debug(f"Package {pkg_id} is still pending Step 4 completion (status: {status}).")
                  break

        if all_potential_targets_done_or_failed and potential_target_package_ids:
             logger.info("All potential target packages for Step 4 are now processed or failed.")
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  self.state_manager.update_workflow_status('step4_complete')
        elif not overall_success_this_run:
             current_global_status = self.state_manager.get_state().get('workflow_status')
             if not (current_global_status and 'failed' in current_global_status):
                  self.state_manager.update_workflow_status('failed_step4', "One or more packages failed during mapping definition in the latest run.")

        logger.info(f"--- Finished Step 4 Execution Run (Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run

    # --- Callback Method ---
    def _log_step4_task_completion(self, task_output: TaskOutput):
        """Logs information about each completed task within the Step 4 crew."""
        try:
            agent_role = task_output.agent.role if task_output.agent else "Unknown Agent"
            task_desc_snippet = task_output.task.description[:100] + "..." if task_output.task else "Unknown Task"
            output_snippet = task_output.raw_output[:150].replace('\n', ' ') + "..." if task_output.raw_output else "No output"

            logger.info(f"[Step 4 Crew Callback] Task Completed:")
            logger.info(f"  - Agent: {agent_role}")
            logger.info(f"  - Task Desc (Start): {task_desc_snippet}")
            logger.info(f"  - Output (Start): {output_snippet}")
        except Exception as e:
            logger.error(f"[Step 4 Crew Callback] Error processing task output: {e}", exc_info=True)
            try:
                 logger.debug(f"Raw task_output object: {task_output}")
            except Exception:
                 logger.debug("Could not log raw task_output object.")
