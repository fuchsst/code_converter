# src/core/executors/step4_mapping_definer.py
# Standard library imports
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

# CrewAI imports
from crewai import Crew, Process, Task, Agent
from crewai.crews.crew_output import CrewOutput # Import for handling crew result type

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens, read_godot_file_content

# Import NEW agents for Step 4
from src.agents.step4.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step4.godot_structure_analyst import get_godot_structure_analyst_agent
from src.agents.step4.conversion_strategist import get_conversation_strategist_agent
from src.agents.step4.task_decomposer import get_task_decomposer_agent
from src.agents.step4.json_output_formatter import get_json_output_fomratter_agent

# Import NEW Task definition for Step 4
from src.tasks.step4.define_mapping import create_hierarchical_define_mapping_task, MappingOutput

# Import utilities
from src.utils.json_utils import parse_json_from_string
from src.logger_setup import get_logger
import src.config as global_config # Use alias
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
        logger.info("Step4Executor initialized.")

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

        # --- Instantiate LLMs and Agents using the new helper ---
        cpp_analyst_llm = self._create_llm_instance('ANALYZER_MODEL')
        godot_analyst_llm = self._create_llm_instance('ANALYZER_MODEL')
        strategist_llm = self._create_llm_instance('DESIGNER_PLANNER_MODEL')
        decomposer_llm = self._create_llm_instance('DESIGNER_PLANNER_MODEL')
        formatter_llm = self._create_llm_instance('UTILITY_MODEL', response_schema_class=MappingOutput)
        manager_llm = self._create_llm_instance('MANAGER_MODEL')

        # Check if all required LLMs were instantiated successfully
        required_llms = {
            'Manager': manager_llm,
            'CPP Analyst': cpp_analyst_llm,
            'Godot Analyst': godot_analyst_llm,
            'Strategist': strategist_llm,
            'Decomposer': decomposer_llm,
            'Formatter': formatter_llm
        }
        missing_llms = [name for name, llm in required_llms.items() if not llm]

        if missing_llms:
             logger.error(f"Missing critical LLM configurations for Step 4: {', '.join(missing_llms)}. Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step4', f"Missing LLM config for: {', '.join(missing_llms)}")
             return False

        cpp_analyst_agent = get_cpp_code_analyst_agent(cpp_analyst_llm)
        godot_analyst_agent = get_godot_structure_analyst_agent(godot_analyst_llm)
        strategist_agent = get_conversation_strategist_agent(strategist_llm)
        decomposer_agent = get_task_decomposer_agent(decomposer_llm)
        formatter_agent = get_json_output_fomratter_agent(formatter_llm)
        logger.info("Instantiated LLMs and Agents for Step 4 execution.")


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

        # --- Load Global Packages Summary (packages.json content) --- # Now loaded via StateManager
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
                              content = read_godot_file_content(str(godot_project_dir / relative_to_res))
                              if content is not None:
                                   referenced_godot_content[res_path] = content
                              else: logger.warning(f"Failed to read content for referenced Godot file: {res_path}")
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

                # --- Assemble Context using ContextManager methods ---
                context_parts = []

                # 1. General Instructions
                instr_context = self.context_manager.get_instruction_context()

                # 2. Package Description
                pkg_desc = pkg_info.get('description', 'N/A')
                context_parts.append(f"**Current Package ({pkg_id}) Description:**\n{pkg_desc}")

                # 3. Source Files List & Roles
                source_files_list = self.context_manager.get_source_file_list(pkg_id)
                if source_files_list:
                    source_list_str = "\n".join([f"- `{f['file_path']}`: {f['role']}" for f in source_files_list])
                    context_parts.append(f"**Source Files & Roles for {pkg_id}:**\n{source_list_str}")

                # 4. Proposed Godot Structure (from Step 3 artifact)
                structure_str = json.dumps(structure_json_content, indent=2)
                context_parts.append(f"**Proposed Godot Structure for {pkg_id}:**\n```json\n{structure_str}\n```")
                if structure_notes:
                     context_parts.append(f"**Structure Notes:**\n{structure_notes}")

                # 5. Source Code Content (respecting limits)
                temp_context_so_far = "\n\n".join(context_parts)
                tokens_so_far = count_tokens(temp_context_so_far) + count_tokens(instr_context)
                max_source_tokens = int((global_config.MAX_CONTEXT_TOKENS - global_config.PROMPT_TOKEN_BUFFER) * 0.6) - tokens_so_far
                if max_source_tokens > 0:
                    source_code = self.context_manager.get_work_package_source_code_content(pkg_id, max_tokens=max_source_tokens)
                    if source_code:
                        context_parts.append(f"**Source Code for {pkg_id}:**\n{source_code}")
                    else:
                        logger.warning(f"Could not retrieve source code for {pkg_id} within token limits for Step 4.")
                else:
                    logger.warning(f"Not enough token budget remaining for source code of {pkg_id} in Step 4.")

                # 6. Existing Mapping (if refining)
                if existing_mapping_json:
                     context_parts.append(f"**Existing Mapping Definition for {pkg_id} (for refinement):**\n```json\n{json.dumps(existing_mapping_json, indent=2)}\n```")
                     if referenced_godot_content:
                          ref_content_str = "\n\n".join([f"// File: {p}\n```gdscript\n{c}\n```" for p, c in referenced_godot_content.items()])
                          context_parts.append(f"**Content of Referenced Godot Files:**\n{ref_content_str}")

                # 7. Remapping Feedback (if provided)
                if pkg_id in feedback_override:
                     context_parts.append(f"**Feedback on Previous Mapping Attempt:**\n{feedback_override[pkg_id]}")

                # 8. Globally Defined Godot Files (from overall mappings artifact)
                overall_mappings = self.state_manager.load_artifact("mappings.json", expect_json=True)
                all_defined_godot_files = []
                if overall_mappings and isinstance(overall_mappings.get("all_output_files"), list):
                     all_defined_godot_files = overall_mappings["all_output_files"]
                     if existing_mapping_json: # Use existing mapping to filter
                          current_pkg_files = set()
                          for group in existing_mapping_json.get("task_groups", []):
                               if isinstance(group, dict):
                                    for task in group.get("tasks", []):
                                         if isinstance(task, dict) and "output_godot_file" in task:
                                              current_pkg_files.add(task["output_godot_file"])
                          all_defined_godot_files = [f for f in all_defined_godot_files if f not in current_pkg_files]

                if all_defined_godot_files:
                     files_list_str = "\n".join([f"- `{f}`" for f in sorted(all_defined_godot_files)])
                     context_parts.append(f"**Globally Defined Godot Files (excluding current package if refining):**\n{files_list_str}")


                # Combine all context parts intended for the main prompt
                context_str = "\n\n---\n\n".join(context_parts)
                final_tokens = count_tokens(context_str) + count_tokens(instr_context)
                logger.info(f"Assembled context for Step 4 - {pkg_id} ({final_tokens} tokens).")
                if final_tokens >= (global_config.MAX_CONTEXT_TOKENS - global_config.PROMPT_TOKEN_BUFFER):
                     logger.warning(f"Context for {pkg_id} in Step 4 might be near or exceeding token limits!")

                if not context_str:
                    raise ValueError(f"Failed to assemble any context for Step 4 - {pkg_id}.")


                # --- Create Hierarchical Task ---
                # Manager agent is implicit in hierarchical process
                task = create_hierarchical_define_mapping_task(
                    manager_agent=None, # Manager is implicit
                    context=context_str,
                    package_id=pkg_id,
                    instructions=instr_context
                )

                # --- Create and run Hierarchical Crew ---
                crew = Crew(
                    agents=[ # Pass agents instantiated in this method
                        cpp_analyst_agent,
                        godot_analyst_agent,
                        strategist_agent,
                        decomposer_agent,
                        formatter_agent
                    ],
                    tasks=[task], # Single task for the manager
                    process=Process.hierarchical,
                    manager_llm=manager_llm, # Pass manager LLM instance
                    planning_llm=manager_llm, # Use manager for planning too
                    planning=True,
                    memory=True,
                    verbose=True,
                    task_callback=self._log_step4_task_completion # Add the callback
                )
                logger.info(f"Kicking off Hierarchical Crew for Step 4 (Package: {pkg_id}, Remapping: {is_remapping_run})...")
                result = crew.kickoff() # Expecting final JSON string from FormatterAgent or CrewOutput
                logger.info(f"Step 4 Hierarchical Crew finished for package: {pkg_id}")
                logger.debug(f"Crew Result Raw Output (Type: {type(result)}):\n{result}")

                # --- Result Extraction and Parsing & Validation ---
                mapping_data_dict = None
                raw_output_for_parsing = None # Store the string to be parsed

                if isinstance(result, CrewOutput):
                    logger.info(f"Crew returned CrewOutput object for {pkg_id}. Extracting raw output.")
                    if hasattr(result, 'raw') and isinstance(result.raw, str):
                        raw_output_for_parsing = result.raw
                        logger.debug(f"Extracted raw output from CrewOutput.raw")
                    elif result.tasks_output and isinstance(result.tasks_output, list) and len(result.tasks_output) > 0:
                         last_task_output = result.tasks_output[-1]
                         if isinstance(last_task_output, TaskOutput) and isinstance(last_task_output.raw_output, str):
                              raw_output_for_parsing = last_task_output.raw_output
                              logger.warning("Extracted raw output from last task in CrewOutput.tasks_output (fallback).")
                         else:
                              logger.error("Could not extract raw string output from CrewOutput's last task.")
                              raw_output_for_parsing = str(result)
                    else:
                         logger.error("CrewOutput object did not contain expected 'raw' string or usable 'tasks_output'.")
                         raw_output_for_parsing = str(result)

                elif isinstance(result, MappingOutput): # Check for Pydantic model
                    logger.info(f"Crew returned validated Pydantic object for {pkg_id}.")
                    mapping_data_dict = result.model_dump()
                elif isinstance(result, str): # Direct string output
                    logger.info(f"Crew returned a string directly for {pkg_id}.")
                    raw_output_for_parsing = result
                elif isinstance(result, dict): # Direct dict output
                     logger.warning(f"Crew returned a dictionary directly for {pkg_id}. Attempting validation.")
                     try:
                          MappingOutput(**result)
                          logger.info("Crew output dict conforms to MappingOutput model.")
                          mapping_data_dict = result
                     except Exception as pydantic_err:
                          logger.error(f"Crew output dict does NOT conform to MappingOutput model: {pydantic_err}")
                          raw_output_for_parsing = json.dumps(result)
                else:
                    logger.error(f"Unexpected final crew output type for {pkg_id}: {type(result)}")
                    raw_output_for_parsing = str(result)

                # --- Parse the extracted raw string if necessary ---
                if raw_output_for_parsing is not None and mapping_data_dict is None:
                    logger.info(f"Attempting to parse extracted/received raw output string as JSON.")
                    # Use the imported parse_json_from_string utility
                    parsed_dict_attempt = parse_json_from_string(raw_output_for_parsing)
                    if parsed_dict_attempt and isinstance(parsed_dict_attempt, dict):
                        try:
                            MappingOutput(**parsed_dict_attempt) # Validate structure
                            logger.info("Parsed JSON string conforms to MappingOutput model.")
                            mapping_data_dict = parsed_dict_attempt
                        except Exception as pydantic_err:
                            logger.error(f"Parsed JSON string does NOT conform to MappingOutput model: {pydantic_err}", exc_info=True)
                            logger.debug(f"Invalid JSON string received: {raw_output_for_parsing}")
                    else:
                        logger.error(f"Failed to parse string output as JSON dictionary for {pkg_id}.")
                        logger.debug(f"String content that failed parsing: {raw_output_for_parsing}")

                # --- Final Check ---
                if not mapping_data_dict:
                    logger.debug(f"Final raw output for {pkg_id} before failing: {raw_output_for_parsing}")
                    raise ValueError(f"Step 4 Hierarchical Crew failed to produce valid mapping data for {pkg_id} after parsing.")

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
    def _log_step4_task_completion(self, task_output: Any): # Use Any for initial check
        """Logs information about each completed task within the Step 4 crew."""
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
                     logger.warning(f"[Step 4 Crew Callback] task_output.agent is not an Agent object (Type: {type(task_output.agent)}).")
                     agent_role = f"Agent (Type: {type(task_output.agent).__name__})"
                else:
                     logger.warning("[Step 4 Crew Callback] task_output object missing 'agent' attribute.")

                # Access task description safely
                if hasattr(task_output, 'task') and task_output.task and hasattr(task_output.task, 'description'):
                    task_desc_snippet = task_output.task.description[:100] + "..."
                elif hasattr(task_output, 'description') and isinstance(task_output.description, str):
                     task_desc_snippet = task_output.description[:100] + "..."
                else:
                     logger.warning("[Step 4 Crew Callback] Could not determine task description from task_output.")

                # Access raw_output safely
                if hasattr(task_output, 'raw_output') and isinstance(task_output.raw_output, str):
                    output_snippet = task_output.raw_output[:150].replace('\n', ' ') + "..."
                elif hasattr(task_output, 'output') and isinstance(task_output.output, str):
                     output_snippet = task_output.output[:150].replace('\n', ' ') + "..."
                else:
                     logger.warning("[Step 4 Crew Callback] Could not determine raw output string from task_output.")

            else:
                logger.warning(f"[Step 4 Crew Callback] Received unexpected type for task_output: {type(task_output)}. Cannot extract details.")
                try:
                    output_snippet = str(task_output)[:150].replace('\n', ' ') + "..."
                except Exception:
                    output_snippet = "[Could not get string representation]"

            # --- Log Information ---
            logger.info(f"[Step 4 Crew Callback] Task Update:")
            logger.info(f"  - Agent: {agent_role}")
            logger.info(f"  - Task Desc (Start): {task_desc_snippet}")
            logger.info(f"  - Output (Start): {output_snippet}")

        except Exception as e:
            logger.error(f"[Step 4 Crew Callback] Error processing task output: {e}", exc_info=True)
            try:
                 logger.debug(f"Raw task_output object during error: {task_output}")
            except Exception:
                 logger.debug("Could not log raw task_output object during error.")
