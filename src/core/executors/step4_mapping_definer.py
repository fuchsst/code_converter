# src/core/executors/step4_mapping_definer.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from src.agents.mapping_definer import MappingDefinerAgent
from src.tasks.define_mapping import DefineMappingTask, MappingOutput
from src.utils.parser_utils import parse_json_from_string
from src.utils.formatting_utils import format_structure_to_markdown
from crewai import Crew, Process
from crewai import LLM as CrewAI_LLM # Alias default LLM
from src.llms.google_genai_llm import GoogleGenAI_LLM # Import custom Gemini LLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

class Step4Executor(StepExecutor):
    """Executes Step 4: C++ to Godot Mapping Definition."""

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
        feedback_override = kwargs.get('feedback_override', {}) # Keep feedback for potential future use
        logger.info(f"--- Starting Step 4 Execution: Define Mapping (Packages: {package_ids or 'All Eligible'}, Force={force}) ---")

        # --- Identify Eligible Packages ---
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
             logger.error("GODOT_PROJECT_DIR not found in config. Cannot scan for existing output files.")
             self.state_manager.update_workflow_status('failed_step4', "GODOT_PROJECT_DIR missing in config.")
             return False
        godot_project_dir = Path(godot_project_dir_str).resolve()


        # --- Get LLM Instance ---
        mapper_llm_config = self._get_llm_config('mapper')
        mapper_llm_instance = None
        if mapper_llm_config:
            model_identifier = mapper_llm_config.get("model", "")
            try:
                if model_identifier.startswith("gemini/"):
                    # Use GoogleGenAI_LLM for Gemini, request JSON output via schema
                    mapper_llm_config['timeout'] = config.GEMINI_TIMEOUT # Add timeout
                    mapper_llm_config['response_schema'] = MappingOutput # Pass the Pydantic model
                    mapper_llm_config['response_mime_type'] = "application/json" # Set mime type
                    mapper_llm_instance = GoogleGenAI_LLM(**mapper_llm_config)
                    logger.info(f"Instantiated custom GoogleGenAI_LLM for role 'mapper': {model_identifier} with timeout {config.GEMINI_TIMEOUT}s, JSON output, and MappingOutput schema")
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
             logger.error("Mapper LLM instance could not be created. Cannot execute Step 4.")
             self.state_manager.update_workflow_status('failed_step4', "Mapper LLM not configured or failed to instantiate.")
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

            # --- Define Artifact Names ---
            suffix = "_remapped" if is_remapping_run and pkg_info_before_run.get('status') != target_status else ""
            json_artifact_filename = f"package_{pkg_id}_mapping{suffix}.json"

            existing_mapping_json = None
            referenced_godot_content = {}

            # --- Check for Existing Mapping (for refinement context) ---
            # Look for the non-suffixed version first using StateManager
            original_json_artifact_filename = f"package_{pkg_id}_mapping.json"
            loaded_existing_mapping = self.state_manager.load_artifact(original_json_artifact_filename, expect_json=True)

            if loaded_existing_mapping is not None:
                if isinstance(loaded_existing_mapping, dict):
                    logger.info(f"Found and loaded existing mapping file for refinement: {original_json_artifact_filename}")
                    existing_mapping_json = loaded_existing_mapping # Assign the loaded dict

                    # --- Read Referenced Godot Files from Existing Mapping ---
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
                                # Use ContextManager method which now takes godot_project_dir
                                content = self.context_manager.read_godot_file_content(str(godot_project_dir), relative_to_res)
                                if content is not None:
                                     referenced_godot_content[res_path] = content
                                     logger.debug(f"Read content for referenced Godot file: {res_path}")
                                else:
                                     logger.warning(f"Failed to read content for referenced Godot file: {res_path}")
                            else:
                                logger.warning(f"Referenced file path '{res_path}' does not start with 'res://'. Skipping content read.")
                else:
                    logger.error(f"Loaded existing mapping file {original_json_artifact_filename} but it's not a dictionary. Ignoring.")
                    existing_mapping_json = None # Reset on error

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id) # Get fresh info
                if not pkg_info:
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                # --- Load the structure JSON artifact (Still need direct path for this) ---
                structure_artifact_filename = pkg_info.get('artifacts', {}).get('structure_json')
                if not structure_artifact_filename:
                    raise FileNotFoundError(f"Structure definition JSON artifact missing for package {pkg_id}.")

                structure_json_path = analysis_dir / structure_artifact_filename
                if not structure_json_path.exists():
                     raise FileNotFoundError(f"Structure definition JSON file not found: {structure_json_path}")

                # Load structure content (consider using state_manager.load_artifact here too?)
                # For now, keep direct load for structure as it's tightly coupled to Step 3 artifact name
                try:
                    with open(structure_json_path, 'r', encoding='utf-8') as f:
                        structure_json_content = json.load(f)
                except Exception as struct_load_err:
                     raise FileNotFoundError(f"Failed to load structure JSON file {structure_json_path}: {struct_load_err}")

                # --- Format the loaded structure into Markdown ---
                structure_markdown = format_structure_to_markdown(structure_json_content)


                # --- Determine C++ files for context ---
                primary_files = pkg_info.get('files', [])
                dependency_files = self.context_manager._get_dependencies_for_package(primary_files)

                # --- Assemble context for the agent ---
                context_kwargs = {
                    "primary_relative_paths": primary_files,
                    "dependency_relative_paths": dependency_files,
                    "work_package_id": pkg_id,
                    "work_package_description": pkg_info.get('description', ''),
                    "proposed_godot_structure_md": structure_markdown, # Pass formatted Markdown
                    "global_packages_summary": packages_json_content, # packages.json
                    "existing_godot_outputs": existing_godot_outputs, # Files in target dir
                    "existing_mapping_json": existing_mapping_json, # Loaded mapping file (if exists)
                    "referenced_godot_content": referenced_godot_content # Content of files in existing mapping
                }
                if pkg_id in feedback_override:
                     context_kwargs["previous_mapping_feedback"] = feedback_override[pkg_id]
                     logger.info(f"Adding remapping feedback for package {pkg_id}.")

                step_name_log = "MAPPING_DEFINITION_REFINEMENT" if existing_mapping_json else "MAPPING_DEFINITION"
                context = self.context_manager.get_context_for_step(step_name_log, **context_kwargs)

                if not context:
                     raise ValueError("Failed to assemble context for Step 4.")

                # --- Instantiate Agent and Task ---
                agent = MappingDefinerAgent().get_agent(llm_instance=mapper_llm_instance)
                task = DefineMappingTask().create_task(agent, context)

                # --- Create and run Crew ---
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=mapper_llm_instance,
                    process=Process.sequential,
                    verbose=True
                )
                logger.info(f"Kicking off Crew for Step 4 (Package: {pkg_id}, Remapping: {is_remapping_run})...")
                result = crew.kickoff() # Expecting MappingOutput object or JSON string
                logger.info(f"Step 4 Crew finished for package: {pkg_id}")

                # --- Result Parsing & Validation ---
                mapping_data_dict = None
                raw_output_for_debug = None

                if isinstance(result, MappingOutput):
                    logger.info(f"Crew returned validated Pydantic object for {pkg_id}.")
                    mapping_data_dict = result.model_dump() # Convert Pydantic to dict
                elif isinstance(result, str):
                    logger.warning(f"Crew returned a string for {pkg_id}. Attempting to parse as JSON.")
                    raw_output_for_debug = result
                    parsed_dict_attempt = parse_json_from_string(result)
                    if parsed_dict_attempt:
                        try:
                            MappingOutput(**parsed_dict_attempt) # Validate structure
                            logger.info("Parsed JSON string conforms to MappingOutput model.")
                            mapping_data_dict = parsed_dict_attempt
                        except Exception as pydantic_err:
                            logger.error(f"Parsed JSON does NOT conform to MappingOutput model: {pydantic_err}")
                    else:
                        logger.error(f"Failed to parse string output as JSON for {pkg_id}.")
                else:
                    # Handle unexpected types
                    logger.error(f"Step 4 Crew did not return a MappingOutput object or a parsable JSON string. Type: {type(result)}")
                    raw_output_for_debug = str(result)

                if not mapping_data_dict:
                    logger.debug(f"Raw output for {pkg_id} (if available): {raw_output_for_debug}")
                    raise ValueError("Step 4 Crew did not produce a valid mapping output dictionary.")

                # --- Save JSON artifact using StateManager ---
                save_json_ok = self.state_manager.save_artifact(json_artifact_filename, mapping_data_dict, is_json=True)
                if not save_json_ok:
                     raise IOError(f"Failed to save structured mapping artifact via StateManager: {json_artifact_filename}")

                # --- Update Overall Mapping Summary ---
                overall_mapping_summary["package_summaries"][pkg_id] = {
                    "mapping_file": json_artifact_filename,
                }
                # Extract output files from the validated dictionary
                if "task_groups" in mapping_data_dict:
                    for group in mapping_data_dict.get("task_groups", []):
                        if isinstance(group, dict):
                            for task_item in group.get("tasks", []):
                                if isinstance(task_item, dict) and "output_godot_file" in task_item:
                                    overall_mapping_summary["all_output_files"].add(task_item["output_godot_file"])
                logger.debug(f"Updated overall mapping summary for package {pkg_id}.")

                # --- Update state ---
                artifacts_to_update = {
                    'mapping_json': json_artifact_filename # Use new key for structured JSON
                }
                self.state_manager.update_package_state(
                    pkg_id,
                    status=completed_status, # 'mapping_defined'
                    artifacts=artifacts_to_update,
                    increment_remap_attempt=is_remapping_run
                )
                logger.info(f"Package {pkg_id} successfully (re)mapped. Status set to '{completed_status}'.")

                # --- Save Overall Mapping Summary (Inside Loop After Successful Package) ---
                try:
                    # Create a copy and convert set to sorted list for JSON serialization
                    overall_mapping_to_save = overall_mapping_summary.copy()
                    overall_mapping_to_save["all_output_files"] = sorted(list(overall_mapping_summary["all_output_files"]))
                    save_overall_ok = self.state_manager.save_artifact(overall_mapping_filename, overall_mapping_to_save, is_json=True)
                    if save_overall_ok:
                        logger.info(f"Saved intermediate overall mapping summary ({overall_mapping_filename}) after processing {pkg_id}.")
                    else:
                        # Log error but don't fail the package processing just for this
                        logger.error(f"Failed to save intermediate overall mapping summary ({overall_mapping_filename}) after processing {pkg_id}.")
                except Exception as e_save:
                    logger.error(f"Unexpected error saving intermediate overall mapping summary after {pkg_id}: {e_save}", exc_info=True)


            except Exception as e:
                logger.error(f"An error occurred during Step 4 processing for package {pkg_id}: {e}", exc_info=True)
                fail_status = failed_status_prefix_remapping if is_remapping_run else failed_status_prefix_mapping
                self.state_manager.update_package_state(pkg_id, status=fail_status, error=str(e))
                overall_success_this_run = False

        # --- Final Workflow Status Check ---
        all_potential_targets_done_or_failed = True
        final_packages_state = self.state_manager.get_all_packages() # Re-fetch latest state
        for pkg_id in potential_target_package_ids:
             status = final_packages_state.get(pkg_id, {}).get('status')
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
