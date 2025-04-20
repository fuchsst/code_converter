# src/core/executors/step4_mapping_definer.py
import os
import json
from typing import Any, Dict, List, Optional
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from src.agents.mapping_definer import MappingDefinerAgent
from src.tasks.define_mapping import DefineMappingTask
from src.utils.parser_utils import parse_step4_output
from crewai import Crew, Process, LLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

class Step4Executor(StepExecutor):
    """Executes Step 4: C++ to Godot Mapping Definition."""

    def execute(self, package_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Runs the C++ to Godot mapping definition for specified or all eligible packages.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process.
                                                If None, processes all eligible packages.
            **kwargs: Accepts 'feedback_override' dict mapping package_id to feedback string
                      for remapping scenarios.

        Returns:
            bool: True if mapping definition was successful for all processed packages, False otherwise.
        """
        feedback_override = kwargs.get('feedback_override', {})
        logger.info(f"--- Starting Step 4 Execution: Define Mapping (Packages: {package_ids or 'All Eligible'}) ---")
        # If specific IDs are provided for remapping, use them directly. Otherwise, find eligible.
        if package_ids:
             eligible_packages = package_ids # Assume provided IDs are the ones to process (for remapping)
             logger.info(f"Processing specific packages provided: {package_ids}")
        else:
             eligible_packages = self._get_eligible_packages(target_status='structure_defined')

        if not eligible_packages:
            logger.warning("No eligible packages found for Step 4.")
            return True # Indicate nothing failed

        self.state_manager.update_workflow_status('running_step4')
        overall_success = True
        analysis_dir = self.config.get("ANALYSIS_OUTPUT_DIR", "analysis_output")

        # Get the LLM config and instantiate the LLM object
        mapper_llm_config = self._get_llm_config('mapper')
        mapper_llm_instance = None
        if mapper_llm_config:
            try:
                mapper_llm_instance = LLM(**mapper_llm_config)
                logger.info(f"Instantiated crewai.LLM for role 'mapper': {mapper_llm_config.get('model')}")
            except Exception as e:
                logger.error(f"Failed to instantiate crewai.LLM for role 'mapper': {e}", exc_info=True)
        else:
            logger.error("Mapper LLM configuration ('mapper') not found.")

        if not mapper_llm_instance:
             logger.error("Mapper LLM instance could not be created. Cannot execute Step 4.")
             self.state_manager.update_workflow_status('failed_step4', "Mapper LLM not configured or failed to instantiate.")
             return False

        for pkg_id in eligible_packages:
            logger.info(f"Processing Step 4 for package: {pkg_id}")
            # Check if this is a remapping attempt and update status accordingly
            is_remapping = pkg_id in feedback_override
            current_status = 'running_remapping' if is_remapping else 'running_mapping'
            self.state_manager.update_package_state(pkg_id, status=current_status)

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                # Load the structure JSON artifact
                structure_artifact_filename = pkg_info.get('artifacts', {}).get('structure_json')
                if not structure_artifact_filename:
                    raise FileNotFoundError(f"Structure definition JSON artifact missing for package {pkg_id}.")

                structure_json_path = os.path.join(analysis_dir, structure_artifact_filename)
                if not os.path.exists(structure_json_path):
                     raise FileNotFoundError(f"Structure definition JSON file not found: {structure_json_path}")

                with open(structure_json_path, 'r', encoding='utf-8') as f:
                    structure_json_content = json.load(f)

                # Determine files for context
                primary_files = pkg_info.get('files', [])
                # Determine dependencies using the existing method on ContextManager
                if hasattr(self.context_manager, '_get_dependencies_for_package'):
                     dependency_files = self.context_manager._get_dependencies_for_package(primary_files) # Pass the list of files
                else:
                     # This case should ideally not happen if ContextManager is correctly initialized
                     logger.error(f"Critical: Method '_get_dependencies_for_package' not found on ContextManager. Cannot determine dependencies for {pkg_id}.")
                     dependency_files = []

                # Assemble context, including feedback if remapping
                context_kwargs = {
                    "primary_relative_paths": primary_files,
                    "dependency_relative_paths": dependency_files,
                    "work_package_id": pkg_id,
                    "work_package_description": pkg_info.get('description', ''),
                    "proposed_godot_structure": structure_json_content
                }
                step_name = "MAPPING_DEFINITION"
                if is_remapping:
                     context_kwargs["previous_mapping_feedback"] = feedback_override[pkg_id]
                     step_name = "MAPPING_DEFINITION_WITH_FEEDBACK"

                context = self.context_manager.get_context_for_step(step_name, **context_kwargs)

                if not context:
                     raise ValueError("Failed to assemble context for Step 4.")

                # Instantiate Agent and Task
                agent = MappingDefinerAgent().get_agent(llm_instance=mapper_llm_instance)
                task = DefineMappingTask().create_task(agent, context)

                # Create and run Crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=mapper_llm_instance,
                    process=Process.sequential,
                    verbose=True
                )
                logger.info(f"Kicking off Crew for Step 4 (Package: {pkg_id}, Remapping: {is_remapping})...")
                # Expecting combined Markdown + JSON string
                combined_output = crew.kickoff()
                logger.info(f"Step 4 Crew finished for package: {pkg_id}")

                if not combined_output or not isinstance(combined_output, str):
                     raise ValueError("Step 4 Crew did not return a valid string output.")

                # Parse the combined output
                mapping_strategy_md, task_list_json = parse_step4_output(combined_output)

                if mapping_strategy_md is None or task_list_json is None:
                    logger.error(f"Failed to parse Step 4 combined output for package {pkg_id}.")
                    logger.debug(f"Raw output: {combined_output}")
                    raise ValueError("Parsing Step 4 output failed (separator missing or JSON invalid).")

                # Save artifacts (potentially add suffix if remapping)
                suffix = "_remapped" if is_remapping else ""
                md_artifact_filename = f"package_{pkg_id}_mapping{suffix}.md"
                json_artifact_filename = f"package_{pkg_id}_tasks{suffix}.json"
                md_artifact_path = os.path.join(analysis_dir, md_artifact_filename)
                json_artifact_path = os.path.join(analysis_dir, json_artifact_filename)

                try:
                    os.makedirs(analysis_dir, exist_ok=True) # Ensure dir exists
                    with open(md_artifact_path, 'w', encoding='utf-8') as f:
                        f.write(mapping_strategy_md)
                    logger.info(f"Saved mapping strategy artifact: {md_artifact_path}")

                    with open(json_artifact_path, 'w', encoding='utf-8') as f:
                        json.dump(task_list_json, f, indent=2)
                    logger.info(f"Saved task list artifact: {json_artifact_path}")

                    # Update state: set status back to 'mapping_defined' and update artifacts
                    # If remapping, also store the incremented attempt count
                    artifacts_to_update = {
                        'mapping_md': md_artifact_filename,
                        'tasks_json': json_artifact_filename
                    }
                    self.state_manager.update_package_state(
                        pkg_id,
                        status='mapping_defined',
                        artifacts=artifacts_to_update,
                        increment_remap_attempt=is_remapping # Increment only if it was a remapping run
                    )
                    logger.info(f"Package {pkg_id} successfully (re)mapped. Ready for Step 5.")

                except IOError as e:
                    raise IOError(f"Failed to save Step 4 artifacts for package {pkg_id}: {e}")

            except Exception as e:
                logger.error(f"An error occurred during Step 4 for package {pkg_id}: {e}", exc_info=True)
                fail_status = 'failed_remapping' if is_remapping else 'failed_mapping'
                self.state_manager.update_package_state(pkg_id, status=fail_status, error=str(e))
                overall_success = False

        # Update overall status only if processing all eligible (non-remapping run)
        if overall_success and not package_ids:
             all_processed_or_skipped = True
             final_packages_state = self.state_manager.get_all_packages()
             for pkg_id in eligible_packages: # Check only those that were meant to be processed
                 status = final_packages_state.get(pkg_id, {}).get('status')
                 if status not in ['mapping_defined', 'processed']: # Add future states if needed
                      all_processed_or_skipped = False
                      break
             if all_processed_or_skipped:
                  self.state_manager.update_workflow_status('step4_complete')
        elif not overall_success:
             # If any package failed, mark the step as failed
             self.state_manager.update_workflow_status('failed_step4', "One or more packages failed during mapping definition.")

        logger.info("--- Finished Step 4 Execution ---")
        return overall_success
