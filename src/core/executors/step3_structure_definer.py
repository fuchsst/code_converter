# src/core/executors/step3_structure_definer.py
import os
import json
from typing import Any, Dict, List, Optional
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
from src.agents.structure_definer import StructureDefinerAgent
from src.tasks.define_structure import DefineStructureTask
from crewai import Crew, Process
from logger_setup import get_logger

logger = get_logger(__name__)

class Step3Executor(StepExecutor):
    """Executes Step 3: Godot Structure Definition."""

    def execute(self, package_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Runs the Godot structure definition for specified or all eligible packages.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process.
                                                If None, processes all eligible packages.
            **kwargs: Not used in this step.

        Returns:
            bool: True if structure definition was successful for all processed packages, False otherwise.
        """
        logger.info(f"--- Starting Step 3 Execution: Define Structure (Packages: {package_ids or 'All Eligible'}) ---")
        eligible_packages = self._get_eligible_packages(target_status='identified', specific_ids=package_ids)

        if not eligible_packages:
            logger.warning("No eligible packages found for Step 3.")
            # If specific packages were requested and none were eligible, maybe return True?
            # If no specific packages requested and none eligible, it's effectively success (nothing to do).
            return True # Indicate nothing failed

        self.state_manager.update_workflow_status('running_step3')
        overall_success = True
        analysis_dir = self.config.get("ANALYSIS_OUTPUT_DIR", "analysis_output")

        # Get the LLM instance for the mapper role
        mapper_llm = self._get_llm('mapper') # Assuming 'mapper' is the key in llm_map
        if not mapper_llm:
             logger.error("Mapper LLM instance not found. Cannot execute Step 3.")
             self.state_manager.update_workflow_status('failed_step3', "Mapper LLM not configured.")
             return False

        for pkg_id in eligible_packages:
            logger.info(f"Processing Step 3 for package: {pkg_id}")
            self.state_manager.update_package_state(pkg_id, status='running_structure')

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info:
                     raise ValueError(f"Could not retrieve package info for {pkg_id} from state.")

                primary_files = pkg_info.get('files', [])
                # Determine dependencies using the existing method on ContextManager
                if hasattr(self.context_manager, '_get_dependencies_for_package'):
                     dependency_files = self.context_manager._get_dependencies_for_package(primary_files) # Pass the list of files
                else:
                     # This case should ideally not happen if ContextManager is correctly initialized
                     logger.error(f"Critical: Method '_get_dependencies_for_package' not found on ContextManager. Cannot determine dependencies for {pkg_id}.")
                     dependency_files = [] # Proceed without dependencies

                # Assemble context
                context = self.context_manager.get_context_for_step(
                    step_name="STRUCTURE_DEFINITION",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    # Pass package info as other context
                    work_package_id=pkg_id,
                    work_package_description=pkg_info.get('description', ''),
                    work_package_files=primary_files # Send file list again for clarity
                )

                if not context:
                     raise ValueError("Failed to assemble context for Step 3.")

                # Instantiate Agent and Task
                agent = StructureDefinerAgent().get_agent()
                task = DefineStructureTask().create_task(agent, context)

                # Create and run Crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=mapper_llm, # Use the mapper LLM
                    process=Process.sequential,
                    verbose=2 # Or use config value
                )
                logger.info(f"Kicking off Crew for Step 3 (Package: {pkg_id})...")
                # Expecting JSON object directly due to output_json=True in the task
                result_json = crew.kickoff()
                logger.info(f"Step 3 Crew finished for package: {pkg_id}")

                if not result_json or not isinstance(result_json, dict):
                     logger.error(f"Step 3 Crew did not return a valid JSON dictionary. Type: {type(result_json)}")
                     logger.debug(f"Raw output (if available): {result_json}")
                     raise ValueError("Step 3 Crew did not return a valid JSON dictionary.")

                # Save the artifact as JSON
                artifact_filename = f"package_{pkg_id}_structure.json"
                artifact_path = os.path.join(analysis_dir, artifact_filename)
                try:
                    os.makedirs(analysis_dir, exist_ok=True) # Ensure dir exists
                    with open(artifact_path, 'w', encoding='utf-8') as f:
                        json.dump(result_json, f, indent=2) # Save as JSON
                    logger.info(f"Saved structure definition artifact: {artifact_path}")
                    # Update state with new artifact key
                    self.state_manager.update_package_state(pkg_id, status='structure_defined', artifacts={'structure_json': artifact_filename})
                except IOError as e:
                    raise IOError(f"Failed to save structure artifact {artifact_path}: {e}")
                except TypeError as e:
                     raise TypeError(f"Failed to serialize structure result to JSON: {e}")

            except Exception as e:
                logger.error(f"An error occurred during Step 3 for package {pkg_id}: {e}", exc_info=True)
                self.state_manager.update_package_state(pkg_id, status='failed_structure', error=str(e))
                overall_success = False
            # No finally block needed here as state is saved within update_package_state

        # Update overall status only if processing all packages
        if overall_success and not package_ids:
             # Check if ALL initially eligible packages are now 'structure_defined' or beyond
             all_processed_or_skipped = True
             final_packages_state = self.state_manager.get_all_packages()
             for pkg_id in eligible_packages: # Check only those that were meant to be processed
                 status = final_packages_state.get(pkg_id, {}).get('status')
                 if status not in ['structure_defined', 'mapping_defined', 'processed']: # Add future states if needed
                      all_processed_or_skipped = False
                      break
             if all_processed_or_skipped:
                  self.state_manager.update_workflow_status('step3_complete')
        elif not overall_success:
             # If any package failed, mark the step as failed
             self.state_manager.update_workflow_status('failed_step3', "One or more packages failed during structure definition.")

        logger.info("--- Finished Step 3 Execution ---")
        return overall_success
