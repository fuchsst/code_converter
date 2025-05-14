# src/core/executors/step4_mapping_executor.py
from typing import List, Dict, Any, Optional, Set, Type

from src.core.state_manager import StateManager
from src.core.context_manager import ContextManager
from src.core.step_executor import StepExecutor
from src.flows.mapping_flow import MappingFlow
from src.utils.json_utils import parse_json_from_string
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

class Step4MappingExecutor(StepExecutor):
    """Executes Step 4: C++ to Godot Mapping Definition using CrewAI Flow."""

    def __init__(
        self,
        state_manager: StateManager,
        context_manager: ContextManager,
        config_dict: Dict[str, Any],
        llm_configs: Dict[str, Dict[str, Any]],
        tools: Dict[Type, Any]
    ):
        super().__init__(state_manager, context_manager, config_dict, llm_configs, tools)
        logger.info("Step4MappingExecutor initialized.")

    def execute(
        self, 
        package_ids: Optional[List[str]] = None, 
        force: bool = False, 
        **kwargs
    ) -> bool:
        """
        Runs the C++ to Godot mapping definition for specified or all eligible packages.

        Args:
            package_ids: Specific package IDs to process. If None, processes all eligible packages.
            force: If True, forces reprocessing of packages even if already mapped or failed.
            **kwargs: Accepts 'feedback_override' dict mapping package_id to feedback string
                      for remapping scenarios.

        Returns:
            True if mapping definition was successful for the processed packages in this run, False otherwise.
        """
        feedback_override = kwargs.get('feedback_override', {})
        logger.info(f"--- Starting Step 4 Execution: Define Mapping (Packages: {package_ids or 'All Eligible'}, Force={force}) ---")

        # --- Create LLM configurations for the flow ---
        llm_config = {
            "ANALYZER_MODEL": self._create_llm_instance('ANALYZER_MODEL'),
            "DESIGNER_PLANNER_MODEL": self._create_llm_instance('DESIGNER_PLANNER_MODEL'),
            "UTILITY_MODEL": self._create_llm_instance('UTILITY_MODEL')
        }

        # Check if all required LLMs were instantiated successfully
        missing_llms = [name for name, llm in llm_config.items() if not llm]
        if missing_llms:
            logger.error(f"Missing critical LLM configurations for Step 4: {', '.join(missing_llms)}. Cannot proceed.")
            self.state_manager.update_workflow_status('failed_step4', f"Missing LLM config for: {', '.join(missing_llms)}")
            return False

        # --- Identify Eligible Packages ---
        packages_to_process = self._identify_eligible_packages(package_ids, force)
        if not packages_to_process:
            logger.info("No packages require processing in this Step 4 run.")
            return True

        logger.info(f"Packages to process in this Step 4 run (Force={force}): {packages_to_process}")
        self.state_manager.update_workflow_status('running_step4')
        
        # --- Initialize Overall Mapping Summary ---
        overall_mapping_summary = self._initialize_mapping_summary(force)

        # --- Process Each Package ---
        overall_success = True
        for pkg_id in packages_to_process:
            try:
                success = self._process_package(
                    pkg_id=pkg_id,
                    llm_config=llm_config,
                    feedback_override=feedback_override,
                    overall_mapping_summary=overall_mapping_summary,
                    force=force
                )
                if not success:
                    overall_success = False
            except Exception as e:
                logger.error(f"An error occurred during Step 4 processing for package {pkg_id}: {e}", exc_info=True)
                overall_success = False
                # Determine if this was a remapping attempt
                pkg_data = self.state_manager.get_package_info(pkg_id)
                current_status = pkg_data.get('status', '')
                is_remapping = 'remapping' in current_status
                failure_status = 'failed_remapping' if is_remapping else 'failed_mapping'
                self.state_manager.update_package_state(pkg_id, status=failure_status, error=str(e))

        # --- Update Workflow Status ---
        self._update_workflow_status(packages_to_process, overall_success)
        
        logger.info(f"--- Finished Step 4 Execution Run (Success: {overall_success}) ---")
        return overall_success
        
    def _identify_eligible_packages(self, package_ids: Optional[List[str]], force: bool) -> List[str]:
        """Identifies packages eligible for processing."""
        packages_to_process = []
        potential_target_package_ids = set()
        
        # Status constants
        target_status = 'structure_defined'
        failed_mapping_status = 'failed_mapping'
        failed_remapping_status = 'failed_remapping'
        needs_remapping_status = 'needs_remapping'
        completed_status = 'mapping_defined'
        
        all_packages = self.state_manager.get_all_packages()
        if not all_packages:
            logger.warning("No packages found in state. Cannot proceed with Step 4.")
            self.state_manager.update_workflow_status('step4_complete')
            return []
        
        # Identify potential targets
        for pkg_id, pkg_data in all_packages.items():
            if not package_ids or pkg_id in package_ids:
                current_status = pkg_data.get('status', '')
                
                # Check if package should be processed
                is_target = current_status == target_status
                is_needs_remapping = current_status == needs_remapping_status
                is_failed = current_status.startswith(failed_mapping_status) or current_status.startswith(failed_remapping_status)
                is_completed = current_status == completed_status
                
                if is_target or is_needs_remapping or (force and (is_failed or is_completed)):
                    potential_target_package_ids.add(pkg_id)
                    
                    # Reset status for forced packages
                    if force and (is_failed or is_completed):
                        logger.info(f"Force=True: Resetting package '{pkg_id}' status to '{target_status}'")
                        self.state_manager.update_package_state(pkg_id, target_status, error=None)
                    
                    packages_to_process.append(pkg_id)
        
        return packages_to_process
    
    def _initialize_mapping_summary(self, force: bool) -> Dict[str, Any]:
        """Initializes or loads the overall mapping summary."""
        overall_mapping_summary = {
            "package_summaries": {},
            "all_output_files": set()
        }
        
        # Load existing mapping if not forcing rebuild
        if not force:
            existing_summary = self.state_manager.load_artifact("mappings.json", expect_json=True)
            if existing_summary and isinstance(existing_summary, dict):
                if "package_summaries" in existing_summary and "all_output_files" in existing_summary:
                    overall_mapping_summary = existing_summary
                    overall_mapping_summary["all_output_files"] = set(overall_mapping_summary.get("all_output_files", []))
                    logger.info("Loaded existing overall mapping summary via StateManager.")
                    
        return overall_mapping_summary
    
    def _process_package(
        self, 
        pkg_id: str, 
        llm_config: Dict[str, Any],
        feedback_override: Dict[str, str],
        overall_mapping_summary: Dict[str, Any],
        force: bool
    ) -> bool:
        """Processes a single package."""
        logger.info(f"Processing Step 4 for package: {pkg_id}")
        
        # Determine if this is a remapping run
        pkg_info = self.state_manager.get_package_info(pkg_id)
        current_status = pkg_info.get('status', '')
        is_remapping = 'remapping' in current_status or (force and current_status == 'mapping_defined')
        
        # Update status
        status_to_set = 'running_remapping' if is_remapping else 'running_mapping'
        self.state_manager.update_package_state(pkg_id, status=status_to_set)
        
        # Define artifact filename
        suffix = "_remapped" if is_remapping else ""
        json_artifact_filename = f"package_{pkg_id}_mapping{suffix}.json"
        
        # Get feedback if available
        feedback = feedback_override.get(pkg_id, None)
        
        # Create and run the mapping flow
        try:
            mapping_flow = MappingFlow(
                state_manager=self.state_manager,
                context_manager=self.context_manager,
                llm_config=llm_config,
                package_id=pkg_id,
                is_remapping=is_remapping,
                feedback=feedback
            )
            
            # Run the flow and get the mapping output
            mapping_output = mapping_flow.run()
            
            # Save the mapping output
            save_json_ok = self.state_manager.save_artifact(
                json_artifact_filename, 
                mapping_output, 
                is_json=True
            )
            
            if not save_json_ok:
                raise IOError(f"Failed to save structured mapping artifact: {json_artifact_filename}")
            
            # Update overall mapping summary
            overall_mapping_summary["package_summaries"][pkg_id] = {"mapping_file": json_artifact_filename}
            
            if "task_groups" in mapping_output:
                for group in mapping_output.get("task_groups", []):
                    if isinstance(group, dict):
                        for task_item in group.get("tasks", []):
                            if isinstance(task_item, dict) and "output_godot_file" in task_item:
                                overall_mapping_summary["all_output_files"].add(task_item["output_godot_file"])
            
            # Update state
            artifacts_to_update = {'mapping_json': json_artifact_filename}
            self.state_manager.update_package_state(
                pkg_id,
                status='mapping_defined',
                artifacts=artifacts_to_update,
                increment_remap_attempt=is_remapping
            )
            
            # Save overall mapping summary
            self._save_overall_mapping_summary(overall_mapping_summary)
            
            logger.info(f"Package {pkg_id} successfully mapped. Status set to 'mapping_defined'.")
            return True
            
        except Exception as e:
            logger.error(f"Error processing package {pkg_id}: {e}", exc_info=True)
            failure_status = 'failed_remapping' if is_remapping else 'failed_mapping'
            self.state_manager.update_package_state(pkg_id, status=failure_status, error=str(e))
            return False
    
    def _save_overall_mapping_summary(self, summary: Dict[str, Any]) -> bool:
        """Saves the overall mapping summary."""
        try:
            summary_to_save = summary.copy()
            summary_to_save["all_output_files"] = sorted(list(summary["all_output_files"]))
            return self.state_manager.save_artifact("mappings.json", summary_to_save, is_json=True)
        except Exception as e:
            logger.error(f"Failed to save overall mapping summary: {e}", exc_info=True)
            return False
    
    def _update_workflow_status(self, processed_packages: List[str], overall_success: bool) -> None:
        """Updates the workflow status based on processing results."""
        all_packages = self.state_manager.get_all_packages()
        all_done = True
        
        for pkg_id in processed_packages:
            status = all_packages.get(pkg_id, {}).get('status', '')
            if status != 'mapping_defined' and not status.startswith('failed_mapping') and not status.startswith('failed_remapping'):
                all_done = False
                break
        
        if all_done and processed_packages:
            logger.info("All target packages for Step 4 are now processed or failed.")
            current_global_status = self.state_manager.get_state().get('workflow_status', '')
            
            if 'failed' not in current_global_status:
                if overall_success:
                    self.state_manager.update_workflow_status('step4_complete')
                else:
                    self.state_manager.update_workflow_status('failed_step4', "One or more packages failed during mapping definition.")
