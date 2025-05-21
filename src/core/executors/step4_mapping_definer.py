# src/core/executors/step4_mapping_executor.py
import json
from typing import List, Dict, Any, Optional, Set, Type

from src.models.mapping_models import MappingOutput
from src.core.state_manager import StateManager
from src.core.context_manager import ContextManager
from src.core.step_executor import StepExecutor
from src.flows.mapping_flow import DefineMappingPipelineFlow
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
        retry: bool = False, 
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
        logger.info(f"--- Starting Step 4 Execution: Define Mapping (Packages: {package_ids or 'All Eligible'}, Force={retry}) ---")

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
        packages_to_process = self._identify_eligible_packages(package_ids, retry)
        if not packages_to_process:
            logger.info("No packages require processing in this Step 4 run.")
            return True

        logger.info(f"Packages to process in this Step 4 run (Force={retry}): {packages_to_process}")
        self.state_manager.update_workflow_status('running_step4')
        
        # --- Initialize Overall Mapping Summary ---
        overall_mapping_summary = self._initialize_mapping_summary(retry)

        # --- Process Each Package ---
        overall_success = True
        for pkg_id in packages_to_process:
            try:
                success = self._process_package(
                    pkg_id=pkg_id,
                    llm_config=llm_config,
                    feedback_override=feedback_override,
                    overall_mapping_summary=overall_mapping_summary,
                    force=retry
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
        
    def _identify_eligible_packages(self, package_ids: Optional[List[str]], retry: bool) -> List[str]:
        """Identifies packages eligible for processing."""
        packages_to_process = []
        potential_target_package_ids = set()
        
        # Status constants
        target_status = 'structure_defined'
        failed_mapping_status = 'failed_mapping'
        failed_remapping_status = 'failed_remapping'
        needs_remapping_status = 'needs_remapping'
        is_running_status = 'running_mapping'
        
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
                is_completed = current_status == is_running_status
                
                if is_target or is_needs_remapping or (retry and (is_failed or is_running_status)):
                    potential_target_package_ids.add(pkg_id)
                    
                    # Reset status for failed packages
                    if retry and (is_failed or is_running_status):
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
            # Instantiate the renamed flow class
            # The DefineMappingPipelineFlow expects llm, context_manager, state_manager
            # We need to pass the primary LLM instance or adjust the flow's __init__
            # For now, assuming the 'DESIGNER_PLANNER_MODEL' is the primary one for the flow's internal agent setup.
            # The DefineMappingPipelineFlow now has a parameterless __init__ and a configure method.
            mapping_flow = DefineMappingPipelineFlow()
            mapping_flow.configure(
                llm_instances=llm_config,
                context_manager=self.context_manager,
                state_manager=self.state_manager
            )
            
            # Prepare kickoff inputs for the flow
            # Construct a more comprehensive initial_context_str for the flow
            cpp_content = self.context_manager.get_work_package_source_code_content(pkg_id)
            # Ensure target_file_list is serializable if it contains non-standard objects; it should be list of dicts.
            target_files_list = self.context_manager.get_target_file_list(pkg_id)
            try:
                target_files_json = json.dumps(target_files_list)
            except TypeError as e:
                logger.error(f"Could not serialize target_files_list for package {pkg_id}: {e}", exc_info=True)
                target_files_json = "[]" # Default to empty list string on error

            initial_context_parts = [
                f"--- C++ Source Code for Package {pkg_id} ---\n{cpp_content}",
                f"\n\n--- Target Godot File List for Package {pkg_id} ---\n{target_files_json}"
            ]
            # Potentially add more global context or specific instructions if needed by the flow's step0
            # For example, general instructions from context_manager could be added here.
            # general_instructions_str = self.context_manager.get_instruction_context()
            # if general_instructions_str:
            #     initial_context_parts.append(f"\n\n--- General Instructions ---\n{general_instructions_str}")

            initial_context_str_for_flow = "\n".join(initial_context_parts)

            kickoff_inputs = {
                "package_id": pkg_id,
                "initial_context_str": initial_context_str_for_flow,
                # Pass existing mapping and feedback directly if available,
                # so the flow's state can be initialized with them.
                "existing_mapping_for_strategist": json.dumps(self.state_manager.load_artifact(f"package_{pkg_id}_mapping.json", expect_json=True)) if is_remapping else None,
                "feedback_for_strategist": feedback
            }
            
            # Run the flow and get the mapping output
            mapping_output_model = mapping_flow.kickoff(inputs=kickoff_inputs)

            if not mapping_output_model or not isinstance(mapping_output_model, MappingOutput):
                logger.error(f"Mapping flow for package {pkg_id} did not return a valid MappingOutput model.")
                # Check flow state for errors
                flow_state_error = mapping_flow.state.error_message if hasattr(mapping_flow, 'state') and hasattr(mapping_flow.state, 'error_message') else "Unknown flow error"
                raise ValueError(f"Mapping flow failed or returned invalid output. Error: {flow_state_error}")

            mapping_output = mapping_output_model.model_dump() # Convert Pydantic model to dict
            
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
