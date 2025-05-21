# src/core/executors/step5_process_code.py
import os
import json
from typing import Any, Dict, List, Optional, Type, Literal

# CrewAI imports
from crewai import Crew, Process, Task, Agent
from crewai.tasks.task_output import TaskOutput

# Local application imports
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens, read_godot_file_content
from ...tools.remapping_logic import RemappingLogic # Keep for now, might be used by advisor tool directly

import src.config as config

# Agents for package-level analysis
from src.agents.step5.remapping_advisor import get_remapping_advisor_agent

# Tools for package-level analysis
from src.tools.crewai_tools import RemappingLogicTool # FileReaderTool might be used by flow agents

# Tasks for package-level analysis
from src.tasks.step5.process_code import (
    create_analyze_package_failures_task,
    RemappingAdvice,
    TaskItemProcessingResult, # This is the expected output from the flow per item
)

# Flow for item processing
from src.flows.process_code_item_flow import ProcessCodeItemFlow, ProcessCodeItemFlowInput, ProcessCodeItemFlowOutput

from src.utils.json_utils import parse_json_from_string
from src.logger_setup import get_logger
import re # Import re for regex operations

logger = get_logger(__name__)

class Step5Executor(StepExecutor):
    """Executes Step 5: Iterative Conversion & Refinement using ProcessCodeItemFlow."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config_dict: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]],
                 tools: Dict[Type, Any], 
                 remapping_logic: RemappingLogic): # RemappingLogic might not be directly used here anymore
        super().__init__(state_manager, context_manager, config_dict, llm_configs, tools)
        # self.remapping_logic = remapping_logic # No longer directly used by executor
        self.target_dir = os.path.abspath(config.GODOT_PROJECT_DIR or "output/godot_project")
        self.analysis_dir = os.path.abspath(config.ANALYSIS_OUTPUT_DIR or "output/analysis")
        logger.info("Step5Executor initialized to use ProcessCodeItemFlow.")


    def execute(self, package_ids: Optional[List[str]] = None, retry: bool = False, **kwargs) -> bool:
        logger.info(f"--- Starting Step 5 Execution (Flow-Based): (Packages: {package_ids or 'All Eligible'}, Retry={retry}) ---")

        # --- Instantiate LLMs and Tools for Package-Level Analysis ---
        remapping_advisor_tool = RemappingLogicTool()
        logger.info("Instantiated RemappingLogicTool for package-level analysis.")

        remapping_advisor_llm = self._create_llm_instance('ANALYZER_MODEL', response_schema_class=RemappingAdvice)
        generator_llm_for_flow = self._create_llm_instance('GENERATOR_REFINER_MODEL')

        required_executor_llms = {'Remapping Advisor': remapping_advisor_llm}
        required_flow_llms = {'GENERATOR_REFINER_MODEL': generator_llm_for_flow}

        missing_executor_llms = [name for name, llm in required_executor_llms.items() if not llm]
        missing_flow_llms = [name for name, llm in required_flow_llms.items() if not llm]

        if missing_executor_llms or missing_flow_llms:
             all_missing_llms = missing_executor_llms + missing_flow_llms
             logger.error(f"Missing critical LLM instances for Step 5: {', '.join(all_missing_llms)}. Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step5', f"Missing LLM config for: {', '.join(all_missing_llms)}")
             return False

        remapping_advisor_agent = get_remapping_advisor_agent(remapping_advisor_llm, tools=[remapping_advisor_tool])
        logger.info("Instantiated RemappingAdvisorAgent for package-level analysis.")

        # --- Identify Eligible Packages ---
        target_status = 'mapping_defined'
        failed_status_prefixes = ['failed_processing', 'failed_remapping']
        processed_status = 'processed'
        packages_to_process_this_run = []
        potential_target_package_ids = set()
        all_packages = self.state_manager.get_all_packages()

        if not all_packages:
             logger.warning("No packages found in state. Cannot proceed with Step 5.")
             self.state_manager.update_workflow_status('step5_complete')
             return True

        step4_completed_or_later_states = {
            target_status, 'running_processing', processed_status, 'needs_remapping',
        } | set(failed_status_prefixes)

        for pkg_id_loop, pkg_data_loop in all_packages.items():
             current_status_loop = pkg_data_loop.get('status')
             is_target_loop = (current_status_loop == target_status)
             is_step4_done_or_later_loop = current_status_loop in step4_completed_or_later_states
             matches_specific_request_loop = (not package_ids or pkg_id_loop in package_ids)
             if matches_specific_request_loop and (is_target_loop or (retry and is_step4_done_or_later_loop)):
                       potential_target_package_ids.add(pkg_id_loop)

        for pkg_id_loop in potential_target_package_ids:
             pkg_data_loop = all_packages[pkg_id_loop]
             current_status_loop = pkg_data_loop.get('status')
             is_target_loop = (current_status_loop == target_status)
             is_running_this_step_loop = (current_status_loop == 'running_processing')
             is_step4_done_or_later_loop = current_status_loop in step4_completed_or_later_states
             if is_target_loop or is_running_this_step_loop:
                  if is_running_this_step_loop: logger.info(f"Resuming processing for package '{pkg_id_loop}'.")
                  packages_to_process_this_run.append(pkg_id_loop)
             elif retry and is_step4_done_or_later_loop:
                  logger.info(f"Force=True: Adding package '{pkg_id_loop}' (status: {current_status_loop}) to process list.")
                  self.state_manager.update_package_state(pkg_id_loop, target_status, error=None)
                  packages_to_process_this_run.append(pkg_id_loop)
        
        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 5 run.")
            return True

        logger.info(f"Eligible packages for this Step 5 run (Force={retry}): {packages_to_process_this_run}")
        processing_order = self.state_manager.get_package_processing_order()
        if not processing_order or not isinstance(processing_order, list):
            logger.error("Critical: Package processing order missing or invalid. Cannot proceed.")
            self.state_manager.update_workflow_status('failed_step5', "Processing order missing/invalid.")
            return False

        self.state_manager.update_workflow_status('running_step5')
        overall_success_this_run = True
        processed_a_package = False

        # --- Process Packages ---
        for pkg_id in processing_order:
            if pkg_id not in packages_to_process_this_run:
                continue
            processed_a_package = True
            logger.info(f"Processing Step 5 for package: {pkg_id}")
            self.state_manager.update_package_state(pkg_id, status='running_processing')
            package_overall_success = True

            try:
                pkg_info = self.state_manager.get_package_info(pkg_id)
                if not pkg_info: raise ValueError(f"No package info for {pkg_id}.")

                mapping_artifact_name = pkg_info.get('artifacts', {}).get('mapping_json')
                if not mapping_artifact_name and retry and pkg_info.get('status','').startswith('failed_'):
                    mapping_artifact_name = f"package_{pkg_id}_mapping.json"
                if not mapping_artifact_name: raise FileNotFoundError(f"Mapping JSON missing for {pkg_id}.")
                
                mapping_data = self.state_manager.load_artifact(mapping_artifact_name, expect_json=True)
                if not mapping_data: raise FileNotFoundError(f"Failed to load mapping: {mapping_artifact_name}")

                task_groups = mapping_data.get("task_groups", [])
                task_items = [item for group in task_groups if isinstance(group, dict) for item in group.get("tasks", []) if isinstance(item, dict)]
                if not task_items:
                    logger.warning(f"No task items in mapping for {pkg_id}. Skipping.")
                    self.state_manager.update_package_state(pkg_id, status='processed', error="No tasks in mapping.")
                    continue

                instr_context = self.context_manager.get_instruction_context()
                base_pkg_context_str = self._build_package_context(pkg_id, pkg_info, mapping_data, instr_context)
                
                godot_project_abs_path = config.GODOT_PROJECT_DIR 
                if not godot_project_abs_path:
                    raise ValueError("GODOT_PROJECT_DIR not configured in src.config.")

                all_item_results_for_package: List[TaskItemProcessingResult] = []
                for task_item_details in task_items:
                    item_id_log = task_item_details.get('task_id', task_item_details.get('task_title', 'unknown_item'))
                    logger.info(f"Starting processing for task item: {item_id_log} in package {pkg_id}")
                    
                    item_specific_context_str = self._build_item_specific_context(
                        base_package_context=base_pkg_context_str,
                        task_item_details=task_item_details,
                        godot_project_path=godot_project_abs_path
                    )

                    # --- Execute ProcessCodeItemFlow for this item ---
                    flow_input_data = ProcessCodeItemFlowInput(
                        package_id=pkg_id,
                        task_item_details=task_item_details,
                        item_context_str=item_specific_context_str,
                        godot_project_path=godot_project_abs_path,
                        general_instructions=instr_context
                    )
                    
                    flow_llm_config = {
                        'GENERATOR_REFINER_MODEL': generator_llm_for_flow,
                    }

                    item_flow = ProcessCodeItemFlow(
                        flow_input=flow_input_data,
                        state_manager=self.state_manager,
                        context_manager=self.context_manager,
                        llm_instances=flow_llm_config
                    )
                    
                    flow_output: ProcessCodeItemFlowOutput = item_flow.run()
                    
                    item_processing_result = TaskItemProcessingResult(
                        task_id=flow_output.task_id,
                        status=flow_output.status,
                        target_godot_file=flow_output.target_godot_file,
                        target_element=flow_output.target_element,
                        error_message=flow_output.error_message
                    )
                    # --- End of Flow Execution ---
                    
                    all_item_results_for_package.append(item_processing_result)
                    if item_processing_result.status == 'failed':
                        package_overall_success = False
                
                # --- Analyze Package Failures ---
                parsed_advice = None
                if all_item_results_for_package:
                    item_results_as_dicts = [result.model_dump() for result in all_item_results_for_package]
                    
                    analysis_task_obj = create_analyze_package_failures_task(
                        advisor_agent=remapping_advisor_agent,
                        item_processing_results=item_results_as_dicts,
                        instructions=instr_context
                    )
                    analysis_crew = Crew(agents=[remapping_advisor_agent], tasks=[analysis_task_obj], process=Process.sequential, memory=False, verbose=True)
                    logger.info(f"Kicking off Analysis Crew for package {pkg_id}...")
                    analysis_crew_kickoff_result = analysis_crew.kickoff()
                    logger.info(f"Analysis Crew for {pkg_id} finished. Kickoff result type: {type(analysis_crew_kickoff_result)}")

                    actual_analysis_output_str = None
                    if isinstance(analysis_crew_kickoff_result, TaskOutput):
                        actual_analysis_output_str = analysis_crew_kickoff_result.raw_output
                    elif isinstance(analysis_crew_kickoff_result, str):
                        actual_analysis_output_str = analysis_crew_kickoff_result
                    elif hasattr(analysis_crew_kickoff_result, 'raw') and isinstance(analysis_crew_kickoff_result.raw, str):
                        actual_analysis_output_str = analysis_crew_kickoff_result.raw
                    else:
                        logger.error(f"Unexpected analysis crew kickoff result type: {type(analysis_crew_kickoff_result)}. Full result: {analysis_crew_kickoff_result}")

                    if actual_analysis_output_str:
                        logger.debug(f"Raw output from analysis crew: {actual_analysis_output_str}")
                        parsed_json_advice = parse_json_from_string(actual_analysis_output_str)
                        if parsed_json_advice and isinstance(parsed_json_advice, dict):
                            try:
                                parsed_advice = RemappingAdvice(**parsed_json_advice)
                                logger.info(f"Successfully parsed RemappingAdvice from analysis crew for {pkg_id}.")
                            except Exception as p_err:
                                logger.error(f"Pydantic validation error for RemappingAdvice: {p_err}. JSON was: {parsed_json_advice}")
                        else:
                            logger.error(f"Analysis crew output was not a parseable JSON object: {actual_analysis_output_str}")
                    else:
                         logger.error(f"Could not extract a string output from analysis_crew.kickoff() result.")

                # --- Update Package State ---
                final_report_path = os.path.join(self.analysis_dir, f"package_{pkg_id}_item_processing_results.json")
                try:
                    os.makedirs(self.analysis_dir, exist_ok=True)
                    with open(final_report_path, 'w', encoding='utf-8') as f:
                        json.dump([r.model_dump() for r in all_item_results_for_package], f, indent=2)
                    logger.info(f"Saved item processing report: {final_report_path}")
                    current_artifacts = pkg_info.get('artifacts', {})
                    current_artifacts['item_processing_report'] = os.path.basename(final_report_path)
                except IOError as e: logger.error(f"Failed to save item report {final_report_path}: {e}")

                artifacts_for_final_update = pkg_info.get('artifacts', {}).copy() 
                artifacts_for_final_update['item_processing_report'] = os.path.basename(final_report_path)

                if not package_overall_success:
                    overall_success_this_run = False
                    err_reason = "One or more items failed."
                    if parsed_advice and parsed_advice.recommend_remapping:
                        err_reason = f"Remapping: {parsed_advice.reason or 'Failures suggest mapping issues.'}"
                        if pkg_info.get('remapping_attempts', 0) >= self.config.get("MAX_REMAPPING_ATTEMPTS", 1):
                            self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remaps. Last: {err_reason}", artifacts=artifacts_for_final_update)
                        else:
                            self.state_manager.update_package_state(pkg_id, status='needs_remapping', error=err_reason, increment_remap_attempt=True, artifacts=artifacts_for_final_update)
                    else:
                        self.state_manager.update_package_state(pkg_id, status='failed_processing', error=err_reason, artifacts=artifacts_for_final_update)
                else: 
                    if parsed_advice and parsed_advice.recommend_remapping: 
                        overall_success_this_run = False 
                        err_reason = f"Remapping: {parsed_advice.reason or 'Advisor recommends remapping despite item success.'}"
                        if pkg_info.get('remapping_attempts', 0) >= self.config.get("MAX_REMAPPING_ATTEMPTS", 1):
                            self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remaps. Last: {err_reason}", artifacts=artifacts_for_final_update)
                        else:
                            self.state_manager.update_package_state(pkg_id, status='needs_remapping', error=err_reason, increment_remap_attempt=True, artifacts=artifacts_for_final_update)
                    else: 
                        logger.info(f"Package {pkg_id} successfully processed.")
                        self.state_manager.update_package_state(pkg_id, status='processed', artifacts=artifacts_for_final_update)
            
            except Exception as e:
                logger.error(f"Critical error processing package {pkg_id}: {e}", exc_info=True)
                artifacts_on_error = pkg_info.get('artifacts', {}).copy() if 'pkg_info' in locals() else {}
                if 'final_report_path' in locals() and os.path.exists(final_report_path): 
                     artifacts_on_error['item_processing_report'] = os.path.basename(final_report_path)

                self.state_manager.update_package_state(pkg_id, status='failed_processing', error=f"Executor error: {e}", artifacts=artifacts_on_error)
                overall_success_this_run = False
        
        if processed_a_package:
            all_potential_targets_done_or_failed_final = True
            final_packages_state_final = self.state_manager.get_all_packages()
            for pkg_id_final_check in potential_target_package_ids:
                status_final = final_packages_state_final.get(pkg_id_final_check, {}).get('status')
                if not (status_final == processed_status or status_final == 'needs_remapping' or any(status_final.startswith(fp) for fp in failed_status_prefixes)):
                    all_potential_targets_done_or_failed_final = False
                    break
            if all_potential_targets_done_or_failed_final:
                if not any(p.get('status') == 'needs_remapping' for p_id_chk, p in final_packages_state_final.items() if p_id_chk in potential_target_package_ids):
                    if not (self.state_manager.get_state().get('workflow_status','').startswith('failed_')):
                        self.state_manager.update_workflow_status('step5_complete')
                else:
                    logger.info("Step 5 finished, but some packages need remapping.")
            elif not overall_success_this_run : 
                 if not (self.state_manager.get_state().get('workflow_status','').startswith('failed_')):
                    self.state_manager.update_workflow_status('failed_step5', "One or more packages failed processing.")
        
        logger.info(f"--- Finished Step 5 Execution Run (Overall Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run

    def _build_package_context(self, pkg_id: str, pkg_info: Dict[str, Any], mapping_data: Dict[str, Any], general_instructions: Optional[str]) -> str:
        context_parts = []
        if general_instructions:
            context_parts.append(f"**General Instructions:**\n{general_instructions}")

        pkg_desc = pkg_info.get('description', 'N/A')
        context_parts.append(f"**Current Package ({pkg_id}) Description:**\n{pkg_desc}")
        
        source_files = self.context_manager.get_source_file_list(pkg_id)
        if source_files:
            source_list_str = "\n".join([f"- `{f['file_path']}`: {f['role']}" for f in source_files])
            context_parts.append(f"**Source Files & Roles for {pkg_id}:**\n{source_list_str}")
        
        target_files = self.context_manager.get_target_file_list(pkg_id)
        if target_files:
             target_list_str = "\n".join([f"- `{f['path']}` (Exists: {f['exists']}): {f['purpose']}" for f in target_files])
             context_parts.append(f"**Target Files & Status for {pkg_id}:**\n{target_list_str}")
        
        mapping_strategy = mapping_data.get("mapping_strategy")
        if mapping_strategy:
             context_parts.append(f"**Overall Mapping Strategy for {pkg_id}:**\n{mapping_strategy}")
        
        temp_context_so_far = "\n\n".join(context_parts)
        tokens_so_far = count_tokens(temp_context_so_far)
        max_source_tokens = int((config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER) * 0.6) - tokens_so_far
        
        if max_source_tokens > 0:
            source_code = self.context_manager.get_work_package_source_code_content(pkg_id, max_tokens=max_source_tokens)
            if source_code:
                context_parts.append(f"**Source Code for {pkg_id} (may be truncated):**\n{source_code}")
            else:
                logger.warning(f"Could not retrieve source code for {pkg_id} within token limits for Step 5 base context.")
        else:
            logger.warning(f"Not enough token budget for source code of {pkg_id} in Step 5 base context.")
        
        full_context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Assembled base package context for {pkg_id} ({count_tokens(full_context)} tokens).")
        return full_context

    def _build_item_specific_context(self, base_package_context: str, task_item_details: Dict[str, Any], godot_project_path: str) -> str:
        item_context_parts = [base_package_context]
        item_context_parts.append(f"\n\n**Reference Godot Project Path (for validator):** `{godot_project_path}`")

        target_godot_file_res_path = task_item_details.get('output_godot_file')
        if target_godot_file_res_path:
            absolute_path_str = ""
            if target_godot_file_res_path.startswith("res://"):
                relative_path = target_godot_file_res_path[len("res://"):]
                absolute_path_str = os.path.join(godot_project_path, relative_path)
            else: 
                logger.info(f"Assuming '{target_godot_file_res_path}' is a relative path from project root for item context.")
                absolute_path_str = os.path.join(godot_project_path, target_godot_file_res_path)
            
            if absolute_path_str:
                target_file_content = read_godot_file_content(absolute_path_str)
                if target_file_content is not None:
                    item_context_parts.append(
                        f"\n\n**Current Target File Content (`{target_godot_file_res_path}`):**\n"
                        f"```gdscript\n{target_file_content}\n```"
                    )
                else: 
                    logger.info(f"Target file '{target_godot_file_res_path}' (resolved to '{absolute_path_str}') not found or empty, assuming new file for item context.")
            else:
                 logger.warning(f"Could not resolve absolute path for target file '{target_godot_file_res_path}' for item context.")
        
        return "\n".join(item_context_parts)
