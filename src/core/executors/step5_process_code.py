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
from ...tools.remapping_logic import RemappingLogic

import src.config as config

from src.agents.step5.code_generator import get_code_generator_agent
from src.agents.step5.output_format_decider import get_output_format_decider_agent
from src.agents.step5.search_block_extractor import get_search_block_extractor_agent
from src.agents.step5.syntax_validator import get_project_validation_agent
from src.agents.step5.code_refiner import get_code_refinement_agent
from src.agents.step5.file_manager import get_file_manager_agent
from src.agents.step5.remapping_advisor import get_remapping_advisor_agent

from src.tools.crewai_tools import (
    FileWriterTool,
    FileReplacerTool,
    GodotProjectValidatorTool,
    RemappingLogicTool,
    FileReaderTool
)

from src.tasks.step5.process_code import (
    create_analyze_package_failures_task,
    RemappingAdvice,
    TaskItemProcessingResult,
)

from src.utils.json_utils import parse_json_from_string
from src.logger_setup import get_logger
import re # Import re for regex operations

logger = get_logger(__name__)

class Step5Executor(StepExecutor):
    """Executes Step 5: Iterative Conversion & Refinement with explicit per-item processing."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config_dict: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]],
                 tools: Dict[Type, Any], # tools from main, not used directly here now
                 remapping_logic: RemappingLogic): # remapping_logic passed if needed by advisor
        super().__init__(state_manager, context_manager, config_dict, llm_configs, tools) # Pass config_dict as 'config' to base
        self.remapping_logic = remapping_logic # Keep if advisor uses it directly, or pass via tool
        # Use the global config module for these paths
        self.target_dir = os.path.abspath(config.GODOT_PROJECT_DIR or "output/godot_project")
        self.analysis_dir = os.path.abspath(config.ANALYSIS_OUTPUT_DIR or "output/analysis")
        logger.info("Step5Executor initialized with explicit item processing.")


    def execute(self, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        logger.info(f"--- Starting Step 5 Execution (Explicit Item Processing): (Packages: {package_ids or 'All Eligible'}, Force={force}) ---")

        # --- Instantiate LLMs, Tools, and Agents ---
        file_writer_tool = FileWriterTool()
        file_replacer_tool = FileReplacerTool()
        project_validator_tool = GodotProjectValidatorTool()
        file_reader_tool = FileReaderTool()
        remapping_advisor_tool = RemappingLogicTool()
        logger.info("Instantiated CrewAI tools for Step 5.")

        # LLMs for each specialized agent
        generator_llm = self._create_llm_instance('GENERATOR_REFINER_MODEL') # Outputs code string
        format_decider_llm = self._create_llm_instance('UTILITY_MODEL') # Outputs 'FULL_FILE' or 'CODE_BLOCK'
        search_extractor_llm = self._create_llm_instance('ANALYZER_MODEL') # Outputs search block string or 'NULL'
        validator_llm = self._create_llm_instance('UTILITY_MODEL') # For project validator agent
        refiner_llm = self._create_llm_instance('GENERATOR_REFINER_MODEL') # Outputs refined code string
        file_manager_llm = self._create_llm_instance('UTILITY_MODEL') # Executes file ops
        remapping_advisor_llm = self._create_llm_instance('ANALYZER_MODEL', response_schema_class=RemappingAdvice) # For final analysis

        required_llms = {
            'Generator': generator_llm, 'FormatDecider': format_decider_llm, 'SearchExtractor': search_extractor_llm,
            'Validator': validator_llm, 'Refiner': refiner_llm, 'File Manager': file_manager_llm,
            'Remapping Advisor': remapping_advisor_llm
        }
        missing_llms = [name for name, llm in required_llms.items() if not llm]
        if missing_llms:
             logger.error(f"Missing critical LLM instances for Step 5: {', '.join(missing_llms)}. Cannot proceed.")
             self.state_manager.update_workflow_status('failed_step5', f"Missing LLM config for: {', '.join(missing_llms)}")
             return False

        # Instantiate all agents
        code_generator_agent = get_code_generator_agent(generator_llm, tools=[file_reader_tool])
        output_format_decider_agent = get_output_format_decider_agent(format_decider_llm)
        search_block_extractor_agent = get_search_block_extractor_agent(search_extractor_llm)
        # project_validator_agent is not directly used in the item loop if tool is called directly
        code_refinement_agent = get_code_refinement_agent(refiner_llm, tools=[file_reader_tool])
        # file_manager_agent is not directly used in the item loop if tools are called directly
        remapping_advisor_agent = get_remapping_advisor_agent(remapping_advisor_llm, tools=[remapping_advisor_tool])
        logger.info("Instantiated LLMs and Agents for Step 5 execution (some agents might be for specific sub-tasks like analysis).")

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
             if matches_specific_request_loop and (is_target_loop or (force and is_step4_done_or_later_loop)):
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
             elif force and is_step4_done_or_later_loop:
                  logger.info(f"Force=True: Adding package '{pkg_id_loop}' (status: {current_status_loop}) to process list.")
                  self.state_manager.update_package_state(pkg_id_loop, target_status, error=None)
                  packages_to_process_this_run.append(pkg_id_loop)
        
        if not packages_to_process_this_run:
            logger.info("No packages require processing in this Step 5 run.")
            return True

        logger.info(f"Eligible packages for this Step 5 run (Force={force}): {packages_to_process_this_run}")
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
                if not mapping_artifact_name and force and pkg_info.get('status','').startswith('failed_'):
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
                
                godot_project_abs_path = config.GODOT_PROJECT_DIR # Use direct config
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
                    
                    item_result = self._process_single_task_item(
                        task_item_details=task_item_details,
                        item_context_str=item_specific_context_str,
                        godot_project_path=godot_project_abs_path,
                        agents={ # Pass agents needed for LLM-driven tasks
                            "code_generator": code_generator_agent,
                            "output_format_decider": output_format_decider_agent,
                            "search_block_extractor": search_block_extractor_agent,
                            "code_refiner": code_refinement_agent,
                        },
                        tools={ # Pass tool instances needed for direct calls
                            "project_validator_tool": project_validator_tool,
                            "file_writer_tool": file_writer_tool,
                            "file_replacer_tool": file_replacer_tool,
                        },
                        llm_instances={ # Pass LLMs needed for ad-hoc crews within the method
                            "generator": generator_llm,
                            "format_decider": format_decider_llm,
                            "search_extractor": search_extractor_llm,
                            "validator": validator_llm,
                            "refiner": refiner_llm,
                            "file_manager": file_manager_llm
                        },
                        general_instructions=instr_context
                    )
                    all_item_results_for_package.append(item_result)
                    if item_result.status == 'failed':
                        package_overall_success = False
                
                # --- Analyze Package Failures ---
                parsed_advice = None
                if all_item_results_for_package:
                    # Convert TaskItemProcessingResult objects to a list of dictionaries
                    item_results_as_dicts = [result.model_dump() for result in all_item_results_for_package]
                    
                    analysis_task_obj = create_analyze_package_failures_task( # Renamed variable for clarity
                        advisor_agent=remapping_advisor_agent,
                        item_processing_results=item_results_as_dicts, # Pass list of dicts
                        instructions=instr_context
                    )
                    analysis_crew = Crew(agents=[remapping_advisor_agent], tasks=[analysis_task_obj], process=Process.sequential, memory=False, verbose=True)
                    logger.info(f"Kicking off Analysis Crew for package {pkg_id}...")
                    analysis_crew_kickoff_result = analysis_crew.kickoff()
                    logger.info(f"Analysis Crew for {pkg_id} finished. Kickoff result type: {type(analysis_crew_kickoff_result)}")

                    # Extract raw string from CrewOutput if necessary
                    actual_analysis_output_str = None
                    if isinstance(analysis_crew_kickoff_result, TaskOutput): # If it's a TaskOutput object
                        actual_analysis_output_str = analysis_crew_kickoff_result.raw_output
                    elif isinstance(analysis_crew_kickoff_result, str): # If it's already a string
                        actual_analysis_output_str = analysis_crew_kickoff_result
                    elif hasattr(analysis_crew_kickoff_result, 'raw') and isinstance(analysis_crew_kickoff_result.raw, str): # For CrewOutput
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
                    # This specific call to update_package_state was only for artifacts,
                    # the main status update happens below.
                    # However, update_package_state always requires a 'status'.
                    # We can defer this artifact update or include a non-changing status.
                    # For now, let's assume the status update below will handle artifacts too.
                    # If not, this needs to be self.state_manager.update_package_state(pkg_id, status=pkg_info.get('status'), artifacts=current_artifacts)
                    # For simplicity, let's ensure the main status update below also handles artifacts.
                    # So, this specific call might be redundant if the logic below is comprehensive.
                    # Let's comment it out for now and ensure the later status update includes artifacts.
                    # self.state_manager.update_package_state(pkg_id, artifacts=current_artifacts) # This line was causing an error
                except IOError as e: logger.error(f"Failed to save item report {final_report_path}: {e}")


                # --- State Update based on package_overall_success and parsed_advice ---
                # This is the main status update section for the package after item processing and analysis.
                # We need to ensure artifacts are included here.
                artifacts_for_final_update = pkg_info.get('artifacts', {}).copy() # Start with existing
                artifacts_for_final_update['item_processing_report'] = os.path.basename(final_report_path) # Ensure it's set

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
                else: # package_overall_success is True
                    if parsed_advice and parsed_advice.recommend_remapping: # Success, but still remapping
                        overall_success_this_run = False # Not a "final" success for the run
                        err_reason = f"Remapping: {parsed_advice.reason or 'Advisor recommends remapping despite item success.'}"
                        if pkg_info.get('remapping_attempts', 0) >= self.config.get("MAX_REMAPPING_ATTEMPTS", 1):
                            self.state_manager.update_package_state(pkg_id, status='failed_remapping', error=f"Max remaps. Last: {err_reason}", artifacts=artifacts_for_final_update)
                        else:
                            self.state_manager.update_package_state(pkg_id, status='needs_remapping', error=err_reason, increment_remap_attempt=True, artifacts=artifacts_for_final_update)
                    else: # True success
                        logger.info(f"Package {pkg_id} successfully processed.")
                        self.state_manager.update_package_state(pkg_id, status='processed', artifacts=artifacts_for_final_update)
            
            except Exception as e:
                logger.error(f"Critical error processing package {pkg_id}: {e}", exc_info=True)
                # Ensure artifacts (like the report) are still updated even on critical error if possible
                artifacts_on_error = pkg_info.get('artifacts', {}).copy() if 'pkg_info' in locals() else {}
                if 'final_report_path' in locals() and os.path.exists(final_report_path): # Check if report was saved
                     artifacts_on_error['item_processing_report'] = os.path.basename(final_report_path)

                self.state_manager.update_package_state(pkg_id, status='failed_processing', error=f"Executor error: {e}", artifacts=artifacts_on_error)
                overall_success_this_run = False
        
        # --- Final Workflow Status Update (similar to previous) ---
        # ... (logic to set step5_complete or failed_step5 based on overall_success_this_run and processed_a_package)
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
            elif not overall_success_this_run : # If not all done AND this run had failures
                 if not (self.state_manager.get_state().get('workflow_status','').startswith('failed_')):
                    self.state_manager.update_workflow_status('failed_step5', "One or more packages failed processing.")
        
        logger.info(f"--- Finished Step 5 Execution Run (Overall Success This Run: {overall_success_this_run}) ---")
        return overall_success_this_run

    def _build_package_context(self, pkg_id: str, pkg_info: Dict[str, Any], mapping_data: Dict[str, Any], general_instructions: Optional[str]) -> str:
        context_parts = []
        if general_instructions: # Prepend general instructions if available
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
        max_source_tokens = int((config.MAX_CONTEXT_TOKENS - config.PROMPT_TOKEN_BUFFER) * 0.6) - tokens_so_far # Adjusted multiplier
        
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
        # Add Godot project path for validator context
        item_context_parts.append(f"\n\n**Reference Godot Project Path (for validator):** `{godot_project_path}`")

        target_godot_file_res_path = task_item_details.get('output_godot_file')
        if target_godot_file_res_path:
            absolute_path_str = ""
            if target_godot_file_res_path.startswith("res://"):
                relative_path = target_godot_file_res_path[len("res://"):]
                absolute_path_str = os.path.join(godot_project_path, relative_path)
            else: # Assume it's already a relative path from project root
                logger.info(f"Assuming '{target_godot_file_res_path}' is a relative path from project root for item context.")
                absolute_path_str = os.path.join(godot_project_path, target_godot_file_res_path)
            
            if absolute_path_str:
                target_file_content = read_godot_file_content(absolute_path_str)
                if target_file_content is not None:
                    item_context_parts.append(
                        f"\n\n**Current Target File Content (`{target_godot_file_res_path}`):**\n"
                        f"```gdscript\n{target_file_content}\n```"
                    )
                else: # File might not exist yet, which is fine for new files
                    logger.info(f"Target file '{target_godot_file_res_path}' (resolved to '{absolute_path_str}') not found or empty, assuming new file for item context.")
            else:
                 logger.warning(f"Could not resolve absolute path for target file '{target_godot_file_res_path}' for item context.")
        
        return "\n".join(item_context_parts)

    def _process_single_task_item(self,
                                  task_item_details: Dict[str, Any],
                                  item_context_str: str,
                                  godot_project_path: str,
                                  agents: Dict[str, Agent], # Agents needed for generation/decision/refinement
                                  tools: Dict[str, Any], # Tool instances needed for direct calls (validator, writer, replacer)
                                  llm_instances: Dict[str, Any], # LLM instances for ad-hoc crews
                                  general_instructions: Optional[str] = None
                                 ) -> TaskItemProcessingResult:
        task_id = task_item_details.get('task_id', task_item_details.get('task_title', 'unknown_task'))
        target_godot_file = task_item_details.get('output_godot_file', 'unknown_file')
        target_element = task_item_details.get('target_element')
        
        current_status = "failed" # Default to failed
        error_log = []

        # --- Step 1: Generate Code ---
        logger.info(f"[{task_id}] Step 1: Generate Code")
        generated_code_string: Optional[str] = None
        code_gen_agent = agents['code_generator']
        code_gen_task_desc = (
            f"{general_instructions or ''}\n\n"
            f"**Task: Generate Godot Code String for item '{task_id}'**\n"
            f"Target File Hint: `{target_godot_file}`\n"
            f"Task Item Details: ```json\n{json.dumps(task_item_details, indent=2)}\n```\n"
            f"Context (C++ source, existing Godot code if any, project path etc.):\n"
            f"--- START OF PROVIDED CONTEXT ---\n{item_context_str}\n--- END OF PROVIDED CONTEXT ---\n"
            f"Generate the `{config.TARGET_LANGUAGE}` code string based on the task details and context. "
            f"Output ONLY the raw code string."
        )
        code_gen_task = Task(description=code_gen_task_desc, expected_output=f"The raw {config.TARGET_LANGUAGE} code string.", agent=code_gen_agent)
        code_gen_crew = Crew(agents=[code_gen_agent], tasks=[code_gen_task], process=Process.sequential, memory=False, llm=llm_instances['generator'])
        try:
            code_gen_kickoff_result = code_gen_crew.kickoff()
            logger.debug(f"[{task_id}] S1: CodeGen crew kickoff result type: {type(code_gen_kickoff_result)}")

            actual_code_gen_output_str = None
            if isinstance(code_gen_kickoff_result, TaskOutput):
                actual_code_gen_output_str = code_gen_kickoff_result.raw_output
            elif isinstance(code_gen_kickoff_result, str):
                actual_code_gen_output_str = code_gen_kickoff_result
            elif hasattr(code_gen_kickoff_result, 'raw') and isinstance(code_gen_kickoff_result.raw, str): # For CrewOutput
                actual_code_gen_output_str = code_gen_kickoff_result.raw
            else:
                error_log.append(f"S1: CodeGen crew returned unexpected kickoff result type: {type(code_gen_kickoff_result)}. Full result: {code_gen_kickoff_result}")
            
            if actual_code_gen_output_str is not None: # Check if we got a string
                extracted_code = self._extract_fenced_code(actual_code_gen_output_str)
                if extracted_code is not None: # Helper found and extracted from fences
                    generated_code_string = extracted_code
                    logger.info(f"[{task_id}] Step 1 Succeeded. Code extracted from fences (length: {len(generated_code_string)}).")
                else:
                    # No valid fences found, use the whole string, stripped.
                    generated_code_string = actual_code_gen_output_str.strip()
                    if generated_code_string:
                        logger.info(f"[{task_id}] Step 1 Succeeded. Used raw output (no fences detected) (length: {len(generated_code_string)}).")
                    else:
                        # This means the original string was empty or only whitespace
                        error_log.append(f"S1: CodeGen returned an empty string after stripping and no valid fences. Original: '{actual_code_gen_output_str[:100]}...'")
            
        except Exception as e: error_log.append(f"S1 Exception: {e}"); logger.error(f"[{task_id}] S1 Exception: {e}", exc_info=True)

        if not generated_code_string:
            return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log) or "Code generation failed.")

        # --- Step 1.5: Decide Output Format ---
        logger.info(f"[{task_id}] Step 1.5: Decide Output Format")
        output_format: Optional[Literal['FULL_FILE', 'CODE_BLOCK']] = None
        format_decider_agent = agents['output_format_decider']
        format_desc = (
             f"{general_instructions or ''}\n\n"
             f"**Task: Decide File Output Format for item '{task_id}'**\n"
             f"Target File Hint: `{target_godot_file}`\n"
             f"Task Item Details: ```json\n{json.dumps(task_item_details, indent=2)}\n```\n"
             f"Generated Code Snippet (for context):\n```gdscript\n{generated_code_string[:500]}...\n```\n" # Show snippet
             f"Context (including existing target file content if available):\n"
             f"--- START OF PROVIDED CONTEXT ---\n{item_context_str}\n--- END OF PROVIDED CONTEXT ---\n"
             f"Analyze the task, generated code, and existing file content. Decide if the operation should be 'FULL_FILE' or 'CODE_BLOCK'. "
             f"Output ONLY the string 'FULL_FILE' or 'CODE_BLOCK'."
        )
        format_task = Task(description=format_desc, expected_output="The string 'FULL_FILE' or 'CODE_BLOCK'.", agent=format_decider_agent)
        format_crew = Crew(agents=[format_decider_agent], tasks=[format_task], process=Process.sequential, memory=False, llm=llm_instances['format_decider'])
        try:
            format_crew_kickoff_result = format_crew.kickoff()
            logger.debug(f"[{task_id}] S1.5: FormatDecider kickoff result type: {type(format_crew_kickoff_result)}")

            actual_format_output_str = None
            if isinstance(format_crew_kickoff_result, TaskOutput):
                actual_format_output_str = format_crew_kickoff_result.raw_output
            elif isinstance(format_crew_kickoff_result, str):
                actual_format_output_str = format_crew_kickoff_result
            elif hasattr(format_crew_kickoff_result, 'raw') and isinstance(format_crew_kickoff_result.raw, str): # For CrewOutput
                actual_format_output_str = format_crew_kickoff_result.raw
            else:
                error_log.append(f"S1.5: FormatDecider crew returned unexpected kickoff result type: {type(format_crew_kickoff_result)}. Full result: {format_crew_kickoff_result}")

            if actual_format_output_str is not None:
                logger.debug(f"[{task_id}] S1.5: FormatDecider raw string output: '{actual_format_output_str}'")
                cleaned_format_result = actual_format_output_str.strip().upper()
                if "FULL_FILE" in cleaned_format_result:
                    output_format = 'FULL_FILE'
                elif "CODE_BLOCK" in cleaned_format_result:
                    output_format = 'CODE_BLOCK'
                
                if output_format:
                    logger.info(f"[{task_id}] Step 1.5 Succeeded. Determined format: {output_format} from raw string: '{actual_format_output_str}'")
                else:
                    error_log.append(f"S1.5: FormatDecider output did not contain 'FULL_FILE' or 'CODE_BLOCK'. Raw string: '{actual_format_output_str}'")
            # If actual_format_output_str is None, an error was already logged by the type check.

        except Exception as e: error_log.append(f"S1.5 Exception: {e}"); logger.error(f"[{task_id}] S1.5 Exception: {e}", exc_info=True)

        if not output_format:
             return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log) or "Failed to determine output format.")

        # --- Step 1.7: Extract Search Block (Conditional) ---
        search_block: Optional[str] = None
        if output_format == 'CODE_BLOCK':
            logger.info(f"[{task_id}] Step 1.7: Extract Search Block (since format is CODE_BLOCK)")
            search_extractor_agent = agents['search_block_extractor']
            search_desc = (
                f"{general_instructions or ''}\n\n"
                f"**Task: Extract Search Block for item '{task_id}'**\n"
                f"Target File Hint: `{target_godot_file}`. The operation mode is 'CODE_BLOCK'.\n"
                f"Task Item Details: ```json\n{json.dumps(task_item_details, indent=2)}\n```\n"
                f"Generated Code Snippet (for context of what will replace the block):\n```gdscript\n{generated_code_string}\n```\n"
                f"Context (MUST contain 'Current Target File Content' of `{target_godot_file}`):\n"
                f"--- START OF PROVIDED CONTEXT ---\n{item_context_str}\n--- END OF PROVIDED CONTEXT ---\n"
                f"Identify and extract the **exact, original, unfenced code block** from 'Current Target File Content' that the generated code should replace. "
                f"If no specific block needs replacement or cannot be confidently identified, your entire output **MUST BE the exact literal string 'NULL'**. "
                f"Otherwise, your entire output **MUST BE ONLY the extracted code block string itself**, without any surrounding text, explanations, or markdown fences."
            )
            search_task = Task(description=search_desc, expected_output="The exact search block string or the literal string 'NULL'.", agent=search_extractor_agent)
            search_crew = Crew(agents=[search_extractor_agent], tasks=[search_task], process=Process.sequential, memory=False, llm=llm_instances['search_extractor'])
            
            raw_search_output_str: Optional[str] = None
            try:
                search_crew_kickoff_result = search_crew.kickoff()
                logger.debug(f"[{task_id}] S1.7: SearchExtractor kickoff result type: {type(search_crew_kickoff_result)}")
                
                if isinstance(search_crew_kickoff_result, TaskOutput):
                    raw_search_output_str = search_crew_kickoff_result.raw_output
                elif isinstance(search_crew_kickoff_result, str):
                    raw_search_output_str = search_crew_kickoff_result
                elif hasattr(search_crew_kickoff_result, 'raw') and isinstance(search_crew_kickoff_result.raw, str): # For CrewOutput
                    raw_search_output_str = search_crew_kickoff_result.raw
                else:
                    error_log.append(f"S1.7: SearchExtractor crew returned unexpected kickoff result type: {type(search_crew_kickoff_result)}. Full result: {search_crew_kickoff_result}")

                if raw_search_output_str is not None:
                    logger.debug(f"[{task_id}] S1.7: SearchExtractor raw string output: '{raw_search_output_str[:200]}...'")
                    cleaned_search_output = raw_search_output_str.strip()
                    if cleaned_search_output == 'NULL':
                        search_block = None
                        logger.info(f"[{task_id}] Step 1.7: SearchExtractor indicated 'NULL' - no search block.")
                    elif cleaned_search_output:
                        # Attempt to extract from fences as a safety net, though agent is instructed not to use them for the block itself.
                        fenced_content = self._extract_fenced_code(cleaned_search_output)
                        if fenced_content is not None and fenced_content.strip():
                            search_block = fenced_content
                            logger.info(f"[{task_id}] Step 1.7: Extracted search block from fences (length: {len(search_block)}).")
                        else: # No valid fences, or agent returned thoughts then 'NULL', or just the block.
                            search_block = cleaned_search_output # Use the stripped output directly
                            logger.info(f"[{task_id}] Step 1.7: Used raw stripped output as search block (length: {len(search_block)}).")
                        
                        if not search_block.strip(): # If it ended up empty after all processing
                            search_block = None
                            logger.warning(f"[{task_id}] Step 1.7: Search block became empty after processing. Original raw: '{raw_search_output_str[:200]}...'")
                    else: # Raw output was empty or only whitespace
                        search_block = None
                        error_log.append(f"S1.7: SearchExtractor returned an empty string. Original raw: '{raw_search_output_str[:200]}...'")
                # If raw_search_output_str is None, error already logged.
            except Exception as e:
                error_log.append(f"S1.7 Exception: {e}")
                logger.error(f"[{task_id}] S1.7 Exception: {e}", exc_info=True)
        
        # Critical Mismatch Check
        if output_format == 'CODE_BLOCK' and (search_block is None or not search_block.strip()):
            error_message = f"S1.7 Critical Mismatch: Output format is 'CODE_BLOCK', but no valid search_block was provided or extracted by SearchBlockExtractorAgent. Raw output from extractor: '{raw_search_output_str if 'raw_search_output_str' in locals() else 'N/A'}'"
            logger.error(f"[{task_id}] {error_message}")
            error_log.append(error_message)
            return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log))

        # --- Step 2: Initial File Operation (Direct Tool Call) ---
        logger.info(f"[{task_id}] Step 2: Initial File Operation (Format: {output_format})")
        file_op_success = False
        # Get tool instances from the 'tools' dict passed to this method
        file_writer_tool_instance = tools['file_writer_tool']
        file_replacer_tool_instance = tools['file_replacer_tool']

        try:
            op_result_str = ""
            if output_format == 'FULL_FILE':
                logger.debug(f"[{task_id}] S2: Calling FileWriterTool.")
                op_result_str = file_writer_tool_instance._run(file_path=target_godot_file, content=generated_code_string)
            elif output_format == 'CODE_BLOCK':
                if search_block is None: # Should ideally be caught by SearchBlockExtractor if it returns 'NULL' appropriately
                    error_log.append("S2 Error: Output format is CODE_BLOCK, but search_block is missing/null.")
                else:
                    logger.debug(f"[{task_id}] S2: Calling FileReplacerTool.")
                    diff_string = f"<<<<<<< SEARCH\n{search_block}\n=======\n{generated_code_string}\n>>>>>>> REPLACE"
                    op_result_str = file_replacer_tool_instance._run(file_path=target_godot_file, diff=diff_string)
            else: # Should not happen if format validation worked
                 error_log.append(f"S2 Error: Invalid output_format '{output_format}' received.")

            logger.info(f"[{task_id}] S2 Direct File Op Result: {op_result_str}")
            if op_result_str and "success" in op_result_str.lower():
                file_op_success = True
            elif not error_log: # Only add tool message if no prior error logged for this step
                 error_log.append(f"S2: File op failed. Tool Msg: {op_result_str}")

        except Exception as e:
            error_log.append(f"S2 Exception during direct tool call: {e}")
            logger.error(f"[{task_id}] S2 Exception (direct call): {e}", exc_info=True)

        if not file_op_success:
            return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log))

        # --- Step 3: Validate Project (Initial) ---
        # (Validation logic remains the same, using direct tool call)
        logger.info(f"[{task_id}] Step 3: Validate Project (Initial)")
        validation_passed = False
        validation_errors_str: Optional[str] = None
        project_validator_tool_instance = tools['project_validator_tool'] # The CrewAI Tool instance

        try:
            # The tool's _run method expects godot_project_path and target_file_path (res://)
            validation_result_str = project_validator_tool_instance._run(
                godot_project_path=godot_project_path, # Absolute path to project
                target_file_path=target_godot_file     # res:// path to the modified file
            )
            logger.info(f"[{task_id}] S3 Validation Result: {validation_result_str}")
            if "Project validation successful" in validation_result_str: # Check for specific success message
                validation_passed = True
            else:
                validation_errors_str = validation_result_str # Contains "Errors related to..." or full errors
                error_log.append(f"S3: Initial validation failed. Details: {validation_errors_str}")
        except Exception as e: error_log.append(f"S3 Exception: {e}"); logger.error(f"[{task_id}] S3 Exception: {e}", exc_info=True)
        
        if validation_passed:
            logger.info(f"[{task_id}] S3 Succeeded. Initial validation passed.")
            current_status = "completed"
            return TaskItemProcessingResult(task_id=task_id, status=current_status, target_godot_file=target_godot_file, target_element=target_element, error_message=None)

        if not validation_errors_str: # Validation failed but no errors captured
             error_log.append("S3: Validation failed, no specific errors captured by tool.")
             return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log))

        # --- Step 4: Refine Code ---
        logger.info(f"[{task_id}] Step 4: Refine Code")
        refined_code: Optional[str] = None
        code_refiner_agent = agents['code_refiner']
        
        refinement_desc = (
            f"{general_instructions or ''}\n\n"
            f"Task: Refine Code for '{task_id}'. Target: `{target_godot_file}`.\n"
            f"Validation Errors:\n```\n{validation_errors_str}\n```\n"
            f"Context:\n--- START ---\n{item_context_str}\n--- END ---\n"
            f"Read current content of `{target_godot_file}` using 'File Reader' tool. "
            f"Produce corrected full code string. If unable, return empty or error string."
        )
        refinement_task = Task(description=refinement_desc, expected_output="Corrected full GDScript string or error.", agent=code_refiner_agent)
        refinement_crew = Crew(agents=[code_refiner_agent], tasks=[refinement_task], process=Process.sequential, memory=False, llm=llm_instances['refiner'])

        try:
            refinement_kickoff_result = refinement_crew.kickoff()
            logger.debug(f"[{task_id}] S4: Refinement crew kickoff result type: {type(refinement_kickoff_result)}")

            actual_refined_code_str = None
            if isinstance(refinement_kickoff_result, TaskOutput):
                actual_refined_code_str = refinement_kickoff_result.raw_output
            elif isinstance(refinement_kickoff_result, str):
                actual_refined_code_str = refinement_kickoff_result
            elif hasattr(refinement_kickoff_result, 'raw') and isinstance(refinement_kickoff_result.raw, str): # For CrewOutput
                actual_refined_code_str = refinement_kickoff_result.raw
            else:
                error_log.append(f"S4: Refinement crew returned unexpected kickoff result type: {type(refinement_kickoff_result)}. Full result: {refinement_kickoff_result}")

            if actual_refined_code_str is not None:
                logger.debug(f"[{task_id}] S4: Refinement raw string output: '{actual_refined_code_str[:200]}...'")
                # Check if the refined code is not an error message and has substantial content
                if not actual_refined_code_str.strip().lower().startswith("error:") and len(actual_refined_code_str.strip()) > 10:
                    refined_code = actual_refined_code_str # Keep original stripping for this check, but use the full string
                    logger.info(f"[{task_id}] S4 Succeeded. Code refined (length: {len(refined_code)}).")
                else:
                    error_log.append(f"S4: Refinement failed, returned an error, or produced too short code. Raw: '{actual_refined_code_str[:200]}...'")
            # If actual_refined_code_str is None, an error was already logged by the type check.

        except Exception as e: error_log.append(f"S4 Exception: {e}"); logger.error(f"[{task_id}] S4 Exception: {e}", exc_info=True)

        if not refined_code:
            return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log))

        # --- Step 5: Re-Apply Refined Code (Direct Tool Call) ---
        logger.info(f"[{task_id}] Step 5: Re-Apply Refined Code")
        reapply_success = False
        file_writer_tool_instance = tools['file_writer_tool'] # Get from passed tools dict

        try:
            logger.debug(f"[{task_id}] S5: Calling FileWriterTool with refined code.")
            reapply_str = file_writer_tool_instance._run(file_path=target_godot_file, content=refined_code)
            logger.info(f"[{task_id}] S5 Direct Re-apply Result: {reapply_str}")
            if reapply_str and "success" in reapply_str.lower():
                reapply_success = True
            else:
                error_log.append(f"S5: Re-applying refined code failed. Tool Msg: {reapply_str}")
        except Exception as e:
            error_log.append(f"S5 Exception during direct tool call: {e}")
            logger.error(f"[{task_id}] S5 Exception (direct call): {e}", exc_info=True)

        if not reapply_success:
            return TaskItemProcessingResult(task_id=task_id, status="failed", target_godot_file=target_godot_file, target_element=target_element, error_message="; ".join(error_log))

        # --- Step 6: Re-Validate Project ---
        # (Validation logic remains the same, using direct tool call)
        logger.info(f"[{task_id}] Step 6: Re-Validate Project")
        re_validation_passed = False
        try:
            re_val_str = project_validator_tool_instance._run(godot_project_path=godot_project_path, target_file_path=target_godot_file)
            logger.info(f"[{task_id}] S6 Re-validation Result: {re_val_str}")
            if "Project validation successful" in re_val_str:
                re_validation_passed = True
                current_status = "completed"
            else: error_log.append(f"S6: Re-validation failed. Details: {re_val_str}")
        except Exception as e: error_log.append(f"S6 Exception: {e}"); logger.error(f"[{task_id}] S6 Exception: {e}", exc_info=True)
        
        # --- Step 7: Consolidate Result ---
        logger.info(f"[{task_id}] Step 7: Consolidate. Final status: {current_status}")
        return TaskItemProcessingResult(
            task_id=task_id, status=current_status, target_godot_file=target_godot_file, target_element=target_element,
            error_message="; ".join(error_log) if current_status == "failed" and error_log else None
        )

    def _log_step5_task_completion(self, task_output: TaskOutput):
        try:
            agent_role = "Unknown Agent"
            if hasattr(task_output, 'agent') and task_output.agent:
                if isinstance(task_output.agent, Agent): agent_role = task_output.agent.role
                else: agent_role = f"Agent ({type(task_output.agent).__name__})"
            task_desc_snippet = (task_output.task.description[:100] + "..." if hasattr(task_output, 'task') and task_output.task else 
                                (task_output.description[:100] + "..." if hasattr(task_output, 'description') else "Unknown Task"))
            output_snippet = (str(task_output.raw_output)[:150].replace('\n', ' ') + "..." if hasattr(task_output, 'raw_output') and task_output.raw_output is not None else
                             (str(task_output.output)[:150].replace('\n', ' ') + "..." if hasattr(task_output, 'output') and task_output.output is not None else "No output"))
            logger.info(f"[Step 5 AdHoc Crew Callback] Task Update: Agent: {agent_role}, Task: {task_desc_snippet}, Output: {output_snippet}")
        except Exception as e:
            logger.error(f"[Step 5 AdHoc Crew Callback] Error: {e}", exc_info=True)
            try: logger.debug(f"Raw task_output: {task_output}")
            except: pass

    def _extract_fenced_code(self, text: str) -> Optional[str]:
        """
        Extracts code content from the first well-formed fenced code block.
        Discards any text outside the fences if a block is found.
        """
        # Regex to find content within triple backticks, optionally with a language tag.
        # Captures the content between the fences.
        # - ```(?:[a-zA-Z0-9_]+)? : Matches the opening ``` optionally followed by a language tag.
        # - \s*\n? : Matches optional whitespace then an optional newline.
        # - (.*?) : Non-greedily captures the content (group 1).
        # - \n?\s*``` : Matches an optional newline, optional whitespace, then the closing ```.
        # re.DOTALL allows '.' to match newline characters.
        match = re.search(r"```(?:[a-zA-Z0-9_]+)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            # Return the captured group, stripped of leading/trailing whitespace from within the block
            return match.group(1).strip()
        
        # Fallback for simple one-line fences like ```code``` (less common for blocks but possible)
        # This is less specific, so it's a fallback.
        match_simple = re.search(r"```(.*?)```", text, re.DOTALL)
        if match_simple:
            return match_simple.group(1).strip()
            
        return None # No valid fenced block found
