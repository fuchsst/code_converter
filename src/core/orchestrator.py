# src/core/orchestrator.py
import os
import json
from typing import Any, Dict, List, Optional, Type
from src.logger_setup import get_logger
import src.config as config
from .state_manager import StateManager
from .context_manager import ContextManager
from .remapping_logic import RemappingLogic
from .tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator
from .step_executor import StepExecutor
# Import concrete executors (adjust path if they are in a subfolder like 'executors')
from .executors.step1_analyzer import Step1Executor
from .executors.step2_package_identifier import Step2Executor
from .executors.step3_structure_definer import Step3Executor
from .executors.step4_mapping_definer import Step4Executor
from .executors.step5_process_code import Step5Executor
# Import concrete tool wrappers
from src.tools.framework_tools_wrapper import (
    CrewAIFileWriter, CrewAIFileReader, CustomFileReplacer, GodotSyntaxValidator
)
import litellm # Import litellm

logger = get_logger(__name__)

class Orchestrator:
    """
    Sets up and coordinates the C++ to Godot conversion workflow components.
    Acts as the main entry point and dependency injector.
    """
    def __init__(self,
                 cpp_project_dir: Optional[str] = None,
                 godot_project_dir: Optional[str] = None,
                 analysis_dir: Optional[str] = None,
                 target_language: Optional[str] = None):
        """
        Initializes the Orchestrator and all its components.

        Args:
            cpp_project_dir (str, optional): Path to the C++ project source. Defaults to config.
            godot_project_dir (str, optional): Path to the existing/target Godot project. Defaults to config.
            analysis_dir (str, optional): Directory for analysis files. Defaults to config.
            target_language (str, optional): Target language (e.g., "GDScript"). Defaults to config.
        """
        # --- Configuration Setup ---
        self.config_dict = self._load_config(
            cpp_project_dir, godot_project_dir, analysis_dir, target_language
        )
        self.analysis_dir = self.config_dict["ANALYSIS_OUTPUT_DIR"]
        self.cpp_project_dir = self.config_dict["CPP_PROJECT_DIR"]
        self.include_graph_path = os.path.join(self.analysis_dir, "dependencies.json")

        logger.info("Initializing Orchestrator components...")

        # --- Component Initialization ---
        # State Manager
        self.state_manager = StateManager(analysis_dir=self.analysis_dir)

        # Context Manager (needs graph path, source dir, analysis dir, and state manager)
        self.context_manager = ContextManager(
            include_graph_path=self.include_graph_path,
            cpp_source_dir=self.cpp_project_dir,
            analysis_output_dir=self.analysis_dir,
            state_manager=self.state_manager # Pass state manager instance
        )
        # Ensure ContextManager loads the include graph if Step 1 might have just run
        # self.context_manager.include_graph = self.state_manager.get_state().get("include_graph_data", {}) # This seems redundant if ContextManager loads it
        # Let's rely on ContextManager's __init__ to load the graph.

        # LLM Configurations (for direct litellm use)
        self.llm_configs = self._initialize_llm_configs()

        # Tool Wrappers
        self.tools = self._initialize_tools()

        # Remapping Logic
        self.remapping_logic = RemappingLogic()

        # Step Executors (inject dependencies)
        self.executors = self._initialize_executors()

        logger.info("Orchestrator and components initialized.")
        logger.info(f"  Analysis Dir: {self.analysis_dir}")
        logger.info(f"  State File: {self.state_manager.state_file_path}")
        logger.info(f"  LLM Configs Initialized: {list(self.llm_configs.keys())}")
        logger.info(f"  Tools Initialized: {[t.__name__ for t in self.tools.keys()]}")


    def _load_config(self, cpp_project_dir, godot_project_dir, analysis_dir, target_language) -> Dict[str, Any]:
        """Loads configuration from defaults and overrides."""
        # Create a dictionary representation of the config module for easy access
        # In a real app, this might load from a file or dedicated config object
        cfg = {key: getattr(config, key) for key in dir(config) if not key.startswith('_')}

        # Apply overrides passed to constructor
        if cpp_project_dir: cfg["CPP_PROJECT_DIR"] = cpp_project_dir
        if godot_project_dir: cfg["GODOT_PROJECT_DIR"] = godot_project_dir
        if analysis_dir: cfg["ANALYSIS_OUTPUT_DIR"] = analysis_dir
        if target_language: cfg["TARGET_LANGUAGE"] = target_language

        # Resolve absolute paths
        cfg["CPP_PROJECT_DIR"] = os.path.abspath(cfg["CPP_PROJECT_DIR"])
        cfg["GODOT_PROJECT_DIR"] = os.path.abspath(cfg["GODOT_PROJECT_DIR"])
        cfg["ANALYSIS_OUTPUT_DIR"] = os.path.abspath(cfg["ANALYSIS_OUTPUT_DIR"])

        logger.debug("Configuration loaded and resolved.")
        return cfg

    def _initialize_llm_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initializes LLM configuration dictionaries based on grouped config variables."""
        llm_configs_map = {}
        # Define the config keys for the models we need to configure
        model_config_keys = [
            "MANAGER_MODEL",
            "ANALYZER_MODEL",
            "DESIGNER_PLANNER_MODEL",
            "GENERATOR_REFINER_MODEL",
            "UTILITY_MODEL",
            "DEFAULT_AGENT_MODEL" # Include default as a potential fallback
        ]

        # Common LLM settings from config (remains the same)
        common_params = {
            "temperature": self.config_dict.get("DEFAULT_TEMPERATURE", 0.9),
            "top_p": self.config_dict.get("DEFAULT_TOP_P", 0.95),
            #"top_k": self.config_dict.get("DEFAULT_TOP_K", 40),
            # Add other common params if needed by litellm.completion
        }

        for config_key in model_config_keys:
            model_name = self.config_dict.get(config_key)
            if model_name:
                # Prepare the config dictionary using the config key itself
                config_for_key = {
                    "model": model_name,
                    **common_params # Add common params
                }

                # Add API key if needed and available (LiteLLM primarily uses env vars)
                # Example for Gemini:
                if model_name.startswith(("gemini/", "google/")):
                    api_key = self.config_dict.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if api_key:
                        config_for_key["api_key"] = api_key
                        logger.debug(f"Adding GEMINI_API_KEY to config for key '{config_key}'")
                    else:
                        logger.warning(f"GEMINI_API_KEY not found in config or env for key '{config_key}' using model '{model_name}'. LiteLLM might fail.")
                # Add elif blocks for other providers (e.g., OpenAI, Anthropic) if explicit key passing is desired

                llm_configs_map[config_key] = config_for_key
                logger.info(f"Prepared LLM config for key '{config_key}': model={model_name}")
            else:
                # Log if a config key doesn't have a model assigned
                logger.warning(f"No model name found or configured for config key '{config_key}'. Skipping config preparation.")

        if not llm_configs_map:
             logger.error("No LLM configurations were successfully prepared!")
             # Consider raising an error if LLMs are essential

        return llm_configs_map

    def _initialize_tools(self) -> Dict[Type, Any]:
        """Initializes concrete tool wrapper instances."""
        tools = {
            IFileWriter: CrewAIFileWriter(),
            IFileReplacer: CustomFileReplacer(),
            IFileReader: CrewAIFileReader(),
            ISyntaxValidator: GodotSyntaxValidator()
        }
        logger.debug("Tool wrappers initialized.")
        return tools

    def _initialize_executors(self) -> Dict[str, StepExecutor]:
        """Initializes all step executors, injecting dependencies."""
        # Step 1 doesn't use LLMs or Tools, but the base class expects the args. Pass empty dicts.
        executors = {
            "step1": Step1Executor(self.state_manager, self.context_manager, self.config_dict, llm_configs={}, tools={}),
            "step2": Step2Executor(self.state_manager, self.context_manager, self.config_dict, self.llm_configs, self.tools),
            "step3": Step3Executor(self.state_manager, self.context_manager, self.config_dict, self.llm_configs, self.tools),
            "step4": Step4Executor(self.state_manager, self.context_manager, self.config_dict, self.llm_configs, self.tools),
            "step5": Step5Executor(self.state_manager, self.context_manager, self.config_dict, self.llm_configs, self.tools, self.remapping_logic)
        }
        logger.debug("Step executors initialized.")
        return executors

    # --- Public Methods for Pipeline Control (called by CLI) ---

    def run_step(self, step_name: str, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        """
        Executes a specific step by name.

        Args:
            step_name (str): The name of the step to run (e.g., "step1", "step2").
            package_ids (Optional[List[str]]): Specific package IDs to process. If None, processes eligible packages.
            force (bool): If True, attempts to force reprocessing of packages even if they are in a completed or failed state for this step.
            **kwargs: Additional keyword arguments specific to the step executor (e.g., feedback_override for step4).

        Returns:
            bool: True if the step execution was successful (or had nothing to do), False otherwise.
        """
        executor = self.executors.get(step_name)
        if not executor:
            logger.error(f"No executor found for step: {step_name}")
            return False

        # Reload context manager's graph data before steps that might need it
        if step_name in ["step2", "step3", "step4", "step5"]:
             self._reload_context_manager_graph()

        # Pass the force flag and other kwargs down to the executor
        try:
            # Add force to the kwargs being passed to the executor
            kwargs['force'] = force
            return executor.execute(package_ids=package_ids, **kwargs)
        except Exception as e:
            logger.error(f"An unexpected error occurred running step '{step_name}': {e}", exc_info=True)
            # Update state to reflect failure? Executor should ideally handle this.
            self.state_manager.update_workflow_status(f'failed_{step_name}', f"Unexpected error: {e}")
            return False

    def run_full_pipeline(self):
        """Runs the entire conversion pipeline sequentially."""
        logger.info("--- Orchestrator: Starting Full Conversion Pipeline ---")
        steps = ["step1", "step2", "step3", "step4", "step5"]
        for step_name in steps:
            logger.info(f"--- Orchestrator: Executing {step_name} ---")
            success = self.run_step(step_name)
            if not success:
                logger.error(f"Pipeline halted due to failure in {step_name}.")
                # State should have been updated by the executor or run_step
                return
            # Check for remapping needed after step 5? No, step 5 handles triggering it.
            # The loop continues, but step 4/5 executors handle package eligibility.

        # Final status check after all steps attempt execution
        final_status = self._determine_final_status()
        self.state_manager.update_workflow_status(final_status)
        logger.info(f"--- Orchestrator: Full Conversion Pipeline Finished (Final Status: {final_status}) ---")


    def resume_pipeline(self):
        """Resumes the pipeline based on the last saved state."""
        logger.info("--- Orchestrator: Resuming Conversion Pipeline ---")
        last_status = self.state_manager.get_state().get('workflow_status', 'pending')
        logger.info(f"Resuming from status: {last_status}")

        # Determine which steps to run based on status
        steps_to_run = []
        if last_status in ['pending', 'failed_step1']:
            steps_to_run.extend(["step1", "step2", "step3", "step4", "step5"])
        elif last_status in ['step1_complete', 'failed_step2']:
            steps_to_run.extend(["step2", "step3", "step4", "step5"])
        elif last_status in ['step2_complete', 'failed_step3', 'running_step3']:
             # If step 3 failed or was running, try step 3 again (for eligible), then 4, 5
             steps_to_run.extend(["step3", "step4", "step5"])
        elif last_status in ['step3_complete', 'failed_step4', 'running_step4', 'failed_remapping']: # Also retry step 4 if remapping failed
             # If step 4 failed or was running, try step 4 again (for eligible), then 5
             steps_to_run.extend(["step4", "step5"])
        elif last_status in ['step4_complete', 'failed_step5', 'running_step5', 'needs_remapping']:
             # If step 5 failed or needs remapping, try step 5 again (for eligible)
             # If needs_remapping, Step 4 needs to run first
             if last_status == 'needs_remapping':
                  logger.info("Remapping needed, will attempt Step 4 then Step 5.")
                  steps_to_run.extend(["step4", "step5"]) # Pipeline runner handles logic
             else:
                  steps_to_run.extend(["step5"])

        if not steps_to_run:
             logger.info("Pipeline already completed or in an unrecoverable state.")
             return

        logger.info(f"Attempting to resume by running steps: {steps_to_run}")

        # --- The Remapping Loop ---
        # We need a loop here because Step 5 might trigger a remapping (needs_remapping),
        # requiring Step 4 to run again, followed by Step 5 again.
        max_pipeline_loops = self.config_dict.get("MAX_REMAPPING_ATTEMPTS", 1) + 2 # Allow initial run + remap attempts
        current_loop = 0
        while current_loop < max_pipeline_loops:
            current_loop += 1
            logger.info(f"--- Orchestrator: Resume/Remap Loop Iteration {current_loop}/{max_pipeline_loops} ---")
            needs_another_loop = False

            for step_name in steps_to_run:
                logger.info(f"--- Orchestrator: Executing {step_name} (Resume/Remap Loop) ---")
                # Determine eligible packages based on current state for this step
                eligible_packages = self._get_eligible_packages_for_resume(step_name)

                if not eligible_packages:
                     logger.info(f"No eligible packages for {step_name} in this loop iteration.")
                     continue # Skip to next step in steps_to_run

                # Special handling for remapping: Step 4 needs feedback
                kwargs = {}
                if step_name == "step4":
                     packages_needing_remap = [pkg_id for pkg_id in eligible_packages if self.state_manager.get_package_info(pkg_id).get('status') == 'needs_remapping']
                     if packages_needing_remap:
                          logger.info(f"Packages needing remapping for Step 4: {packages_needing_remap}")
                          feedback = self._generate_feedback_for_remapping(packages_needing_remap)
                          kwargs['feedback_override'] = feedback
                          # Only run step 4 for packages needing remapping in this specific case
                          eligible_packages = packages_needing_remap


                success = self.run_step(step_name, package_ids=eligible_packages, **kwargs)

                # Check state after execution
                current_state = self.state_manager.get_state()
                workflow_status = current_state.get('workflow_status')

                if f'failed_{step_name}' in workflow_status:
                    logger.error(f"Pipeline halted during resume due to failure in {step_name}.")
                    return # Exit resume attempt on failure

                # Check if remapping was triggered by Step 5
                if step_name == "step5" and any(p.get('status') == 'needs_remapping' for p in current_state.get('work_packages', {}).values()):
                    logger.info("Remapping triggered by Step 5. Will loop back to Step 4.")
                    needs_another_loop = True
                    steps_to_run = ["step4", "step5"] # Ensure next loop runs 4 then 5
                    break # Exit inner step loop, continue outer while loop

            if not needs_another_loop:
                 logger.info("No further remapping triggered. Exiting resume/remap loop.")
                 break # Exit the while loop

            if current_loop >= max_pipeline_loops:
                 logger.warning("Maximum pipeline resume/remap loops reached.")
                 # Mark packages still needing remap as failed?
                 packages = self.state_manager.get_all_packages()
                 for pkg_id, pkg_data in packages.items():
                      if pkg_data.get('status') == 'needs_remapping':
                           self.state_manager.update_package_state(pkg_id, 'failed_remapping', error="Max pipeline loops reached.")
                 break


        # Determine final overall status after resume attempt
        final_status = self._determine_final_status()
        self.state_manager.update_workflow_status(final_status)
        logger.info(f"--- Orchestrator: Resume Attempt Finished (Final Status: {final_status}) ---")

    def _get_eligible_packages_for_resume(self, step_name: str) -> List[str]:
         """Determines eligible packages for a step during resume."""
         target_status_map = {
             "step1": None, # Step 1 always runs if needed
             "step2": "step1_complete", # Should run if step 1 is done
             "step3": "identified",
             "step4": "structure_defined",
             "step5": "mapping_defined",
         }
         # Special case for remapping
         if step_name == "step4":
              # Include packages needing remapping OR normally eligible
              eligible = []
              packages = self.state_manager.get_all_packages()
              for pkg_id, pkg_data in packages.items():
                   status = pkg_data.get('status')
                   if status == 'needs_remapping' or status == 'structure_defined':
                        eligible.append(pkg_id)
              return eligible
         elif step_name == "step5":
              # Only run for packages that are newly mapped
              return self._get_eligible_packages('mapping_defined')
         else:
              target_status = target_status_map.get(step_name)
              if target_status:
                   return self._get_eligible_packages(target_status)
              else:
                   return [] # Should not happen for defined steps

    def _generate_feedback_for_remapping(self, package_ids: List[str]) -> Dict[str, str]:
         """Generates feedback strings for packages needing remapping."""
         feedback_map = {}
         for pkg_id in package_ids:
              # Need to load the failed task report for this package
              pkg_info = self.state_manager.get_package_info(pkg_id)
              report_artifact = pkg_info.get('artifacts', {}).get('task_results_report')
              if report_artifact:
                   report_path = os.path.join(self.analysis_dir, report_artifact)
                   if os.path.exists(report_path):
                        try:
                             with open(report_path, 'r', encoding='utf-8') as f:
                                  task_results = json.load(f)
                             failed_tasks = [t for t in task_results if t.get('status') == 'failed']
                             feedback_map[pkg_id] = self.remapping_logic.generate_mapping_feedback(failed_tasks)
                        except Exception as e:
                             logger.error(f"Failed to load or parse task results report {report_path} for feedback: {e}")
                             feedback_map[pkg_id] = "Error: Could not load previous failure details."
                   else:
                        feedback_map[pkg_id] = "Error: Previous task results report not found."
              else:
                   feedback_map[pkg_id] = "Error: No task results report artifact found."
         return feedback_map


    def _reload_context_manager_graph(self):
         """Reloads the include graph data into the context manager."""
         logger.debug("Attempting to reload include graph data into ContextManager...")
         graph_path = self.include_graph_path
         if os.path.exists(graph_path):
             try:
                 with open(graph_path, 'r', encoding='utf-8') as f:
                     graph = json.load(f)
                 self.context_manager.include_graph = graph
                 logger.info(f"Reloaded include graph data into ContextManager ({len(graph)} entries).")
             except Exception as e:
                 logger.error(f"Failed to reload include graph {graph_path}: {e}", exc_info=True)
                 self.context_manager.include_graph = {} # Reset on failure
         else:
             logger.warning(f"Include graph file not found for reload: {graph_path}")
             self.context_manager.include_graph = {}


    def _get_eligible_packages(self, target_status: str, specific_ids: Optional[List[str]] = None) -> List[str]:
        """Helper method moved from StepExecutor to be used centrally."""
        eligible = []
        packages = self.state_manager.get_all_packages()
        if not packages:
            logger.warning(f"No work packages found in state to check eligibility for status '{target_status}'.")
            return []

        for pkg_id, pkg_data in packages.items():
            if specific_ids and pkg_id not in specific_ids:
                continue

            current_status = pkg_data.get('status')
            if current_status == target_status:
                eligible.append(pkg_id)
            elif specific_ids and current_status != target_status:
                 logger.warning(f"Requested package '{pkg_id}' is not in the required status '{target_status}' (current: '{current_status}'). Skipping.")

        logger.debug(f"Found {len(eligible)} packages eligible for status '{target_status}' (Specific IDs requested: {specific_ids})")
        return eligible

    def _determine_final_status(self) -> str:
        """Checks the status of all packages to determine the overall workflow status."""
        # This logic remains similar to the original, using the state_manager
        packages = self.state_manager.get_all_packages()
        current_workflow_status = self.state_manager.get_state().get('workflow_status', 'unknown')

        if not packages:
            if current_workflow_status == 'step2_complete': return 'completed_no_packages'
            if current_workflow_status == 'step1_complete': return 'step1_complete'
            return current_workflow_status # Keep current status if no packages

        all_processed = True
        any_failed = False
        any_needs_remapping = False

        for pkg_data in packages.values():
            status = pkg_data.get('status')
            if status != 'processed':
                all_processed = False
            if status and 'failed' in status:
                any_failed = True
            if status == 'needs_remapping':
                 any_needs_remapping = True


        if all_processed:
            return 'completed'
        elif any_needs_remapping:
             return 'needs_remapping' # Indicate remapping is pending
        elif any_failed:
            # Prioritize reporting failure at the earliest step possible
            if any('failed_remapping' in p.get('status', '') for p in packages.values()): return 'failed_remapping'
            if any('failed_processing' in p.get('status', '') for p in packages.values()): return 'failed_step5'
            if any('failed_mapping' in p.get('status', '') for p in packages.values()): return 'failed_step4'
            if any('failed_structure' in p.get('status', '') for p in packages.values()): return 'failed_step3'
            # If only step 1 or 2 failed globally, that status would already be set.
            return 'failed' # Generic failure if specific step isn't clear
        else:
            # Not all processed, none failed, none need remapping -> incomplete
            # Return the status reflecting the furthest completed step globally
            if any(p.get('status') == 'mapping_defined' for p in packages.values()): return 'step4_complete'
            if any(p.get('status') == 'structure_defined' for p in packages.values()): return 'step3_complete'
            if any(p.get('status') == 'identified' for p in packages.values()): return 'step2_complete'
            # If only step 1 completed globally, that status would already be set.
            return 'incomplete' # Default if in intermediate running states
