# src/core/executors/step1_analyzer.py
import os
from typing import Any, Dict, List, Optional
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager
# Use the new regex-based analyzer
#from src.utils.dependency_analyzer import generate_include_graph
from src.utils.dependency_analyzer_regex import generate_include_graph_regex
from src.logger_setup import get_logger

logger = get_logger(__name__)

class Step1Executor(StepExecutor):
    """Executes Step 1: C++ Include Dependency Analysis."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager, # ContextManager might not be strictly needed here, but keep for consistency?
                 config: Dict[str, Any]):
        # Step 1 doesn't need LLMs or specific tools beyond the dependency analyzer util
        super().__init__(state_manager, context_manager, config, llm_map={}, tools={})
        self.cpp_project_dir = os.path.abspath(config.get("CPP_PROJECT_DIR", "data/cpp_project"))
        self.analysis_dir = os.path.abspath(config.get("ANALYSIS_OUTPUT_DIR", "analysis_output"))
        self.include_graph_path = os.path.join(self.analysis_dir, "dependencies.json")

    def execute(self, package_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Runs the dependency analysis.

        Args:
            package_ids: Not used in this step.
            **kwargs: Not used in this step.

        Returns:
            bool: True if analysis was successful, False otherwise.
        """
        logger.info("--- Starting Step 1 Execution: Analyze Dependencies ---")
        self.state_manager.update_workflow_status('running_step1')
        success = False
        try:
            analysis_success = generate_include_graph_regex(self.cpp_project_dir, self.include_graph_path)

            if analysis_success:
                logger.info("Step 1 dependency analysis (regex) completed successfully.")
                # Reload graph data in ContextManager? Orchestrator should handle this after step execution.
                self.state_manager.update_workflow_status('step1_complete')
                success = True
            else:
                logger.error("Step 1 dependency analysis failed during graph generation.")
                self.state_manager.update_workflow_status('failed_step1', "Dependency graph generation failed.")
                success = False

        except Exception as e:
            logger.error(f"An unexpected error occurred during Step 1 execution: {e}", exc_info=True)
            self.state_manager.update_workflow_status('failed_step1', f"Unexpected error in Step 1: {e}")
            success = False
        finally:
            logger.info("--- Finished Step 1 Execution ---")
            return success
