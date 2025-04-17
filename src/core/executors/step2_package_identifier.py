# src/core/executors/step2_package_identifier.py
import os
import json
from typing import Any, Dict, List, Optional
from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens # Import count_tokens
from src.agents.package_identifier import PackageIdentifierAgent
from src.tasks.identify_packages import IdentifyWorkPackagesTask
from crewai import Crew, Process
from src.logger_setup import get_logger

logger = get_logger(__name__)

class Step2Executor(StepExecutor):
    """Executes Step 2: Work Package Identification."""

    def execute(self, package_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Runs the work package identification using an LLM.

        Args:
            package_ids: Not used in this step.
            **kwargs: Not used in this step.

        Returns:
            bool: True if package identification was successful, False otherwise.
        """
        logger.info("--- Starting Step 2 Execution: Identify Work Packages ---")
        self.state_manager.update_workflow_status('running_step2')
        success = False

        # Load include graph data via ContextManager (which loads it on init)
        # Or should Orchestrator pass it explicitly? Let's assume ContextManager holds it.
        # Need to ensure ContextManager reloads if Step 1 just ran.
        # For now, assume graph is available via context_manager.include_graph
        include_graph_data = self.context_manager.include_graph
        if not include_graph_data:
             logger.error("Include graph data is missing from ContextManager. Cannot run Step 2. Run Step 1 first.")
             self.state_manager.update_workflow_status('failed_step2', "Include graph data missing for Step 2.")
             return False

        try:
            # Prepare context (just the graph JSON string)
            include_graph_json = json.dumps(include_graph_data, indent=2)

            # Check token count of the graph JSON before sending to agent
            graph_token_limit = int(self.config.get("MAX_CONTEXT_TOKENS", 800000) * 0.5) # Example: 50% limit for graph
            graph_tokens = count_tokens(include_graph_json)
            logger.debug(f"Include graph JSON token count: {graph_tokens} (Limit: {graph_token_limit})")
            if graph_tokens > graph_token_limit:
                error_msg = f"Include graph JSON ({graph_tokens} tokens) exceeds the estimated limit ({graph_token_limit}) for Step 2 context."
                logger.error(error_msg)
                raise ValueError(error_msg) # Error out as per concept constraints

            # Get the LLM instance for the analyzer role
            analyzer_llm = self._get_llm('analyzer') # Assuming 'analyzer' is the key in llm_map
            if not analyzer_llm:
                  raise ValueError("Analyzer LLM instance not found.")

            # Instantiate Agent and Task
            # Get the base agent definition
            agent_definition = PackageIdentifierAgent().get_agent()
            # Explicitly assign the configured LLM to the agent instance
            agent_definition.llm = analyzer_llm
            logger.debug(f"Assigned LLM {analyzer_llm.model} directly to agent {agent_definition.role}")
            # Create the task using the agent with the LLM assigned
            task = IdentifyWorkPackagesTask().create_task(agent_definition)

            # Create and run Crew, passing the agent with the LLM already set
            crew = Crew(
                agents=[agent_definition], # Use the agent with the LLM assigned
                tasks=[task],
                # llm=analyzer_llm, # Passing LLM here might be redundant now, but keep for safety
                process=Process.sequential,
                verbose=True
            )
            logger.info("Kicking off Crew for Step 2...")
            # Provide input directly to kickoff
            result = crew.kickoff(inputs={'include_graph_json': include_graph_json})

            logger.info("Step 2 Crew finished.")
            logger.debug(f"Step 2 Raw Result:\n{result}")

            # Process result (expected to be JSON string or dict if output_json worked)
            if isinstance(result, str):
                try:
                    parsed_packages = json.loads(result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON output from Step 2: {e}")
                    logger.debug(f"Raw output: {result}")
                    raise ValueError(f"Step 2 LLM output was not valid JSON: {e}")
            elif isinstance(result, dict):
                 parsed_packages = result
            else:
                 logger.error(f"Unexpected result type from Step 2 Crew: {type(result)}")
                 raise TypeError("Step 2 result was not string or dict.")

            # Validate basic structure
            if not isinstance(parsed_packages, dict):
                 raise TypeError("Step 2 result JSON is not a dictionary.")

            # Update state with packages using StateManager
            self.state_manager.set_packages(parsed_packages)
            package_count = len(self.state_manager.get_all_packages())
            logger.info(f"Step 2 identified {package_count} work packages.")
            self.state_manager.update_workflow_status('step2_complete')
            success = True

        except Exception as e:
            logger.error(f"An error occurred during Step 2 execution: {e}", exc_info=True)
            self.state_manager.update_workflow_status('failed_step2', f"Error in Step 2: {e}")
            success = False
        finally:
            logger.info("--- Finished Step 2 Execution ---")
            return success
