# src/flows/mapping_flow.py
import json
from typing import Dict, Any, Optional, List

from crewai import Task
from crewai.flow.flow import Flow, FlowPersistence, listen, start, and_
from crewai.flow import persist
from crewai.tasks.task_output import TaskOutput

from src.logger_setup import get_logger
from src.core.context_manager import ContextManager, STRUCTURE_ARTIFACT_SUFFIX, MAPPING_ARTIFACT_SUFFIX, count_tokens # Import constants and count_tokens
from src.core.state_manager import StateManager
from src.models.mapping_models import DefineMappingFlowState, MappingOutput, TaskGroup # TaskGroup for type hint
from src.utils.json_utils import parse_json_from_string
import src.config as config # For LLM instances or other configs

# Import agent getters from src.agents.step4
from src.agents.step4.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step4.godot_structure_analyst import get_godot_structure_analyst_agent
from src.agents.step4.conversion_strategist import get_conversation_strategist_agent
from src.agents.step4.task_decomposer import get_task_decomposer_agent
from src.agents.step4.json_output_formatter import get_json_output_fomratter_agent

# Import task creators
from src.tasks.step4.analyze_cpp_task import create_analyze_cpp_task
from src.tasks.step4.analyze_godot_structure_task import create_analyze_godot_structure_task
from src.tasks.step4.define_strategy_task import create_define_strategy_task
# Import the consolidated task for final assembly
from src.tasks.step4.merge_task_groups_task import create_merge_task_groups_task
# Task for decomposition (TaskDecomposerAgent)
# We'll define a task for the decomposer agent. It might be a new specific task creator or an ad-hoc Task.
# For now, let's assume we'll create an ad-hoc Task for the decomposer in the flow.

logger = get_logger(__name__)

class StateManagerFlowPersistence(FlowPersistence):
    """
    Custom FlowPersistence implementation using StateManager to save/load flow state.
    """
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.flow_state_artifact_prefix = "flow_state_mapping_" # e.g., flow_state_mapping_package123.json
        self.init_db() # Call init_db during construction

    def init_db(self) -> None:
        """Initialize the persistence backend."""
        # For StateManager, directory creation is handled by StateManager or config.
        # So, this can be a no-op or log initialization.
        logger.info("StateManagerFlowPersistence initialized. No specific DB setup needed.")
        pass

    def save_state(self, flow_uuid: str, method_name: str, state_data: Dict[str, Any]): # Updated signature
        """Saves the flow state using StateManager."""
        # method_name is available if needed for more granular logging or state versioning
        logger.debug(f"Saving state for flow_uuid: {flow_uuid} after method: {method_name}")
        artifact_filename = f"{self.flow_state_artifact_prefix}{flow_uuid}.json"

        # Ensure state_data is a dict if it's a Pydantic model, as save_artifact expects dict for JSON
        data_to_save = state_data
        if hasattr(state_data, 'model_dump') and callable(state_data.model_dump): # Check if it's a Pydantic model
            data_to_save = state_data.model_dump()
        elif not isinstance(state_data, dict):
            logger.error(f"state_data for flow {flow_uuid} is not a dict or Pydantic model. Type: {type(state_data)}. Cannot save.")
            return


        if self.state_manager.save_artifact(artifact_filename, data_to_save, is_json=True):
            logger.info(f"Flow state for '{flow_uuid}' saved to artifact: {artifact_filename}")
            current_step = data_to_save.get("current_step_name", "unknown")
            is_failed = data_to_save.get("is_failed", False)
            status_to_report = f"flow_failed_at_{current_step}" if is_failed else f"flow_paused_at_{current_step}"
            if data_to_save.get("is_complete", False):
                status_to_report = "flow_completed_mapping_definition"

            self.state_manager.update_package_state(
                package_id=flow_uuid,
                status=status_to_report
            )
        else:
            logger.error(f"Failed to save flow state for '{flow_uuid}' using StateManager.")

    def load_state(self, flow_uuid: str) -> Optional[Dict[str, Any]]: # Changed flow_id to flow_uuid
        """Loads the flow state using StateManager."""
        artifact_filename = f"{self.flow_state_artifact_prefix}{flow_uuid}.json"
        loaded_data = self.state_manager.load_artifact(artifact_filename, expect_json=True)
        if loaded_data and isinstance(loaded_data, dict):
            logger.info(f"Flow state for '{flow_uuid}' loaded from artifact: {artifact_filename}")
            return loaded_data
        logger.info(f"No persisted flow state found for '{flow_uuid}' (artifact: {artifact_filename}).")
        return None

class DefineMappingPipelineFlow(Flow[DefineMappingFlowState]):
    """
    CrewAI Flow for defining the C++ to Godot mapping for a work package.
    Manages a 5-step process involving multiple specialized agents.
    """
    # __init__ is removed to rely on the base Flow.__init__
    # Attributes will be set in the configure method.
    llm_instances: Optional[Dict[str, Any]] = None
    context_m: Optional[ContextManager] = None
    state_m: Optional[StateManager] = None
    cpp_analyst: Optional[Any] = None # Using Any for agent types for brevity
    godot_analyst: Optional[Any] = None
    strategist: Optional[Any] = None
    decomposer: Optional[Any] = None
    formatter: Optional[Any] = None
    # persistence will be set in configure

    def configure(
        self,
        llm_instances: Dict[str, Any],
        context_manager: ContextManager,
        state_manager: StateManager,
    ):
        """Configures the flow with necessary dependencies and initializes agents."""
        self.llm_instances = llm_instances
        self.context_m = context_manager
        self.state_m = state_manager

        # Initialize agents with their specific LLMs
        analyzer_llm = self.llm_instances.get("ANALYZER_MODEL")
        designer_planner_llm = self.llm_instances.get("DESIGNER_PLANNER_MODEL")
        utility_llm = self.llm_instances.get("UTILITY_MODEL")

        if not analyzer_llm:
            raise ValueError("ANALYZER_MODEL LLM instance is missing in llm_instances for DefineMappingPipelineFlow.")
        if not designer_planner_llm:
            raise ValueError("DESIGNER_PLANNER_MODEL LLM instance is missing in llm_instances for DefineMappingPipelineFlow.")
        if not utility_llm:
            raise ValueError("UTILITY_MODEL LLM instance is missing in llm_instances for DefineMappingPipelineFlow.")

        self.cpp_analyst = get_cpp_code_analyst_agent(analyzer_llm)
        self.godot_analyst = get_godot_structure_analyst_agent(analyzer_llm)
        self.strategist = get_conversation_strategist_agent(designer_planner_llm)
        self.decomposer = get_task_decomposer_agent(designer_planner_llm)
        self.formatter = get_json_output_fomratter_agent(utility_llm)

        # Setup persistence
        self.persistence = StateManagerFlowPersistence(state_manager)
        logger.info("DefineMappingPipelineFlow configured with StateManagerFlowPersistence and specific LLMs for agents.")

    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """
        Overrides kickoff to ensure 'id' in state is set from 'package_id' in inputs.
        'inputs' should contain 'package_id' and 'initial_context_str'.
        """
        if 'package_id' not in inputs:
            raise ValueError("DefineMappingPipelineFlow kickoff inputs must contain 'package_id'.")
        if 'initial_context_str' not in inputs: # Or handle if it can be optional
            logger.warning("DefineMappingPipelineFlow kickoff inputs missing 'initial_context_str'. Proceeding with None.")
            inputs['initial_context_str'] = None


        # The 'id' for the flow state MUST match 'package_id' for our persistence logic
        inputs['id'] = inputs['package_id']
        logger.info(f"Kicking off DefineMappingPipelineFlow for package_id: {inputs['package_id']}.")
        return super().kickoff(inputs=inputs)

    @start()
    def step0_prepare_contexts(self):
        """
        Prepares specific contexts from the initial comprehensive context string.
        This step updates the flow's state with prepared data.
        """
        self.state.current_step_name = "preparing_contexts"
        logger.info(f"Flow {self.state.id}: Starting step0_prepare_contexts for package {self.state.package_id}")
        try:
            # Use ContextManager to get various pieces of information
            # For simplicity, we'll assume initial_context_str contains enough,
            # or ContextManager methods can derive what's needed from package_id.

            self.state.general_instructions = self.context_m.get_instruction_context() # Global instructions

            # Get C++ source for the package
            self.state.cpp_source_for_analyst = self.context_m.get_work_package_source_code_content(
                self.state.package_id,
                max_tokens=config.MAX_CONTEXT_TOKENS
            )
            if not self.state.cpp_source_for_analyst:
                logger.warning(f"Flow {self.state.id}: No C++ source content found for package {self.state.package_id}.")


            # Get Godot structure (already defined in Step 3, load from artifact)
            structure_artifact_name = f"package_{self.state.package_id}{STRUCTURE_ARTIFACT_SUFFIX}"
            structure_data = self.state_m.load_artifact(structure_artifact_name, expect_json=True)
            if structure_data:
                self.state.godot_structure_for_analyst = json.dumps(structure_data, indent=2)
            else:
                logger.warning(f"Flow {self.state.id}: Godot structure artifact '{structure_artifact_name}' not found or failed to load.")
                self.state.godot_structure_for_analyst = "{}" # Empty JSON object as placeholder

            # Get existing mapping and feedback if remapping (example logic)
            # This would typically be passed in via kickoff `inputs` if it's a remapping run
            # For now, assume they might be in initial_context_str or handled by ContextManager
            # self.state.existing_mapping_for_strategist = self.context_m.get_existing_mapping(self.state.package_id)
            # self.state.feedback_for_strategist = self.context_m.get_feedback(self.state.package_id)
            # For this example, let's assume these are directly from kickoff inputs if needed,
            # and already set in self.state by Flow's default state initialization from kickoff inputs.

            logger.info(f"Flow {self.state.id}: Contexts prepared. CPP source tokens: {count_tokens(self.state.cpp_source_for_analyst or '')}, Godot structure tokens: {count_tokens(self.state.godot_structure_for_analyst or '')}") # Use imported count_tokens
            # This method doesn't return a Task, it prepares state for subsequent tasks.
            # To trigger next steps, it should return something that @listen can pick up.
            # Or, the next steps can listen to the flow start itself if they don't depend on this method's *output*.
            # Let's return a confirmation.
            return {"status": "contexts_prepared"}
        except Exception as e:
            self.state.is_failed = True
            self.state.error_message = f"Error in step0_prepare_contexts: {str(e)}"
            logger.error(f"Flow {self.state.id}: {self.state.error_message}", exc_info=True)
            # Re-raise to stop the flow or handle as per Flow's error handling
            raise

    @listen("step0_prepare_contexts")
    async def step1a_analyze_cpp(self, _previous_output: Any):
        """Analyzes C++ code and returns the raw analysis string."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure", "output": "Error: Previous step failed."}
        self.state.current_step_name = "analyzing_cpp"
        logger.info(f"Flow {self.state.id}: Starting step1a_analyze_cpp for package {self.state.package_id}")

        if not self.state.cpp_source_for_analyst:
            logger.warning(f"Flow {self.state.id}: Skipping C++ analysis due to missing source content.")
            self.state.cpp_analysis_raw = "Error: C++ source content was not available for analysis."
            return self.state.cpp_analysis_raw

        task_def = create_analyze_cpp_task(
            agent=self.cpp_analyst,
            context=self.state.cpp_source_for_analyst,
            package_id=self.state.package_id
        )
        
        from crewai import Crew, Process # Local import for temp crew
        temp_crew = Crew(agents=[self.cpp_analyst], tasks=[task_def], process=Process.sequential, verbose=False)
        logger.info(f"Flow {self.state.id}: Kicking off temporary crew for C++ analysis.")
        crew_result = await temp_crew.kickoff_async()

        self.state.cpp_analysis_raw = crew_result.raw if crew_result and hasattr(crew_result, 'raw') else "C++ analysis failed or produced no output."
        logger.info(f"Flow {self.state.id}: C++ analysis completed. Raw output length: {len(self.state.cpp_analysis_raw)}")
        return self.state.cpp_analysis_raw

    @listen("step0_prepare_contexts")
    async def step1b_analyze_godot_structure(self, _previous_output: Any):
        """Analyzes Godot project structure and returns the raw analysis string."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure", "output": "Error: Previous step failed."}
        self.state.current_step_name = "analyzing_godot_structure"
        logger.info(f"Flow {self.state.id}: Starting step1b_analyze_godot_structure for package {self.state.package_id}")

        if not self.state.godot_structure_for_analyst:
            logger.warning(f"Flow {self.state.id}: Skipping Godot structure analysis due to missing structure content.")
            self.state.godot_analysis_raw = "Error: Godot project structure was not available for analysis."
            return self.state.godot_analysis_raw

        task_def = create_analyze_godot_structure_task(
            agent=self.godot_analyst,
            context=self.state.godot_structure_for_analyst,
            package_id=self.state.package_id
        )

        from crewai import Crew, Process
        temp_crew = Crew(agents=[self.godot_analyst], tasks=[task_def], process=Process.sequential, verbose=False)
        logger.info(f"Flow {self.state.id}: Kicking off temporary crew for Godot structure analysis.")
        crew_result = await temp_crew.kickoff_async()

        self.state.godot_analysis_raw = crew_result.raw if crew_result and hasattr(crew_result, 'raw') else "Godot structure analysis failed or produced no output."
        logger.info(f"Flow {self.state.id}: Godot structure analysis completed. Raw output length: {len(self.state.godot_analysis_raw)}")
        return self.state.godot_analysis_raw

    # Removed save_cpp_analysis_result and save_godot_analysis_result as steps 1a and 1b now update state and return raw data.

    @listen(and_("step1a_analyze_cpp", "step1b_analyze_godot_structure"))
    async def step2_define_strategy(self): # Listener for and_ condition does not receive prior outputs as args
        """Defines the conversion strategy using results from previous steps stored in self.state."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure", "output": "Error: Previous step failed."}
        self.state.current_step_name = "defining_strategy"
        logger.info(f"Flow {self.state.id}: Starting step2_define_strategy for package {self.state.package_id}")

        # Retrieve analysis results from state, which should have been set by step1a and step1b
        cpp_analysis = self.state.cpp_analysis_raw or "C++ analysis not available in state."
        godot_analysis = self.state.godot_analysis_raw or "Godot analysis not available in state."
        
        task_def = create_define_strategy_task(
            agent=self.strategist,
            cpp_analysis=cpp_analysis,
            godot_analysis=godot_analysis,
            package_id=self.state.package_id,
            existing_mapping=self.state.existing_mapping_for_strategist,
            feedback=self.state.feedback_for_strategist
        )
        
        from crewai import Crew, Process
        temp_crew = Crew(agents=[self.strategist], tasks=[task_def], process=Process.sequential, verbose=False)
        logger.info(f"Flow {self.state.id}: Kicking off temporary crew for strategy definition.")
        crew_result = await temp_crew.kickoff_async()
        
        self.state.strategy_raw = crew_result.raw if crew_result and hasattr(crew_result, 'raw') else "Strategy definition failed or produced no output."
        logger.info(f"Flow {self.state.id}: Strategy definition completed. Raw output length: {len(self.state.strategy_raw)}")
        return self.state.strategy_raw

    @listen("step2_define_strategy")
    async def step3_decompose_tasks(self, strategy_raw: str): # Receives raw strategy string
        """Decomposes the strategy into detailed task groups."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure", "output": "Error: Previous step failed."}
        self.state.current_step_name = "decomposing_tasks"
        logger.info(f"Flow {self.state.id}: Starting step3_decompose_tasks for package {self.state.package_id}")
        self.state.strategy_raw = strategy_raw # Update state with received strategy

        decomposition_prompt = (
            f"**Objective:** Decompose the conversion plan for package '{self.state.package_id}' into detailed Task Groups and MappingTasks.\n\n"
            f"**Package ID:** {self.state.package_id}\n\n"
            f"**Overall Conversion Strategy:**\n{self.state.strategy_raw}\n\n"
            f"**C++ Code Analysis Summary:**\n{self.state.cpp_analysis_raw}\n\n"
            f"**Proposed Godot Structure Summary:**\n{self.state.godot_analysis_raw}\n\n"
            f"**Instructions:**\n"
            f"1. Review all provided information.\n"
            f"2. Identify logical features or components based on the strategy and analyses.\n"
            f"3. For each feature/component, define a 'TaskGroup' containing:\n"
            f"   - `group_title` (string): Descriptive title.\n"
            f"   - `feature_description` (string): Overall purpose of this group.\n"
            f"   - `godot_features` (string): Key Godot elements to be used.\n"
            f"   - `tasks` (list of 'MappingTask' objects): Granular tasks with:\n"
            f"     - `task_title` (string)\n"
            f"     - `task_description` (string): Link C++ to Godot implementation.\n"
            f"     - `input_source_files` (list of strings): Relevant C++ files.\n"
            f"     - `output_godot_file` (string): Target Godot file (e.g., res://...).\n"
            f"     - `target_element` (string, optional): Specific function/node.\n"
            f"4. Ensure tasks are small and actionable for a code generation agent.\n"
            f"5. Your final output MUST be a valid JSON object with the following structure:\n"
            f"   ```json\n"
            f"   {{\n"
            f"     \"package_id\": \"{self.state.package_id}\",\n"
            f"     \"mapping_strategy\": \"Your high-level strategy summary\",\n"
            f"     \"task_groups\": [/* Array of TaskGroup objects */]\n"
            f"   }}\n"
            f"   ```\n"
            f"   Do not include any text before or after the JSON object, and do not use markdown formatting like ```json."
        )
        expected_decomposition_output = "A JSON object with 'package_id', 'mapping_strategy', and 'task_groups' fields. The 'task_groups' field should contain an array of TaskGroup objects. Your output MUST be a valid JSON string with no additional text or formatting."

        task_def = Task(
            name=f"DecomposeConversionTasks_{self.state.package_id}",
            description=decomposition_prompt,
            expected_output=expected_decomposition_output,
            agent=self.decomposer,
            output_pydantic=MappingOutput  # Use existing MappingOutput model for validation
        )

        from crewai import Crew, Process
        temp_crew = Crew(agents=[self.decomposer], tasks=[task_def], process=Process.sequential, verbose=False)
        logger.info(f"Flow {self.state.id}: Kicking off temporary crew for task decomposition.")
        crew_result = await temp_crew.kickoff_async()

        # Primary approach: Check if Pydantic parsing was successful
        if crew_result and hasattr(crew_result, 'pydantic') and crew_result.pydantic is not None:
            if isinstance(crew_result.pydantic, MappingOutput) and hasattr(crew_result.pydantic, 'task_groups'):
                task_groups = crew_result.pydantic.task_groups
                logger.info(f"Flow {self.state.id}: Pydantic parsing successful, got {len(task_groups)} task groups")
                # Convert Pydantic models to dict and then to JSON string
                task_groups_list = [
                    group.model_dump() if hasattr(group, 'model_dump') else group 
                    for group in task_groups
                ]
                self.state.task_groups_json_str = json.dumps(task_groups_list, indent=2)
            else:
                logger.warning(f"Flow {self.state.id}: Pydantic result is not a MappingOutput or has no task_groups. Type: {type(crew_result.pydantic)}")
                # Fall back to raw output parsing
                self.state.task_groups_json_str = "[]"
        else:
            # Fallback approach: Use parse_json_from_string utility
            logger.info(f"Flow {self.state.id}: Pydantic parsing not available, falling back to manual JSON parsing")
            raw_output = crew_result.raw if crew_result and hasattr(crew_result, 'raw') else ""
            
            if raw_output:
                # Try to extract JSON from the raw output
                parsed_data = parse_json_from_string(raw_output)
                if parsed_data is not None:
                    if isinstance(parsed_data, list):
                        # If it's a list, assume it's a list of task groups
                        logger.info(f"Flow {self.state.id}: Successfully parsed {len(parsed_data)} task groups (as list) using parse_json_from_string")
                        self.state.task_groups_json_str = json.dumps(parsed_data, indent=2)
                    elif isinstance(parsed_data, dict) and 'task_groups' in parsed_data and isinstance(parsed_data['task_groups'], list):
                        # If it's a dict with a task_groups field, extract the task_groups
                        task_groups = parsed_data['task_groups']
                        logger.info(f"Flow {self.state.id}: Successfully parsed {len(task_groups)} task groups (from MappingOutput) using parse_json_from_string")
                        self.state.task_groups_json_str = json.dumps(task_groups, indent=2)
                    else:
                        logger.warning(f"Flow {self.state.id}: Parsed data is not a list or MappingOutput. Type: {type(parsed_data)}")
                        self.state.task_groups_json_str = "[]"
                else:
                    logger.warning(f"Flow {self.state.id}: Failed to parse task groups from raw output using parse_json_from_string")
                    self.state.task_groups_json_str = "[]"
                    # Log a snippet of the raw output for debugging
                    logger.debug(f"Flow {self.state.id}: Raw output snippet: {raw_output[:500]}...")
            else:
                logger.warning(f"Flow {self.state.id}: No raw output available from task decomposition")
                self.state.task_groups_json_str = "[]"
        
        logger.info(f"Flow {self.state.id}: Task decomposition completed. Final task_groups_json_str length: {len(self.state.task_groups_json_str)}")
        return self.state.task_groups_json_str

    @listen("step3_decompose_tasks")
    async def step4_assemble_final_mapping(self, task_groups_json_str: str): # Receives JSON string
        """Assembles the final MappingOutput JSON by executing the task internally."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure", "output": None}
        self.state.current_step_name = "assembling_mapping"
        logger.info(f"Flow {self.state.id}: Starting step4_assemble_final_mapping for package {self.state.package_id}")
        self.state.task_groups_json_str = task_groups_json_str # Update state

        parsed_task_groups = []
        try:
            # Check if task_groups_json_str is empty or just whitespace
            if not self.state.task_groups_json_str or self.state.task_groups_json_str.strip() == "":
                logger.warning(f"Flow {self.state.id}: task_groups_json_str is empty or whitespace. Using empty list.")
            elif self.state.task_groups_json_str.strip() == "[]":
                logger.warning(f"Flow {self.state.id}: task_groups_json_str is an empty array. Using empty list.")
            else:
                # Try to parse the JSON string
                parsed_task_groups = json.loads(self.state.task_groups_json_str)
                if not isinstance(parsed_task_groups, list):
                    logger.warning(f"Flow {self.state.id}: Decomposer output was not a list. Using empty list.")
                    parsed_task_groups = []
                else:
                    logger.info(f"Flow {self.state.id}: Successfully parsed {len(parsed_task_groups)} task groups from decomposer output.")
                    # Log the first few task groups for debugging
                    for i, group in enumerate(parsed_task_groups[:3]):  # Log up to 3 groups
                        group_title = group.get('group_title', f"Group {i+1}")
                        task_count = len(group.get('tasks', []))
                        logger.debug(f"Flow {self.state.id}: Task Group {i+1}: '{group_title}' with {task_count} tasks")
        except json.JSONDecodeError as e:
            logger.error(f"Flow {self.state.id}: Failed to parse task_groups_json_str: {e}. Using empty list.")
            # Log more details about the JSON string for debugging
            logger.debug(f"Flow {self.state.id}: Raw task_groups_json_str (first 500 chars): {self.state.task_groups_json_str[:500]}...")
            logger.debug(f"Flow {self.state.id}: Raw task_groups_json_str (last 500 chars): {self.state.task_groups_json_str[-500:] if len(self.state.task_groups_json_str) > 500 else ''}")
            
            # Try to use parse_json_from_string as a fallback
            logger.info(f"Flow {self.state.id}: Attempting to use parse_json_from_string as fallback")
            fallback_parsed = parse_json_from_string(self.state.task_groups_json_str)
            if fallback_parsed is not None and isinstance(fallback_parsed, list):
                logger.info(f"Flow {self.state.id}: Successfully parsed {len(fallback_parsed)} task groups using parse_json_from_string")
                parsed_task_groups = fallback_parsed
            else:
                logger.warning(f"Flow {self.state.id}: Fallback parsing also failed. Using empty list.")
                parsed_task_groups = []

        final_assembly_task_def = create_merge_task_groups_task(
            agent=self.formatter, # Agent is passed to task creator, but task needs to be run
            strategy=self.state.strategy_raw or "Strategy not available.",
            group_tasks_outputs=parsed_task_groups,
            package_id=self.state.package_id
        )
        
        # Log the parsed task groups for debugging
        logger.info(f"Flow {self.state.id}: Preparing to assemble final mapping with {len(parsed_task_groups)} task groups.")
        if parsed_task_groups:
            for i, group in enumerate(parsed_task_groups):
                group_title = group.get('group_title', f"Group {i+1}")
                task_count = len(group.get('tasks', []))
                logger.info(f"Flow {self.state.id}: Task Group {i+1}: '{group_title}' with {task_count} tasks")
        
        # Execute the task using the agent
        # This requires the agent to have a method to execute a task, or wrap in a temp crew
        # For simplicity, assuming agent.execute_task(task_definition) exists or similar
        # More robust: Create a temporary crew
        from crewai import Crew, Process
        temp_crew = Crew(
            agents=[self.formatter],
            tasks=[final_assembly_task_def],
            process=Process.sequential,
            verbose=False # Keep this less verbose for sub-steps
        )
        logger.info(f"Flow {self.state.id}: Kicking off temporary crew for final assembly task.")
        crew_result = await temp_crew.kickoff_async() # Assuming kickoff_async is available and appropriate

        if crew_result and crew_result.pydantic and isinstance(crew_result.pydantic, MappingOutput):
            logger.info(f"Flow {self.state.id}: Final assembly task completed successfully.")
            
            # Log details about the output for debugging
            mapping_output = crew_result.pydantic
            task_groups_count = len(mapping_output.task_groups) if mapping_output.task_groups else 0
            logger.info(f"Flow {self.state.id}: Final MappingOutput contains {task_groups_count} task groups")
            
            if task_groups_count == 0:
                logger.warning(f"Flow {self.state.id}: Final MappingOutput has EMPTY task_groups list!")
                # Check if raw output contains task_groups
                if crew_result.raw and '"task_groups"' in crew_result.raw:
                    logger.debug(f"Flow {self.state.id}: Raw output contains task_groups key, but parsed as empty. Raw sample: {crew_result.raw[:500]}...")
            
            return crew_result.pydantic # Return the Pydantic model directly
        else:
            raw_output_sample = str(crew_result.raw)[:500] if crew_result and hasattr(crew_result, 'raw') else "N/A"
            logger.error(f"Flow {self.state.id}: Final assembly task failed or returned unexpected output. Raw: {raw_output_sample}")
            self.state.is_failed = True
            self.state.error_message = f"Final assembly task failed. Raw output sample: {raw_output_sample}"
            return None # Indicate failure to the next step

    @listen("step4_assemble_final_mapping")
    def step5_complete_flow(self, final_mapping_output: Optional[MappingOutput]): # Expects MappingOutput or None
        """Completes the flow and saves the final mapping output."""
        if self.state.is_failed: return {"status": "skipped_due_to_previous_failure"} # If already failed, skip
        if not final_mapping_output: # If previous step indicated failure by returning None
            self.state.is_failed = True
            if not self.state.error_message: # Ensure an error message is set
                 self.state.error_message = "Final assembly step (step4) returned no output."
            logger.error(f"Flow {self.state.id}: {self.state.error_message}")
            self.state_m.update_package_state(package_id=self.state.package_id, status="mapping_definition_failed", error=self.state.error_message)
            return None

        self.state.current_step_name = "completing_flow"
        logger.info(f"Flow {self.state.id}: Starting step5_complete_flow for package {self.state.package_id}")

        self.state.final_mapping_output = final_mapping_output
        # Save the final MappingOutput as a primary artifact for the package
        mapping_artifact_name = f"package_{self.state.package_id}{MAPPING_ARTIFACT_SUFFIX}"
        if self.state_m.save_artifact(mapping_artifact_name, self.state.final_mapping_output.model_dump(), is_json=True):
            logger.info(f"Flow {self.state.id}: Final mapping output saved as artifact: {mapping_artifact_name}")
            self.state_m.update_package_state(
                package_id=self.state.package_id,
                status="mapping_defined",
                artifacts={"mapping_json": mapping_artifact_name}
            )
        else:
            self.state.is_failed = True
            self.state.error_message = f"Failed to save final mapping artifact: {mapping_artifact_name}"
            logger.error(f"Flow {self.state.id}: {self.state.error_message}")

        self.state.is_complete = not self.state.is_failed
        logger.info(f"Flow {self.state.id}: Processing complete. Success: {self.state.is_complete}")
        return self.state.final_mapping_output # Return the Pydantic model
        # No specific error handling here for the case where final_mapping_output is None,
        # as it's caught at the beginning of the method.
