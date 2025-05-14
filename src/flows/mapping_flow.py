# src/flows/mapping_flow.py
from crewai import Crew, CrewOutput, Flow, Process, Task # Flow is not directly used from crewai like this in the draft, but Task is.
# The draft defines class MappingFlow(Flow) where Flow is from crewai.flow.flow
# Assuming the user meant from crewai.flow.flow import Flow
from crewai.flow.flow import Flow
from typing import Dict, Any, List, Optional, Tuple
from src.logger_setup import get_logger
from src.core.context_manager import ContextManager
from src.core.state_manager import StateManager
from src.models.mapping_models import MappingOutput, TaskGroup, MappingTask # Ensure this path is correct
import src.config as config
import json

# Import agent getters from src.agents.step4
from src.agents.step4.cpp_code_analyst import get_cpp_code_analyst_agent
from src.agents.step4.godot_structure_analyst import get_godot_structure_analyst_agent
from src.agents.step4.conversion_strategist import get_conversation_strategist_agent
from src.agents.step4.task_decomposer import get_task_decomposer_agent
from src.agents.step4.json_output_formatter import get_json_output_fomratter_agent

# Import task creators (as per MappingFlow draft usage)
# These files will need to be created based on the draft or inferred.
from src.tasks.step4.analyze_cpp_task import create_analyze_cpp_task
from src.tasks.step4.analyze_godot_structure_task import create_analyze_godot_structure_task
from src.tasks.step4.define_strategy_task import create_define_strategy_task
# The draft's MappingFlow uses an inline task for decompose_tasks and merge_task_groups
# and calls create_format_json_task
from src.tasks.step4.format_json_task import create_format_json_task


# Import tools
from src.tools.cpp_code_analysis_tool import CppCodeAnalysisTool
from src.tools.structure_analysis_tool import StructureAnalysisTool

logger = get_logger(__name__)

class MappingFlow(Flow):
    """Flow for C++ to Godot mapping definition with component-based task planning."""
    
    # Define component types and their processing order
    COMPONENT_TYPES = ["scripts", "resources", "migration_scripts", "scenes"]
    
    def __init__(
        self, 
        state_manager: StateManager,
        context_manager: ContextManager,
        llm_config: Dict[str, Any], # Expects LLM instances: {"ANALYZER_MODEL": llm_obj, ...}
        package_id: str,
        is_remapping: bool = False,
        feedback: Optional[str] = None
    ):
        """
        Initialize the mapping flow.
        
        Args:
            state_manager: State manager instance
            context_manager: Context manager instance
            llm_config: LLM configuration containing actual LLM instances.
            package_id: ID of the package to process
            is_remapping: Whether this is a remapping run
            feedback: Optional feedback from previous run
        """
        self.state_manager = state_manager
        self.context_manager = context_manager
        self.llm_config = llm_config # Stores the passed LLM instances
        self.package_id = package_id
        self.is_remapping = is_remapping
        self.feedback = feedback
        self.structure_json_content = None  # Will be loaded in build_flow
        
        # Initialize agents with appropriate LLMs from llm_config
        analyzer_llm = self.llm_config.get("ANALYZER_MODEL")
        designer_planner_llm = self.llm_config.get("DESIGNER_PLANNER_MODEL")
        utility_llm = self.llm_config.get("UTILITY_MODEL")

        if not analyzer_llm or not designer_planner_llm or not utility_llm:
            # Log an error or raise an exception if any LLM is missing
            missing_llms_str = ", ".join(
                [name for name, llm_instance in {
                    "ANALYZER_MODEL": analyzer_llm,
                    "DESIGNER_PLANNER_MODEL": designer_planner_llm,
                    "UTILITY_MODEL": utility_llm
                }.items() if not llm_instance]
            )
            error_msg = f"Missing LLM instances in llm_config for MappingFlow: {missing_llms_str}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.cpp_analyst = get_cpp_code_analyst_agent(analyzer_llm)
        self.godot_analyst = get_godot_structure_analyst_agent(analyzer_llm)
        self.strategist = get_conversation_strategist_agent(designer_planner_llm) # Using 'get_conversation_strategist_agent'
        self.decomposer = get_task_decomposer_agent(designer_planner_llm)
        self.formatter = get_json_output_fomratter_agent(utility_llm) # Using 'get_json_output_fomratter_agent'
        
        # Initialize tools
        self.cpp_analysis_tool = CppCodeAnalysisTool(context_manager)
        self.structure_analysis_tool = StructureAnalysisTool(context_manager)
        
        # Add tools to agents (as per draft, though current step4 agents don't show tool usage in their definitions)
        # This might need adjustment if the step4 agents are not designed to use these tools.
        # For now, following the draft's MappingFlow structure.
        if hasattr(self.cpp_analyst, 'tools'):
            self.cpp_analyst.tools = [self.cpp_analysis_tool]
        if hasattr(self.godot_analyst, 'tools'):
            self.godot_analyst.tools = [self.structure_analysis_tool]
        
        super().__init__(
            name=f"C++ToGodotMappingFlow_{package_id}", # Ensure unique name if multiple flows run
            description=f"Flow for defining C++ to Godot mapping for package {package_id}",
            # expected_output="A structured JSON mapping definition with component-based task planning", # Not a Flow param
        )
    
    def _create_component_task(
        self, 
        component_type: str, 
        component_data: Dict[str, Any], 
        component_index: int,
        dependencies: List[str] # These are names of other tasks in the flow
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a task for planning conversion tasks for a specific component.
        
        Args:
            component_type: Type of component (scripts, resources, etc.)
            component_data: Component data from structure JSON
            component_index: Index of the component in its category
            dependencies: List of task names this task depends on
            
        Returns:
            Tuple of (task_name, task_definition for CrewAI flow)
        """
        task_name = f"plan_{component_type}_{component_index}"
        
        component_json = json.dumps(component_data, indent=2)
        component_context_str = (
            f"--- COMPONENT TYPE: {component_type} ---\n"
            f"--- COMPONENT DATA ---\n{component_json}\n"
        )
        
        # Context for this task will be:
        # 1. cpp_analysis_output (from 'analyze_cpp' task)
        # 2. godot_analysis_output (from 'analyze_godot_structure' task)
        # 3. strategy_output (from 'define_strategy' task)
        # 4. component_context_str (defined above)
        # These need to be referenced in the description using placeholders like {task_name.output}
        
        description = (
            f"Plan the conversion of this {component_type} component to Godot.\n\n"
            f"{component_context_str}\n\n"
            f"C++ Analysis:\n{{analyze_cpp.output}}\n\n"
            f"Godot Structure Analysis:\n{{analyze_godot_structure.output}}\n\n"
            f"Overall Conversion Strategy:\n{{define_strategy.output}}\n\n"
            f"Using the C++ analysis, Godot structure analysis, and overall conversion strategy, "
            f"define specific tasks needed to implement this {component_type} component in Godot.\n\n"
            f"For each task, include:\n"
            f"1. A clear title\n"
            f"2. A detailed description linking C++ elements to Godot implementation\n"
            f"3. Input source files from C++\n"
            f"4. Output Godot file path\n"
            f"5. Target element within the file (if applicable)"
        )
        
        component_task_obj = Task( # Renamed from component_task to avoid conflict
            name=task_name, # CrewAI Task name, not for flow definition
            description=description,
            expected_output=(
                f"A list of MappingTask items for this {component_type} component, each with:\n"
                f"- task_title: Concise title\n"
                f"- task_description: Detailed explanation\n"
                f"- input_source_files: List of C++ source files\n"
                f"- output_godot_file: Target Godot file path\n"
                f"- target_element: (Optional) Specific function/node"
            ),
            agent=self.decomposer,
            async_execution=False,
            # output_json=True # If expecting JSON list of MappingTask
        )
        
        task_definition = {
            "name": task_name, # Name for the flow step
            "task": component_task_obj,
            "dependencies": dependencies
        }
        
        return task_name, task_definition
    
    def _create_component_group_task(
        self, 
        component_type: str, 
        component_task_names: List[str] # Names of individual component planning tasks
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Creates a task for grouping component tasks into a TaskGroup.
        
        Args:
            component_type: Type of component (scripts, resources, etc.)
            component_task_names: List of flow task names whose outputs are lists of MappingTasks
            
        Returns:
            Tuple of (task_name, task_definition for CrewAI flow)
        """
        group_flow_task_name = f"group_{component_type}" # Name for the flow step
        
        dependency_outputs_str = "\n\n".join([f"--- Output of {dep_name} ---\n{{{dep_name}.output}}" for dep_name in component_task_names])
        
        description = (
            f"Group all the {component_type} component tasks (provided as outputs from previous steps) into a single TaskGroup object.\n\n"
            f"Inputs (outputs from previous tasks):\n{dependency_outputs_str}\n\n"
            f"Create a TaskGroup with:\n"
            f"1. group_title: A suitable title for the {component_type} group (e.g., '{component_type.replace('_', ' ').title()} Implementation')\n"
            f"2. feature_description: A comprehensive description of this component type's purpose in the project.\n"
            f"3. godot_features: Key Godot features, APIs, or patterns commonly used for this component type.\n"
            f"4. tasks: A single combined list of all individual MappingTask items from all the input component planning tasks."
        )
        
        group_task_obj = Task( # Renamed from group_task
            name=group_flow_task_name, # CrewAI Task name
            description=description,
            expected_output=(
                f"A single TaskGroup object (JSON-compatible dict) for {component_type} with:\n"
                f"- group_title (string)\n"
                f"- feature_description (string)\n"
                f"- godot_features (string)\n"
                f"- tasks (list of MappingTask-like dictionaries)"
            ),
            agent=self.decomposer, # Or a dedicated grouping agent if complex
            async_execution=False,
            # output_json=True # If expecting a TaskGroup JSON
        )
        
        task_definition = {
            "name": group_flow_task_name, # Name for the flow step
            "task": group_task_obj,
            "dependencies": component_task_names # Depends on all individual component planning tasks
        }
        
        return group_flow_task_name, task_definition
    
    def build_flow(self) -> List[Dict[str, Any]]:
        """
        Build the flow definition with component-based task planning.
        This method is called by the Flow base class.
        Returns:
            List of task definitions for the CrewAI flow.
        """
        pkg_info = self.state_manager.get_package_info(self.package_id)
        if not pkg_info:
            raise ValueError(f"Could not retrieve package info for {self.package_id}")
        
        structure_artifact_filename = pkg_info.get('artifacts', {}).get('structure_json')
        if not structure_artifact_filename:
            raise FileNotFoundError(f"Structure definition JSON artifact missing for package {self.package_id}")
        
        self.structure_json_content = self.state_manager.load_artifact(structure_artifact_filename, expect_json=True)
        if not self.structure_json_content:
            raise FileNotFoundError(f"Failed to load structure JSON artifact: {structure_artifact_filename}")
        
        existing_mapping_json_str = None
        if self.is_remapping:
            original_json_artifact_filename = f"package_{self.package_id}_mapping.json" # Assuming this is the non-remapped one
            existing_mapping_data = self.state_manager.load_artifact(original_json_artifact_filename, expect_json=True)
            if existing_mapping_data:
                existing_mapping_json_str = json.dumps(existing_mapping_data, indent=2)
        
        cpp_context_for_task = self.context_manager.get_work_package_source_code_content(
            self.package_id, max_tokens=config.MAX_CONTEXT_TOKENS // 3 # Adjusted token limit
        )
        godot_structure_context_for_task = json.dumps(self.structure_json_content, indent=2)
        
        # --- Define Tasks for the Flow ---
        # Task 1: Analyze C++
        analyze_cpp_task_obj = create_analyze_cpp_task(
            agent=self.cpp_analyst,
            context=cpp_context_for_task, # Provide C++ code
            package_id=self.package_id
        )
        analyze_cpp_flow_step = {
            "name": "analyze_cpp", "task": analyze_cpp_task_obj, "dependencies": []
        }
        
        # Task 2: Analyze Godot Structure
        analyze_godot_task_obj = create_analyze_godot_structure_task(
            agent=self.godot_analyst,
            context=godot_structure_context_for_task, # Provide Godot structure
            package_id=self.package_id
        )
        analyze_godot_flow_step = {
            "name": "analyze_godot_structure", "task": analyze_godot_task_obj, "dependencies": []
        }
        
        # Task 3: Define Strategy
        # Outputs from analyze_cpp_task_obj and analyze_godot_task_obj will be passed via context.
        define_strategy_task_obj: Task = create_define_strategy_task(
            agent=self.strategist,
            package_id=self.package_id,
            existing_mapping_json_str=existing_mapping_json_str, # Pass existing mapping directly
            feedback=self.feedback # Pass feedback directly
        )
        # Set the context for the strategy task
        define_strategy_task_obj.context = [analyze_cpp_task_obj, analyze_godot_task_obj]
        
        # The 'dependencies' key in flow_step is for conceptual understanding or a custom runner,
        # CrewAI's sequential process uses the task order and task.context.
        define_strategy_flow_step = {
            "name": "define_strategy", 
            "task": define_strategy_task_obj
            # "dependencies": ["analyze_cpp", "analyze_godot_structure"] # This is informational for this structure
        }
        
        flow_definition = [analyze_cpp_flow_step, analyze_godot_flow_step, define_strategy_flow_step]
        
        all_component_group_task_names = []
        
        for component_type in self.COMPONENT_TYPES:
            components_of_type = self.structure_json_content.get(component_type, [])
            if not components_of_type:
                logger.info(f"No {component_type} found in structure for package {self.package_id}, skipping component planning for this type.")
                continue
            
            individual_component_task_names_for_type = []
            for i, component_data_item in enumerate(components_of_type):
                comp_task_name, comp_task_def = self._create_component_task(
                    component_type=component_type,
                    component_data=component_data_item,
                    component_index=i,
                    # This task will need cpp_analysis, godot_analysis, and strategy outputs
                    # These will be set via its .context attribute
                    dependencies=["define_strategy"] 
                )
                # Set context for the component task
                # Assuming _create_component_task returns the Task object in task_def['task']
                comp_task_object: Task = comp_task_def['task']
                comp_task_object.context = [analyze_cpp_task_obj, analyze_godot_task_obj, define_strategy_task_obj]
                flow_definition.append(comp_task_def)
                individual_component_task_names_for_type.append(comp_task_name)
            
            if individual_component_task_names_for_type:
                group_task_name, group_task_def = self._create_component_group_task(
                    component_type=component_type,
                    component_task_names=individual_component_task_names_for_type # These are names, not Task objects
                    # The group task will need the *outputs* of these tasks.
                    # Its .context should be set to the list of actual Task objects.
                )
                # Set context for the group task
                group_task_object: Task = group_task_def['task']
                group_task_object.context = [flow_definition[flow_definition.index(next(fd for fd in flow_definition if fd['name'] == task_name))]['task'] for task_name in individual_component_task_names_for_type]

                flow_definition.append(group_task_def)
                all_component_group_task_names.append(group_task_name)
        
        # Task: Merge all component groups (TaskGroups)
        merge_all_groups_task_obj = Task( # Direct Task creation, description will refer to context
            name="merge_all_groups_task",
            description=(
                "Combine all component group outputs (TaskGroup objects provided in context) into a final list of TaskGroups. "
                "Also consider the overall strategy (provided in context) if it influences the merging or final structure."
            ),
            expected_output="A single list containing all TaskGroup objects/dictionaries from the component group tasks.",
            agent=self.decomposer,
            async_execution=False
        )
        # Set context for merge_all_groups_task
        # It needs the output of define_strategy and all group_{component_type} tasks
        merge_context_tasks = [define_strategy_task_obj] + \
                              [fd['task'] for fd in flow_definition if fd['name'] in all_component_group_task_names]
        merge_all_groups_task_obj.context = merge_context_tasks
        
        merge_all_groups_flow_step = {
            "name": "merge_all_groups",
            "task": merge_all_groups_task_obj
        }
        flow_definition.append(merge_all_groups_flow_step)
        
        # Final Task: Format JSON
        format_json_task_obj: Task = create_format_json_task( # Signature will change
            agent=self.formatter,
            package_id=self.package_id
            # strategy and task_groups will come from context
        )
        # Set context for format_json_task
        format_json_task_obj.context = [define_strategy_task_obj, merge_all_groups_task_obj]

        format_json_flow_step = {
            "name": "format_json", 
            "task": format_json_task_obj
        }
        flow_definition.append(format_json_flow_step)
        
        return flow_definition
    
    def run(self) -> Dict[str, Any]:
        """
        Builds the flow definition, creates a Crew, runs it, and returns the final JSON output.
        
        Returns:
            The mapping output as a dictionary.
        """
        # 1. Build the flow definition (list of task steps with Task objects)
        flow_steps_definition: List[Dict[str, Any]] = self.build_flow()

        # 2. Extract Task objects for the Crew
        # The `build_flow` method returns a list of dictionaries,
        # where each dictionary has a "task" key holding the actual Task object.
        tasks_for_crew: List[Task] = [step_def['task'] for step_def in flow_steps_definition]
        
        # 3. Gather all agents to be part of the Crew
        all_agents_for_crew = [
            self.cpp_analyst, self.godot_analyst, 
            self.strategist, self.decomposer, self.formatter
        ]

        # 4. Create the main Crew instance
        # The task descriptions use placeholders like {task_name.output}.
        # For these to work with a standard CrewAI sequential process, the tasks
        # must be defined such that their `context` parameter can be used, or agents
        # are expected to find previous outputs in memory.
        # The current task creation functions (e.g., create_define_strategy_task)
        # embed these placeholders directly in the description.
        # This implies that the Crew's execution process or a custom setup
        # would handle replacing these placeholders.
        # Standard sequential processing makes the output of task N available to task N+1.
        # If tasks need output from non-adjacent prior tasks, memory or explicit context passing is key.
        # For this implementation, we assume sequential context passing is sufficient or
        # agents are designed to retrieve necessary context.
        
        mapping_crew = Crew(
            agents=all_agents_for_crew,
            tasks=tasks_for_crew, # Pass the list of Task objects
            process=Process.sequential, 
            verbose=2, # TODO: Consider making this configurable via self.llm_config or similar
            memory=True # Enable memory for context sharing between tasks
        )

        logger.info(f"Kicking off main Mapping Crew for package: {self.package_id}")
        crew_result: CrewOutput = mapping_crew.kickoff() # Explicitly type hint for clarity
        
        # Log token usage if available
        if crew_result and hasattr(crew_result, 'token_usage') and crew_result.token_usage:
            logger.info(f"Main Mapping Crew finished for package: {self.package_id}. Tokens used: {crew_result.token_usage}")
        else:
            logger.info(f"Main Mapping Crew finished for package: {self.package_id}.")

        # 5. Process the crew_result to get the final output
        final_output_dict = None

        if not isinstance(crew_result, CrewOutput):
            logger.error(f"Crew kickoff did not return a CrewOutput object. Got: {type(crew_result)}")
            # Attempt to handle if it's a direct string or dict (less ideal for complex flows)
            if isinstance(crew_result, str):
                try:
                    final_output_dict = json.loads(crew_result)
                except json.JSONDecodeError:
                    raise ValueError(f"Crew kickoff returned a string that is not valid JSON: {crew_result}")
            elif isinstance(crew_result, dict):
                final_output_dict = crew_result
            else:
                raise ValueError(f"Unexpected output type from crew kickoff: {type(crew_result)}")

        else: # It is a CrewOutput object
            # The final output should be from the last task (format_json_task)
            if crew_result.tasks_output and len(crew_result.tasks_output) > 0:
                last_task_output = crew_result.tasks_output[-1]
                
                # Prioritize Pydantic output if the task was configured for it
                if hasattr(last_task_output, 'output_pydantic') and last_task_output.output_pydantic:
                    logger.info("Using Pydantic output from the last task for final result.")
                    final_output_dict = last_task_output.output_pydantic.model_dump()
                # Next, check for JSON dictionary output if task was configured for output_json
                elif hasattr(last_task_output, 'json_dict') and isinstance(last_task_output.json_dict, dict):
                    logger.info("Using json_dict (from output_json) from the last task for final result.")
                    final_output_dict = last_task_output.json_dict
                # Then, check raw_output if it's already a dictionary
                elif isinstance(last_task_output.raw_output, dict):
                     logger.info("Using raw_output (already a dict) from the last task for final result.")
                     final_output_dict = last_task_output.raw_output
                # If raw_output is a string, try to parse it
                elif isinstance(last_task_output.raw_output, str):
                    logger.info("Using raw_output (string) from the last task. Attempting to parse as JSON.")
                    try:
                        final_output_dict = json.loads(last_task_output.raw_output)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse raw_output string from last task as JSON: {e}")
                        logger.debug(f"Problematic raw_output string from last task: {last_task_output.raw_output}")
                        # Fallback to crew's raw output if last task's output is problematic
                        if isinstance(crew_result.raw, str):
                            logger.warning("Falling back to crew_result.raw due to parsing error in last task output.")
                            try:
                                final_output_dict = json.loads(crew_result.raw)
                            except json.JSONDecodeError as e_raw:
                                raise ValueError(f"Last task raw_output and crew_result.raw were strings but not valid JSON.") from e_raw
                        elif isinstance(crew_result.raw, dict):
                             final_output_dict = crew_result.raw
                        else:
                            raise ValueError(f"Last task raw_output was a string but not valid JSON, and crew_result.raw is not usable.") from e
                else:
                    logger.warning(f"Last task output in CrewOutput is not Pydantic, JSON string, or dict. Raw output type: {type(last_task_output.raw_output)}, Content: {str(last_task_output.raw_output)[:200]}")
                    # Try crew_result.raw as a last resort before failing
                    if isinstance(crew_result.raw, str):
                        try:
                            final_output_dict = json.loads(crew_result.raw)
                        except json.JSONDecodeError: pass # Let it fail at the None check
                    elif isinstance(crew_result.raw, dict):
                        final_output_dict = crew_result.raw

            elif isinstance(crew_result.raw, str): # If tasks_output is empty, try crew_result.raw
                logger.info("tasks_output is empty. Using raw output string from CrewOutput. Attempting to parse as JSON.")
                try:
                    final_output_dict = json.loads(crew_result.raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Crew's raw output was a string but not valid JSON: {crew_result.raw}") from e
            elif isinstance(crew_result.raw, dict):
                 logger.info("tasks_output is empty. Using raw output (dict) from CrewOutput.")
                 final_output_dict = crew_result.raw
            else:
                logger.error("CrewOutput received, but no usable output found (tasks_output empty and raw is not string/dict).")
                raise ValueError("Crew execution did not produce a usable final output.")
        
        if final_output_dict is None:
            logger.error(f"Could not derive a final dictionary output for package {self.package_id} after processing crew result.")
            raise ValueError(f"Could not derive a final dictionary output for package {self.package_id}.")

        # Validate the final dictionary against the Pydantic model
        try:
            validated_output = MappingOutput(**final_output_dict)
            logger.info(f"Final output for package {self.package_id} validated against MappingOutput model.")
            return validated_output.model_dump() # Return as dict
        except Exception as e:
            logger.error(f"Final output for package {self.package_id} does not conform to MappingOutput model: {e}", exc_info=True)
            logger.debug(f"Final output dict that failed validation for package {self.package_id}: {final_output_dict}")
            raise
