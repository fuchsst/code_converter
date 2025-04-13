# src/core/orchestrator.py
import os
import json
from logger_setup import get_logger
import config
from .context_manager import ContextManager, read_file_content
from .api_utils import call_gemini_api, get_gemini_model
from src.agents.package_identifier import PackageIdentifierAgent
from src.tasks.identify_packages import IdentifyWorkPackagesTask
from src.agents.structure_definer import StructureDefinerAgent
from src.tasks.define_structure import DefineStructureTask
from src.agents.mapping_definer import MappingDefinerAgent
from src.tasks.define_mapping import DefineMappingTask
from src.agents.code_processor import CodeProcessorAgent
from src.tasks.process_code import ProcessCodeTask
from src.utils.dependency_analyzer import generate_include_graph
from src.utils.parser_utils import parse_step4_output
from crewai import Crew, Process

logger = get_logger(__name__)

DEFAULT_STATE_FILE = "orchestrator_state.json"

class Orchestrator:
    """
    Manages the C++ to Godot conversion workflow, orchestrating agents and tasks
    across multiple steps, handling state persistence for interruptibility.
    """
    def __init__(self,
                 cpp_project_dir: str = config.CPP_PROJECT_DIR,
                 godot_project_dir: str = config.GODOT_PROJECT_DIR,
                 output_dir: str = config.OUTPUT_DIR,
                 analysis_dir: str = config.ANALYSIS_OUTPUT_DIR,
                 target_language: str = config.TARGET_LANGUAGE):
        """
        Initializes the Orchestrator.

        Args:
            cpp_project_dir (str): Path to the C++ project source.
            godot_project_dir (str): Path to the existing/target Godot project.
            output_dir (str): Directory to save final converted Godot files.
            analysis_dir (str): Directory to save intermediate analysis files (graph, state, etc.).
            target_language (str): Target language (e.g., "GDScript").
        """
        self.cpp_project_dir = os.path.abspath(cpp_project_dir)
        self.godot_project_dir = os.path.abspath(godot_project_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_dir = os.path.abspath(analysis_dir)
        self.target_language = target_language
        self.state_file_path = os.path.join(self.analysis_dir, DEFAULT_STATE_FILE)
        self.include_graph_path = os.path.join(self.analysis_dir, "dependencies.json")

        # Ensure analysis directory exists
        os.makedirs(self.analysis_dir, exist_ok=True)

        self.state = self.load_state() # Load existing state or initialize new
        self.include_graph = self._load_include_graph_data()
        self.context_manager = ContextManager(self.include_graph_path, self.cpp_project_dir)

        # --- Initialize LLM Instances ---
        self.analyzer_llm = get_gemini_model(config.ANALYZER_MODEL)
        self.mapper_llm = get_gemini_model(config.MAPPER_MODEL)
        self.generator_llm = get_gemini_model(config.GENERATOR_EDITOR_MODEL)
        # Add others if needed (e.g., reviewer_llm)
        if not all([self.analyzer_llm, self.mapper_llm, self.generator_llm]):
             # Log details within get_gemini_model
             logger.error("One or more LLM instances failed to initialize. Check API key and model names.")
             # Decide if this is fatal - likely yes for most operations
             # raise RuntimeError("Failed to initialize required LLM instances.")

        logger.info(f"Orchestrator initialized.")
        logger.info(f"  CPP Project Dir: {self.cpp_project_dir}")
        logger.info(f"  Godot Project Dir: {self.godot_project_dir}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Analysis Dir: {self.analysis_dir}")
        logger.info(f"  State File: {self.state_file_path}")
        logger.info(f"  Target Language: {self.target_language}")
        logger.info(f"  Initial State Keys: {list(self.state.keys())}")


    # --- State Management ---

    def load_state(self) -> dict:
        """Loads the orchestrator state from the state file."""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"Loaded existing state from {self.state_file_path}")
                # Basic validation/migration could happen here if state format changes
                if 'work_packages' not in state: state['work_packages'] = {}
                return state
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load state file {self.state_file_path}: {e}. Initializing new state.", exc_info=True)
                return self._initialize_state()
        else:
            logger.info("No existing state file found. Initializing new state.")
            return self._initialize_state()

    def _initialize_state(self) -> dict:
        """Returns a dictionary representing the initial state."""
        return {
            "workflow_status": "pending", # e.g., pending, running, completed, failed
            "work_packages": {}, # Stores package_id -> {description, files, status, artifacts: {}}
            "last_error": None,
            # Add other global state info if needed
        }

    def save_state(self):
        """Saves the current orchestrator state to the state file."""
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4)
            logger.debug(f"Saved state to {self.state_file_path}")
        except IOError as e:
            logger.error(f"Failed to save state to {self.state_file_path}: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"Unexpected error saving state: {e}", exc_info=True)

    def _load_include_graph_data(self) -> dict:
        """Loads the include graph JSON if it exists."""
        if os.path.exists(self.include_graph_path):
            try:
                with open(self.include_graph_path, 'r', encoding='utf-8') as f:
                    graph = json.load(f)
                logger.info(f"Loaded include graph from {self.include_graph_path}")
                return graph
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load include graph {self.include_graph_path}: {e}. Proceeding without graph data.", exc_info=True)
                return {}
        else:
            logger.warning(f"Include graph file not found: {self.include_graph_path}. Step 1 might need to be run.")
            return {}

    def _update_package_state(self, package_id: str, status: str, artifacts: dict = None, error: str = None):
        """Helper to update the state of a specific work package."""
        if package_id not in self.state.get('work_packages', {}):
            logger.error(f"Attempted to update state for unknown package_id: {package_id}")
            return
        self.state['work_packages'][package_id]['status'] = status
        if artifacts:
            if 'artifacts' not in self.state['work_packages'][package_id]:
                 self.state['work_packages'][package_id]['artifacts'] = {}
            self.state['work_packages'][package_id]['artifacts'].update(artifacts)
        if error:
             self.state['work_packages'][package_id]['last_error'] = error
        else:
             # Clear last error on successful status update? Maybe.
             self.state['work_packages'][package_id].pop('last_error', None)
        logger.debug(f"Updated state for package '{package_id}': status='{status}', artifacts updated: {bool(artifacts)}, error set: {bool(error)}")


    # --- Step Execution Methods ---

    def execute_step1_analyze_dependencies(self):
        """Runs Step 1: C++ Include Dependency Analysis."""
        logger.info("--- Starting Step 1: Analyze Dependencies ---")
        self.state['workflow_status'] = 'running_step1'
        self.save_state()
        try:
            # Call the function from dependency_analyzer.py
            success = generate_include_graph(self.cpp_project_dir, self.include_graph_path)
            if success:
                logger.info("Step 1 completed successfully.")
                self.include_graph = self._load_include_graph_data() # Reload graph data
                self.state['workflow_status'] = 'step1_complete'
                self.state['last_error'] = None
            else:
                logger.error("Step 1 failed during graph generation.")
                self.state['workflow_status'] = 'failed_step1'
                self.state['last_error'] = "Dependency graph generation failed."
        except Exception as e:
            logger.error(f"An unexpected error occurred during Step 1: {e}", exc_info=True)
            self.state['workflow_status'] = 'failed_step1'
            self.state['last_error'] = f"Unexpected error in Step 1: {e}"
        finally:
            self.save_state()
            logger.info("--- Finished Step 1 ---")

    def execute_step2_identify_packages(self):
        """Runs Step 2: Work Package Identification."""
        logger.info("--- Starting Step 2: Identify Work Packages ---")
        if not self.include_graph:
             logger.error("Include graph data is missing. Cannot run Step 2. Run Step 1 first.")
             self.state['workflow_status'] = 'failed_step2'
             self.state['last_error'] = "Include graph data missing for Step 2."
             self.save_state()
             return

        self.state['workflow_status'] = 'running_step2'
        self.save_state()
        try:
            # Prepare context (just the graph JSON string)
            include_graph_json = json.dumps(self.include_graph, indent=2)
            # TODO: Check token count of the graph JSON? Might be large.
            # context_tokens = count_tokens(include_graph_json) # Need count_tokens accessible here
            # if context_tokens > config.MAX_CONTEXT_TOKENS * 0.8: # Example threshold
            #     logger.error("Include graph JSON is too large for LLM context.")
            #     raise ValueError("Include graph too large")

            # Instantiate Agent and Task
            agent = PackageIdentifierAgent().get_agent()
            task = IdentifyWorkPackagesTask().create_task(agent)

            # Create and run Crew
            crew = Crew(
                agents=[agent],
                tasks=[task],
                llm=self.analyzer_llm,
                process=Process.sequential,
                verbose=2
            )
            logger.info("Kicking off Crew for Step 2...")
            # Provide input directly to kickoff if task expects it via 'inputs'
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

            # Update state with packages
            self.state['work_packages'] = {} # Reset packages if re-running
            for pkg_id, pkg_data in parsed_packages.items():
                if isinstance(pkg_data, dict) and 'description' in pkg_data and 'files' in pkg_data:
                    self.state['work_packages'][pkg_id] = {
                        'description': pkg_data['description'],
                        'files': pkg_data['files'],
                        'status': 'identified', # Initial status
                        'artifacts': {}
                    }
                else:
                    logger.warning(f"Skipping invalid package data for ID '{pkg_id}' in Step 2 result.")

            logger.info(f"Step 2 identified {len(self.state['work_packages'])} work packages.")
            self.state['workflow_status'] = 'step2_complete'
            self.state['last_error'] = None

        except Exception as e:
            logger.error(f"An error occurred during Step 2: {e}", exc_info=True)
            self.state['workflow_status'] = 'failed_step2'
            self.state['last_error'] = f"Error in Step 2: {e}"
        finally:
            self.save_state()
            logger.info("--- Finished Step 2 ---")


    def execute_step3_define_structure(self, package_ids: list[str] = None):
        """Runs Step 3: Godot Structure Definition for specified or all eligible packages."""
        logger.info(f"--- Starting Step 3: Define Structure (Packages: {package_ids or 'All Eligible'}) ---")
        eligible_packages = self._get_eligible_packages(target_status='identified', specific_ids=package_ids)

        if not eligible_packages:
            logger.warning("No eligible packages found for Step 3.")
            # Update global status? Maybe not if only specific packages were requested.
            return

        self.state['workflow_status'] = 'running_step3'
        self.save_state()
        overall_success = True

        for pkg_id in eligible_packages:
            logger.info(f"Processing Step 3 for package: {pkg_id}")
            self._update_package_state(pkg_id, status='running_structure')
            self.save_state() # Save state before potentially long LLM call

            try:
                pkg_info = self.state['work_packages'][pkg_id]
                # TODO: Determine primary vs dependency files based on pkg_info['files'] and include_graph
                primary_files = pkg_info.get('files', [])
                # Placeholder for dependency logic - needs graph traversal
                dependency_files = [] # self._get_dependencies_for_package(pkg_id)

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
                    llm=self.mapper_llm,
                    process=Process.sequential,
                    verbose=2
                )
                logger.info(f"Kicking off Crew for Step 3 (Package: {pkg_id})...")
                result_md = crew.kickoff() # Expecting Markdown string
                logger.info(f"Step 3 Crew finished for package: {pkg_id}")

                if not result_md or not isinstance(result_md, str):
                     raise ValueError("Step 3 Crew did not return a valid Markdown string.")

                # Save the artifact
                artifact_filename = f"package_{pkg_id}_structure.md"
                artifact_path = os.path.join(self.analysis_dir, artifact_filename)
                try:
                    with open(artifact_path, 'w', encoding='utf-8') as f:
                        f.write(result_md)
                    logger.info(f"Saved structure definition artifact: {artifact_path}")
                    self._update_package_state(pkg_id, status='structure_defined', artifacts={'structure_md': artifact_filename})
                except IOError as e:
                    raise IOError(f"Failed to save structure artifact {artifact_path}: {e}")

            except Exception as e:
                logger.error(f"An error occurred during Step 3 for package {pkg_id}: {e}", exc_info=True)
                self._update_package_state(pkg_id, status='failed_structure', error=str(e))
                overall_success = False
            finally:
                self.save_state() # Save after each package attempt

        # Update overall status
        if overall_success and not package_ids: # Only mark globally complete if all eligible were processed successfully
             # Check if ALL packages are now 'structure_defined' or beyond
             all_done = all(p.get('status') in ['structure_defined', 'mapping_defined', 'processed'] for p in self.state['work_packages'].values())
             if all_done:
                  self.state['workflow_status'] = 'step3_complete'
        elif not overall_success:
             self.state['workflow_status'] = 'failed_step3' # Mark as failed if any package failed

        self.save_state()
        logger.info("--- Finished Step 3 ---")


    def execute_step4_define_mapping(self, package_ids: list[str] = None):
        """Runs Step 4: C++ to Godot Mapping Definition for specified or all eligible packages."""
        logger.info(f"--- Starting Step 4: Define Mapping (Packages: {package_ids or 'All Eligible'}) ---")
        eligible_packages = self._get_eligible_packages(target_status='structure_defined', specific_ids=package_ids)

        if not eligible_packages:
            logger.warning("No eligible packages found for Step 4.")
            return

        self.state['workflow_status'] = 'running_step4'
        self.save_state()
        overall_success = True

        for pkg_id in eligible_packages:
            logger.info(f"Processing Step 4 for package: {pkg_id}")
            self._update_package_state(pkg_id, status='running_mapping')
            self.save_state()

            try:
                pkg_info = self.state['work_packages'][pkg_id]
                structure_artifact = pkg_info.get('artifacts', {}).get('structure_md')
                if not structure_artifact:
                    raise FileNotFoundError(f"Structure definition artifact missing for package {pkg_id}.")

                structure_md_path = os.path.join(self.analysis_dir, structure_artifact)
                if not os.path.exists(structure_md_path):
                     raise FileNotFoundError(f"Structure definition file not found: {structure_md_path}")

                with open(structure_md_path, 'r', encoding='utf-8') as f:
                    structure_content = f.read()

                # Determine files for context
                primary_files = pkg_info.get('files', [])
                dependency_files = [] # self._get_dependencies_for_package(pkg_id)

                # Assemble context including structure markdown
                context = self.context_manager.get_context_for_step(
                    step_name="MAPPING_DEFINITION",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    work_package_id=pkg_id,
                    work_package_description=pkg_info.get('description', ''),
                    proposed_godot_structure=structure_content # Add the structure MD
                )

                if not context:
                     raise ValueError("Failed to assemble context for Step 4.")

                # Instantiate Agent and Task
                agent = MappingDefinerAgent().get_agent()
                task = DefineMappingTask().create_task(agent, context)

                # Create and run Crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=self.mapper_llm,
                    process=Process.sequential,
                    verbose=2
                )
                logger.info(f"Kicking off Crew for Step 4 (Package: {pkg_id})...")
                # Expecting combined Markdown + JSON string
                combined_output = crew.kickoff()
                logger.info(f"Step 4 Crew finished for package: {pkg_id}")

                if not combined_output or not isinstance(combined_output, str):
                     raise ValueError("Step 4 Crew did not return a valid string output.")

                # Parse the combined output
                mapping_strategy_md, task_list_json = parse_step4_output(combined_output)

                if mapping_strategy_md is None or task_list_json is None:
                    logger.error(f"Failed to parse Step 4 combined output for package {pkg_id}.")
                    logger.debug(f"Raw output: {combined_output}")
                    raise ValueError("Parsing Step 4 output failed (separator missing or JSON invalid).")

                # Save artifacts
                md_artifact_filename = f"package_{pkg_id}_mapping.md"
                json_artifact_filename = f"package_{pkg_id}_tasks.json"
                md_artifact_path = os.path.join(self.analysis_dir, md_artifact_filename)
                json_artifact_path = os.path.join(self.analysis_dir, json_artifact_filename)

                try:
                    with open(md_artifact_path, 'w', encoding='utf-8') as f:
                        f.write(mapping_strategy_md)
                    logger.info(f"Saved mapping strategy artifact: {md_artifact_path}")

                    with open(json_artifact_path, 'w', encoding='utf-8') as f:
                        json.dump(task_list_json, f, indent=2)
                    logger.info(f"Saved task list artifact: {json_artifact_path}")

                    self._update_package_state(pkg_id, status='mapping_defined', artifacts={
                        'mapping_md': md_artifact_filename,
                        'tasks_json': json_artifact_filename
                    })
                except IOError as e:
                    raise IOError(f"Failed to save Step 4 artifacts for package {pkg_id}: {e}")

            except Exception as e:
                logger.error(f"An error occurred during Step 4 for package {pkg_id}: {e}", exc_info=True)
                self._update_package_state(pkg_id, status='failed_mapping', error=str(e))
                overall_success = False
            finally:
                self.save_state()

        # Update overall status
        if overall_success and not package_ids:
             all_done = all(p.get('status') in ['mapping_defined', 'processed'] for p in self.state['work_packages'].values())
             if all_done:
                  self.state['workflow_status'] = 'step4_complete'
        elif not overall_success:
             self.state['workflow_status'] = 'failed_step4'

        self.save_state()
        logger.info("--- Finished Step 4 ---")


    def execute_step5_process_code(self, package_ids: list[str] = None):
        """Runs Step 5: Iterative Conversion & Refinement for specified or all eligible packages."""
        logger.info(f"--- Starting Step 5: Process Code (Packages: {package_ids or 'All Eligible'}) ---")
        eligible_packages = self._get_eligible_packages(target_status='mapping_defined', specific_ids=package_ids)

        if not eligible_packages:
            logger.warning("No eligible packages found for Step 5.")
            return

        self.state['workflow_status'] = 'running_step5'
        self.save_state()
        overall_success = True

        for pkg_id in eligible_packages:
            logger.info(f"Processing Step 5 for package: {pkg_id}")
            self._update_package_state(pkg_id, status='running_processing')
            self.save_state()

            try:
                pkg_info = self.state['work_packages'][pkg_id]
                tasks_artifact = pkg_info.get('artifacts', {}).get('tasks_json')
                if not tasks_artifact:
                    raise FileNotFoundError(f"Task list artifact missing for package {pkg_id}.")

                tasks_json_path = os.path.join(self.analysis_dir, tasks_artifact)
                if not os.path.exists(tasks_json_path):
                     raise FileNotFoundError(f"Task list file not found: {tasks_json_path}")

                with open(tasks_json_path, 'r', encoding='utf-8') as f:
                    task_list = json.load(f) # This is the list of tasks
                    task_list_str = json.dumps(task_list) # String version for the prompt

                # Assemble context - This is tricky. What context does the CodeProcessor need?
                # It needs C++ context relevant *to the tasks in the list*.
                # And potentially existing Godot code if modifying files.
                # This might require analyzing the task list first to gather relevant files.
                # For now, let's pass the original package files + dependencies as a starting point.
                primary_files = pkg_info.get('files', [])
                dependency_files = self._get_dependencies_for_package(pkg_id)

                # --- Identify and load existing Godot files targeted by tasks ---
                existing_godot_files_content = {}
                target_godot_files = set()
                if isinstance(task_list, list):
                    for task_item in task_list:
                        if isinstance(task_item, dict) and task_item.get('target_godot_file'):
                            target_godot_files.add(task_item['target_godot_file'])

                logger.debug(f"Step 5 identified potential target Godot files: {target_godot_files}")
                for rel_path in target_godot_files:
                    abs_path = os.path.join(self.output_dir, rel_path) # output_dir is the godot project dir
                    if os.path.exists(abs_path):
                        logger.info(f"Reading existing Godot file for context: {abs_path}")
                        try:
                            # Use the read_file_content utility from context_manager for consistency
                            content = read_file_content(abs_path)
                            if content is not None:
                                existing_godot_files_content[rel_path] = content
                            else:
                                logger.warning(f"Failed to read existing Godot file: {abs_path}")
                        except Exception as read_err:
                             logger.error(f"Error reading existing Godot file {abs_path}: {read_err}", exc_info=True)
                    # else:
                    #     logger.debug(f"Target Godot file does not exist yet: {abs_path}")
                # -----------------------------------------------------------------

                context = self.context_manager.get_context_for_step(
                    step_name="CODE_PROCESSING",
                    primary_relative_paths=primary_files,
                    dependency_relative_paths=dependency_files,
                    work_package_id=pkg_id,
                    existing_godot_files=existing_godot_files_content,
                    # Include mapping strategy MD? Maybe useful.
                    # mapping_strategy = pkg_info.get('artifacts', {}).get('mapping_md') # Load content if needed
                )

                if not context:
                     raise ValueError("Failed to assemble context for Step 5.")

                # Instantiate Agent and Task
                agent = CodeProcessorAgent().get_agent() # Assumes validator tool is available via agent setup
                task = ProcessCodeTask().create_task(agent, context, task_list_str)

                # Create and run Crew
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    llm=self.generator_llm,
                    process=Process.sequential,
                    verbose=2
                )
                logger.info(f"Kicking off Crew for Step 5 (Package: {pkg_id})...")
                # Expecting JSON list report string or dict
                report_output = crew.kickoff()
                logger.info(f"Step 5 Crew finished for package: {pkg_id}")

                # Parse the report
                if isinstance(report_output, str):
                    try:
                        task_report = json.loads(report_output)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON report from Step 5: {e}")
                        logger.debug(f"Raw output: {report_output}")
                        raise ValueError(f"Step 5 LLM output was not valid JSON: {e}")
                elif isinstance(report_output, list):
                     task_report = report_output # Assume it's already the list
                else:
                     logger.error(f"Unexpected result type from Step 5 Crew: {type(report_output)}")
                     raise TypeError("Step 5 result was not string or list.")

                if not isinstance(task_report, list):
                     raise TypeError("Step 5 parsed report is not a list.")

                # Save the raw report artifact
                report_artifact_filename = f"package_{pkg_id}_report.json"
                report_artifact_path = os.path.join(self.analysis_dir, report_artifact_filename)
                try:
                    with open(report_artifact_path, 'w', encoding='utf-8') as f:
                        json.dump(task_report, f, indent=2)
                    logger.info(f"Saved processing report artifact: {report_artifact_path}")
                    # Update state partially, final status depends on file ops
                    self._update_package_state(pkg_id, status='processing_report_generated', artifacts={'processing_report': report_artifact_filename})
                except IOError as e:
                    raise IOError(f"Failed to save processing report artifact {report_artifact_path}: {e}")

                # --- Apply File Operations ---
                logger.info(f"Applying file operations based on Step 5 report for package {pkg_id}...")
                package_success = self._apply_code_changes(task_report)

                if package_success:
                     logger.info(f"Successfully applied code changes for package {pkg_id}.")
                     self._update_package_state(pkg_id, status='processed')
                else:
                     logger.error(f"Failed to apply some code changes for package {pkg_id}. Check logs.")
                     self._update_package_state(pkg_id, status='failed_processing', error="File operations failed for some tasks.")
                     overall_success = False # Mark overall as failed if any package fails here

            except Exception as e:
                logger.error(f"An error occurred during Step 5 for package {pkg_id}: {e}", exc_info=True)
                self._update_package_state(pkg_id, status='failed_processing', error=str(e))
                overall_success = False
            finally:
                self.save_state()

        # Update overall status
        if overall_success and not package_ids:
             all_done = all(p.get('status') == 'processed' for p in self.state['work_packages'].values())
             if all_done:
                  self.state['workflow_status'] = 'step5_complete' # Or 'completed'
        elif not overall_success:
             self.state['workflow_status'] = 'failed_step5'

        self.save_state()
        logger.info("--- Finished Step 5 ---")


    def _apply_code_changes(self, task_report: list) -> bool:
        """
        Iterates through the task report from Step 5 and applies changes using
        write_to_file or replace_in_file tools (or direct file I/O for now).

        Args:
            task_report (list): The list of task result objects from the CodeProcessorAgent.

        Returns:
            bool: True if all applicable changes were attempted successfully, False otherwise.
        """
        all_succeeded = True
        # Keep track of files modified or created in this run
        modified_files = set()
        task_level_success = True # Track success at the individual task level including validation

        for task_result in task_report:
            task_id = task_result.get('task_id', 'unknown_task')
            task_status = task_result.get('status')
            validation_status = task_result.get('validation_status')
            error_message = task_result.get('error_message')
            generated_code = task_result.get('generated_code')

            # Check for task-level errors first
            if task_status != 'completed':
                logger.warning(f"Task {task_id} did not complete successfully (status: {task_status}). Error: {error_message}")
                task_level_success = False
                all_succeeded = False # Mark overall package application as failed
                continue # Skip file op if task failed

            if not generated_code:
                 logger.warning(f"Task {task_id} completed but has no generated code. Skipping file operation.")
                 # Consider if this should be a failure? Depends on task intent.
                 continue

            # Check validation status (even if task completed)
            if validation_status == 'failure':
                 logger.warning(f"Task {task_id} completed but validation failed. Errors: {task_result.get('validation_errors')}")
                 task_level_success = False # Mark task as having issues
                 # Decide if validation failure prevents file application or just logs warning
                 # For now, let's proceed with file application but mark overall as potentially problematic

            elif validation_status == 'skipped':
                 logger.info(f"Validation skipped for task {task_id}.")
            elif validation_status == 'success':
                 logger.info(f"Validation successful for task {task_id}.")

            # Proceed with file operation if task completed
            output_format = task_result.get('output_format')
            generated_code = task_result.get('generated_code')
            target_rel_path = task_result.get('target_godot_file')
            target_element = task_result.get('target_element') # Needed for replace

            if not target_rel_path:
                logger.error(f"Task {task_result.get('task_id')} completed but missing 'target_godot_file'. Cannot apply changes.")
                all_succeeded = False
                continue

            # Construct full path within the designated output directory
            target_abs_path = os.path.join(self.output_dir, target_rel_path)
            target_dir = os.path.dirname(target_abs_path)

            try:
                # Ensure target directory exists
                os.makedirs(target_dir, exist_ok=True)

                if output_format == 'FULL_FILE':
                    logger.info(f"Applying FULL_FILE change for task {task_result.get('task_id')} to {target_abs_path}")
                    # --- Using write_to_file logic (direct I/O for now) ---
                    # Check if file exists - overwrite cautiously based on plan
                    # Since output_dir == godot_project_dir, we are modifying the project directly.
                    if os.path.exists(target_abs_path):
                         logger.info(f"Overwriting existing file: {target_abs_path}")
                    else:
                         logger.info(f"Creating new file: {target_abs_path}")

                    with open(target_abs_path, 'w', encoding='utf-8') as f:
                        f.write(generated_code)
                    logger.debug(f"Successfully wrote file: {target_abs_path}")
                    modified_files.add(target_rel_path)
                    # --------------------------------------------------------

                elif output_format == 'CODE_BLOCK':
                    logger.info(f"Applying CODE_BLOCK change for task {task_id} to {target_abs_path} (element: {target_element})")
                    # --- Attempting replace_in_file logic ---
                    if not os.path.exists(target_abs_path):
                        logger.error(f"Cannot apply CODE_BLOCK change: Target file {target_abs_path} does not exist.")
                        task_level_success = False
                        all_succeeded = False
                        continue
                    if not target_element:
                         logger.error(f"Cannot apply CODE_BLOCK change for task {task_id}: 'target_element' is missing.")
                         task_level_success = False
                         all_succeeded = False
                         continue

                    try:
                        with open(target_abs_path, 'r', encoding='utf-8') as f:
                            existing_content = f.read()

                        # Basic Strategy: Assume target_element is the start of the block to replace.
                        # Find the start index. This is highly simplistic and likely needs improvement.
                        # For functions, it might be "func target_element(...):"
                        # This needs a much more robust way to define the SEARCH block.
                        # For now, let's assume target_element is a unique line or start of function.
                        search_block_start_line = target_element # Simplistic assumption
                        start_index = existing_content.find(search_block_start_line)

                        if start_index == -1:
                            logger.error(f"Could not find target_element '{target_element}' in {target_abs_path} for CODE_BLOCK replacement.")
                            task_level_success = False
                            all_succeeded = False
                            continue

                        # Problem: How to determine the END of the block to replace?
                        # This is the hardest part without AST parsing or better markers.
                        # For now, we cannot reliably construct the SEARCH block.
                        # We will log the intent but skip the actual replacement.

                        # Placeholder for constructing the actual SEARCH block
                        # search_block = existing_content[start_index:end_index] # Need to find end_index

                        logger.warning(f"CODE_BLOCK replacement: Found target element '{target_element}' but cannot reliably determine block end.")
                        logger.warning(f"Intended replacement in {target_abs_path}:")
                        logger.warning(f"SEARCH (approximate start): {search_block_start_line}")
                        logger.warning(f"REPLACE: {generated_code}")
                        logger.warning("Skipping actual modification due to implementation limitations.")
                        # Mark as failure until implemented properly?
                        # task_level_success = False # Keep task success true for now, just warn
                        # all_succeeded = False # Keep overall success true for now, just warn

                        # --- If we *could* call replace_in_file, it would look like: ---
                        # diff_block = f"<<<<<<< SEARCH\n{search_block}\n=======\n{generated_code}\n>>>>>>> REPLACE"
                        # call_replace_in_file_tool(path=target_abs_path, diff=diff_block)
                        # modified_files.add(target_rel_path)
                        # -------------------------------------------------------------

                    except Exception as e_replace:
                         logger.error(f"Error preparing CODE_BLOCK replacement for {target_abs_path}: {e_replace}", exc_info=True)
                         task_level_success = False
                         all_succeeded = False
                    # --------------------------------------------------------

                else:
                    logger.warning(f"Unknown output_format '{output_format}' for task {task_id}. Skipping file operation.")
                    task_level_success = False
                    all_succeeded = False # Treat unknown format as failure

            except IOError as e:
                 logger.error(f"IOError applying changes for task {task_id} to {target_abs_path}: {e}", exc_info=True)
                 task_level_success = False
                 all_succeeded = False
            except Exception as e:
                 logger.error(f"Unexpected error applying changes for task {task_id} to {target_abs_path}: {e}", exc_info=True)
                 task_level_success = False
                 all_succeeded = False

            # Log task-level success/failure based on checks above
            if task_level_success:
                 logger.debug(f"Task {task_id} file operation checks passed (CODE_BLOCK modification skipped).")
            else:
                 logger.error(f"Task {task_id} encountered issues during file application phase.")


        # Return overall success status for the package
        return all_succeeded


    # --- Pipeline Execution Methods ---

    def run_full_pipeline(self):
        """Runs the entire conversion pipeline sequentially."""
        logger.info("--- Starting Full Conversion Pipeline ---")
        self.execute_step1_analyze_dependencies()
        if self.state['workflow_status'] == 'failed_step1': return

        self.execute_step2_identify_packages()
        if self.state['workflow_status'] == 'failed_step2': return

        self.execute_step3_define_structure()
        if self.state['workflow_status'] == 'failed_step3': return

        self.execute_step4_define_mapping()
        if self.state['workflow_status'] == 'failed_step4': return

        self.execute_step5_process_code()
        # Final status set within step 5

        logger.info(f"--- Full Conversion Pipeline Finished (Status: {self.state['workflow_status']}) ---")


    def resume_pipeline(self):
        """Resumes the pipeline based on the last saved state."""
        logger.info("--- Resuming Conversion Pipeline ---")
        last_status = self.state.get('workflow_status', 'pending')
        logger.info(f"Resuming from status: {last_status}")

        if last_status in ['pending', 'failed_step1']:
            logger.info("Resuming: Running Step 1...")
            self.execute_step1_analyze_dependencies()
            if self.state['workflow_status'] == 'failed_step1': return

        if self.state['workflow_status'] in ['step1_complete', 'failed_step2']:
             logger.info("Resuming: Running Step 2...")
             self.execute_step2_identify_packages()
             if self.state['workflow_status'] == 'failed_step2': return

        if self.state['workflow_status'] in ['step2_complete', 'failed_step3']:
             logger.info("Resuming: Running Step 3 (for eligible packages)...")
             # Run for packages still in 'identified' state
             eligible_ids = [pkg_id for pkg_id, pkg_data in self.state['work_packages'].items() if pkg_data.get('status') == 'identified']
             self.execute_step3_define_structure(package_ids=eligible_ids)
             # Don't exit immediately on failure here, as some packages might succeed

        if self.state['workflow_status'] in ['step3_complete', 'failed_step4', 'running_step3', 'failed_structure']: # Also resume if step 3 failed for some
             logger.info("Resuming: Running Step 4 (for eligible packages)...")
             eligible_ids = [pkg_id for pkg_id, pkg_data in self.state['work_packages'].items() if pkg_data.get('status') == 'structure_defined']
             self.execute_step4_define_mapping(package_ids=eligible_ids)

        if self.state['workflow_status'] in ['step4_complete', 'failed_step5', 'running_step4', 'failed_mapping']: # Also resume if step 4 failed for some
             logger.info("Resuming: Running Step 5 (for eligible packages)...")
             eligible_ids = [pkg_id for pkg_id, pkg_data in self.state['work_packages'].items() if pkg_data.get('status') == 'mapping_defined']
             self.execute_step5_process_code(package_ids=eligible_ids)

        # Determine final overall status after resume attempt
        final_status = self._determine_final_status()
        self.state['workflow_status'] = final_status
        self.save_state()
        logger.info(f"--- Resume Attempt Finished (Final Status: {self.state['workflow_status']}) ---")

    def _get_eligible_packages(self, target_status: str, specific_ids: list[str] = None) -> list[str]:
        """Returns a list of package IDs matching the target status, optionally filtered by specific IDs."""
        eligible = []
        packages = self.state.get('work_packages', {})
        if not packages:
            logger.warning(f"No work packages found in state to check eligibility for status '{target_status}'.")
            return []

        for pkg_id, pkg_data in packages.items():
            if specific_ids and pkg_id not in specific_ids:
                continue # Skip if specific IDs are given and this isn't one of them

            current_status = pkg_data.get('status')
            if current_status == target_status:
                eligible.append(pkg_id)
            elif specific_ids and current_status != target_status:
                 # If specific IDs were requested, warn if they aren't in the right state
                 logger.warning(f"Requested package '{pkg_id}' is not in the required status '{target_status}' (current: '{current_status}'). Skipping.")

        logger.debug(f"Found {len(eligible)} packages eligible for status '{target_status}' (Specific IDs requested: {specific_ids})")
        return eligible

    def _determine_final_status(self) -> str:
        """Checks the status of all packages to determine the overall workflow status."""
        packages = self.state.get('work_packages', {})
        if not packages:
            # If no packages, status depends on whether previous steps completed
            if self.state.get('workflow_status') == 'step2_complete': return 'completed_no_packages'
            elif self.state.get('workflow_status') == 'step1_complete': return 'step1_complete' # Stuck after step 1
            else: return self.state.get('workflow_status', 'pending') # Keep previous status

        all_processed = True
        any_failed = False
        for pkg_data in packages.values():
            status = pkg_data.get('status')
            if status != 'processed':
                all_processed = False
            if status and 'failed' in status:
                any_failed = True

        if all_processed:
            return 'completed'
        elif any_failed:
            # Be more specific about where it failed if possible
            if any('failed_processing' in p.get('status', '') for p in packages.values()): return 'failed_step5'
            if any('failed_mapping' in p.get('status', '') for p in packages.values()): return 'failed_step4'
            if any('failed_structure' in p.get('status', '') for p in packages.values()): return 'failed_step3'
            return 'failed' # Generic failure
        else:
            # Not all processed, none failed - means it's incomplete
            # Return the status of the earliest non-completed step
            if any(p.get('status') == 'identified' for p in packages.values()): return 'step2_complete' # Ready for step 3
            if any(p.get('status') == 'structure_defined' for p in packages.values()): return 'step3_complete' # Ready for step 4
            if any(p.get('status') == 'mapping_defined' for p in packages.values()): return 'step4_complete' # Ready for step 5
            return 'incomplete' # Default if in intermediate running states

    # --- Helper Methods (Placeholder) ---
    def _get_dependencies_for_package(self, package_id: str) -> list[str]:
        """
        Analyzes the include graph to find dependencies for a given package.
         Placeholder - Requires graph traversal logic.

         Args:
             package_id (str): The ID of the work package.

         Returns:
             list[str]: A list of unique relative file paths that are dependencies
                        of the files in the package.
         """
        if not self.include_graph:
            logger.warning("Include graph not loaded. Cannot calculate dependencies.")
            return []
        if package_id not in self.state.get('work_packages', {}):
            logger.error(f"Package ID '{package_id}' not found in state for dependency calculation.")
            return []

        package_files = set(self.state['work_packages'][package_id].get('files', []))
        if not package_files:
            logger.warning(f"Package '{package_id}' has no files listed. Cannot calculate dependencies.")
            return []

        logger.debug(f"Calculating dependencies for package '{package_id}' with {len(package_files)} files...")

        all_dependencies = set()
        queue = list(package_files) # Start traversal from package files
        processed = set(package_files) # Keep track of files already visited
        max_depth = 5 # Limit recursion depth to avoid excessive traversal/cycles
        depth_map = {file: 0 for file in package_files}

        while queue:
            current_file = queue.pop(0)
            current_depth = depth_map.get(current_file, 0)

            if current_depth >= max_depth:
                logger.debug(f"Reached max depth ({max_depth}) at file {current_file}. Stopping traversal for this branch.")
                continue

            # Get direct includes from the graph for the current file
            direct_includes = self.include_graph.get(current_file, [])

            for include_path in direct_includes:
                # Ensure the include is actually in the graph keys (represents a project file)
                # and hasn't been processed yet in this traversal
                if include_path in self.include_graph and include_path not in processed:
                    # Add to dependencies if it's not part of the original package
                    if include_path not in package_files:
                        all_dependencies.add(include_path)

                    # Add to queue for further traversal
                    processed.add(include_path)
                    queue.append(include_path)
                    depth_map[include_path] = current_depth + 1

        dependency_list = sorted(list(all_dependencies))
        logger.info(f"Found {len(dependency_list)} unique dependencies for package '{package_id}' (max_depth={max_depth}).")
        logger.debug(f"Dependencies for '{package_id}': {dependency_list[:10]}...") # Log first few
        return dependency_list
