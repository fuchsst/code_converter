# src/core/executors/step2_package_identifier.py
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import networkx as nx
from crewai import Crew, Process

from src.llms.litellm_gemini_llm import LiteLLMGeminiLLM

from ..step_executor import StepExecutor
from ..state_manager import StateManager
from ..context_manager import ContextManager, count_tokens, read_file_content
from src.logger_setup import get_logger
import src.config as global_config # Use alias for clarity

from src.utils.clustering_utils import cluster_files_by_dependency
from src.agents.step2.package_describer import get_package_describer_agent
from src.tasks.step2.describe_package import create_describe_package_task, PackageDescriptionOutput
from src.agents.step2.package_refiner import get_package_refinement_agent
from src.tasks.step2.refine_descriptions import create_refined_descriptions_task, RefinedDescriptionsOutput
from crewai import LLM as CrewAI_LLM
from src.utils.json_utils import parse_json_from_string

logger = get_logger(__name__)

# --- Constants from Config ---
# These might be overridden by self.config in methods
LLM_DESC_MAX_TOKENS_RATIO = global_config.LLM_DESC_MAX_TOKENS_RATIO
LLM_PROMPT_BUFFER = global_config.PROMPT_TOKEN_BUFFER
LLM_CALL_RETRIES = global_config.LLM_CALL_RETRIES

class Step2Executor(StepExecutor):
    """
    Executes Step 2: Work Package Identification & Description.
    Uses Louvain for initial partitioning, refines for balance,
    and uses an LLM agent for description generation.
    Handles persistence via ContextManager.
    """
    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]],
                 tools: Dict[Type, Any]):
        super().__init__(state_manager, context_manager, config, llm_configs, tools)
        # Store cpp_source_dir as an absolute Path object
        # Ensure config is accessed via self.config hereafter
        self.cpp_source_dir = Path(self.config.get("CPP_PROJECT_DIR", "data/cpp_project")).resolve()

    # --- Graph Building (Remains mostly the same) ---
    def _build_dependency_graph(self, include_graph_data: Dict[str, List[Dict[str, Any]]]) -> nx.DiGraph:
        """Builds a NetworkX DiGraph from the include dependency data."""
        G = nx.DiGraph()
        normalized_graph_data = {}
        all_files_in_graph = set()

        for k, v_list in include_graph_data.items():
            norm_k = k.replace('\\', '/')
            all_files_in_graph.add(norm_k)
            normalized_includes = []
            if isinstance(v_list, list):
                for item in v_list:
                    if isinstance(item, dict) and 'path' in item:
                        norm_path = item['path'].replace('\\', '/')
                        # Ensure weight is numeric, default to 1.0
                        weight = item.get("weight", 1.0)
                        try:
                            weight = float(weight)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid weight '{weight}' for include {norm_path} in {norm_k}. Defaulting to 1.0.")
                            weight = 1.0
                        normalized_includes.append({
                            "path": norm_path,
                            "weight": weight
                        })
                        all_files_in_graph.add(norm_path)
                    else:
                        logger.warning(f"Invalid include item format for {norm_k}: {item}")
            normalized_graph_data[norm_k] = normalized_includes

        # Add all nodes first
        for file_path in all_files_in_graph:
            G.add_node(file_path)

        # Add edges based on includes
        for file_path, includes_with_weights in normalized_graph_data.items():
            if file_path not in G: continue
            for include_item in includes_with_weights:
                included_file = include_item["path"]
                weight = include_item["weight"]
                if included_file in G:
                    G.add_edge(file_path, included_file, weight=weight)

        logger.info(f"Built dependency DiGraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    # --- Node Weight Calculation ---
    def _get_node_weights(self, graph_nodes: List[str]) -> Dict[str, int]:
        """Calculates token counts for all nodes in the graph."""
        file_token_counts: Dict[str, int] = {}
        logger.info("Pre-calculating token counts for all graph nodes...")
        for node in graph_nodes:
            abs_path_str = str(self.cpp_source_dir / node)
            # Use cleaned content for token counting, consistent with LLM context
            content = read_file_content(abs_path_str, remove_comments_blank_lines=True)
            file_token_counts[node] = count_tokens(content) if content else 0
        logger.info("Finished pre-calculating token counts.")
        return file_token_counts

    # _assemble_description_context is removed. ContextManager.get_work_package_source_code_content will be used directly.

    # --- Main Execution Logic ---
    def execute(self, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        """
        Runs Step 2: Identifies packages using Louvain, refines for balance,
        and generates descriptions using a CrewAI agent.

        Args:
            package_ids (Optional[List[str]]): Not typically used in Step 2 as it partitions all code.
            force (bool): If True, forces re-partitioning and description even if step2_complete.
            **kwargs: Additional arguments (unused).
        """
        logger.info(f"--- Starting Step 2 Execution: Package Identification & Description (Force={force}) ---")

        # --- Force Handling ---
        # If force=True, we essentially ignore existing package data and re-run partitioning.
        # We also need to reset the workflow status if it was already 'step2_complete'.
        current_workflow_status = self.state_manager.get_state().get('workflow_status')
        if force:
            logger.warning("Force flag is set. Re-running Step 2 partitioning and description generation.")
            # Reset relevant state? Clear existing packages?
            self.context_manager.save_packages_data({}, None) # Clear saved packages.json
            self.state_manager.set_packages({}) # Clear packages in state
            self.state_manager.set_package_processing_order(None) # Clear order in state
            if current_workflow_status == 'step2_complete':
                 self.state_manager.update_workflow_status('running_step2_forced') # Indicate forced run
            else:
                 self.state_manager.update_workflow_status('running_step2')
            resuming_from_file = False # Ensure we don't resume
            current_packages_data = {} # Start fresh
        else:
            # Normal execution: check if resuming is possible
            self.state_manager.update_workflow_status('running_step2')
            # Check StateManager for existing packages to determine if resuming
            current_packages_data = self.state_manager.get_all_packages() # Get packages from state
            if current_packages_data:
                logger.info(f"Resuming package description using data from StateManager ({len(current_packages_data)} packages found).")
                resuming_from_file = True
            else:
                logger.info("No valid existing package data found in StateManager. Performing full partitioning.")
                resuming_from_file = False

        success = False
        node_weights = {} # Initialize node_weights

        # Calculate node weights if resuming (needed for total_tokens display/potential future use)
        if resuming_from_file:
            all_files_in_loaded_packages = set()
            for pkg_data in current_packages_data.values():
                if isinstance(pkg_data.get("files"), list):
                    all_files_in_loaded_packages.update(pkg_data["files"])
            all_files_in_loaded_packages = set()
            for pkg_data in current_packages_data.values():
                if isinstance(pkg_data.get("files"), list):
                    all_files_in_loaded_packages.update(pkg_data["files"])
            if not all_files_in_loaded_packages:
                 logger.warning("Resuming: Loaded packages data contains no files. Cannot calculate node weights.")
            else:
                 logger.info("Resuming: Calculating node weights for files in loaded packages...")
                 node_weights = self._get_node_weights(list(all_files_in_loaded_packages))

        # --- Graph Loading ---
        # Graph is needed for partitioning (if not resuming) and potentially for description context/ordering
        include_graph_data = self.context_manager.include_graph
        if not include_graph_data and not resuming_from_file: # Only fail if not resuming and graph is needed for partitioning
            logger.error("Include graph data is missing and not resuming. Cannot run partitioning in Step 2.")
            self.state_manager.update_workflow_status('failed_step2', "Include graph data missing for partitioning.")
            return False

        try:
            if not resuming_from_file:
                # --- Steps 1-3: Graph, Weights, Clustering (New Logic) ---
                logger.info("Performing graph building, weight calculation, and dependency clustering...")
                # 1. Build Graph
                dependency_graph = self._build_dependency_graph(include_graph_data)
                if not dependency_graph.nodes:
                    logger.warning("Dependency graph has no nodes after building. Skipping package identification.")
                    self.context_manager.save_packages_data({}) # Save empty via ContextManager
                    self.state_manager.set_packages({"packages": {}})
                    self.state_manager.update_workflow_status('step2_complete')
                    return True

                # 2. Calculate Node Weights (Token Counts)
                node_weights = self._get_node_weights(list(dependency_graph.nodes()))

                # 3. Cluster using the new dependency-based algorithm
                max_package_tokens = self.config.get('MAX_PACKAGE_SIZE_TOKENS', global_config.MAX_PACKAGE_SIZE_TOKENS)
                final_partitions = cluster_files_by_dependency(
                    dependency_graph, node_weights, max_package_tokens
                )

                if not final_partitions:
                    logger.error("Dependency-based clustering yielded no valid packages.")
                    self.context_manager.save_packages_data({}) # Save empty via ContextManager
                    self.state_manager.set_packages({"packages": {}})
                    self.state_manager.update_workflow_status('step2_complete')
                    return True

                # --- Prepare initial package structure for saving and processing ---
                logger.info("Preparing initial package structure from final partitions.")
                current_packages_data = {} # Reset here, as we are creating it fresh
                for i, files in final_partitions.items():
                    package_name = f"package_{i+1}"
                    package_total_tokens = sum(node_weights.get(f, 0) for f in files)
                    current_packages_data[package_name] = {
                        "description": None, # Initialize description as None
                        "file_roles": [], # Initialize roles as empty list
                        "files": files,
                        "total_tokens": package_total_tokens
                    }
                # --- Save initial structure before starting descriptions ---
                logger.info("Saving initial package structure via ContextManager...")
                self.context_manager.save_packages_data(current_packages_data)

            # --- Step 5: Get LLM Descriptions (runs whether resuming or not) ---
            final_packages_output = {} # This will accumulate results as we go

            # --- START: LLM Call Section ---
            analyzer_llm_instance = self._create_llm_instance(llm_role='ANALYZER_MODEL', response_schema_class=PackageDescriptionOutput)

            if not analyzer_llm_instance:
                logger.warning("Proceeding without LLM-generated package descriptions.")
                # Fill in placeholders for any missing descriptions
                needs_save = False
                for pkg_name, pkg_data in current_packages_data.items():
                    # Use the config constant name in the placeholder message
                    invalid_descs = {None, "", "Error: Failed to generate description", f"N/A (LLM '{global_config.ANALYZER_MODEL}' not configured)", f"N/A (LLM '{global_config.ANALYZER_MODEL}' not available)"}
                    if pkg_data.get("description") in invalid_descs:
                         pkg_data["description"] = f"N/A (LLM '{global_config.ANALYZER_MODEL}' not available)"
                         pkg_data["file_roles"] = [{"file_path": f, "role": "N/A"} for f in pkg_data.get("files", [])]
                         needs_save = True
                # Use the potentially updated data as the final output in this case
                final_packages_output = current_packages_data
                # Save the state with placeholders if changes were made
                if needs_save:
                     logger.info("Saving package data with LLM N/A placeholders via ContextManager...")
                     self.context_manager.save_packages_data(final_packages_output)

            else: # LLM instance IS available
                 logger.info("LLM instance available. Proceeding with description generation...")
                 # Setup agent
                 describer_agent = get_package_describer_agent(llm_instance=analyzer_llm_instance)
                 max_retries = self.config.get("LLM_CALL_RETRIES", LLM_CALL_RETRIES)

                 # Determine token limit for description context
                 llm_max_tokens = self.config.get("MAX_CONTEXT_TOKENS", global_config.MAX_CONTEXT_TOKENS)
                 if llm_max_tokens is None:
                     llm_max_tokens = 200_000 # Hardcoded fallback
                     logger.warning(f"config.MAX_CONTEXT_TOKENS was None, using hardcoded default: {llm_max_tokens}")
                 desc_token_limit = int(llm_max_tokens * LLM_DESC_MAX_TOKENS_RATIO) - LLM_PROMPT_BUFFER

                 # --- Iterate through packages in the current data (loaded or newly created) ---
                 total_packages = len(current_packages_data)
                 processed_count = 0
                 # Iterate over a copy of items to allow modification of the original dict
                 packages_to_process = list(current_packages_data.items())

                 for package_name, package_data in packages_to_process:
                     processed_count += 1
                     files = package_data.get("files", [])
                     if not files:
                          logger.warning(f"Skipping package {package_name} as it has no files listed.")
                          # Ensure it's included in final output even if skipped
                          if package_name not in final_packages_output:
                              final_packages_output[package_name] = package_data
                          continue

                     logger.info(f"Processing package {processed_count}/{total_packages}: {package_name} ({len(files)} files)...")

                     # --- Check if description already exists ---
                     existing_desc = package_data.get("description")
                     # Use the config constant name in the placeholder message check
                     invalid_descs = {None, "", "Error: Failed to generate description", f"N/A (LLM '{global_config.ANALYZER_MODEL}' not configured)", f"N/A (LLM '{global_config.ANALYZER_MODEL}' not available)"}
                     if existing_desc not in invalid_descs:
                          logger.info(f"Description already exists for {package_name}. Skipping LLM call.")
                          # Ensure it's included in final output
                          if package_name not in final_packages_output:
                              final_packages_output[package_name] = package_data
                          continue # Move to the next package

                     # --- Description needed, proceed with LLM call ---
                     logger.info(f"Generating description for {package_name}...")

                     # Retrieve source code context using the new ContextManager method
                     source_code_context = self.context_manager.get_work_package_source_code_content(
                         package_id=package_name, # Assuming package_name is the ID used by ContextManager
                         max_tokens=desc_token_limit
                     )
                     if not source_code_context:
                         logger.warning(f"Could not retrieve source code context for {package_name}. Task might lack information.")
                         # Provide minimal context indicating failure
                         source_code_context = f"// Error: Failed to retrieve source code content for package {package_name}."

                     logger.debug(f"Using source code context retrieved via ContextManager for {package_name} description.")

                     # Fetch general instructions
                     instruction_context = self.context_manager.get_instruction_context()

                     # Create the task using the imported function
                     describe_task = create_describe_package_task(
                         agent=describer_agent,
                         package_files=files,
                         context=source_code_context,
                         instructions=instruction_context
                     )

                     # Create and run the Crew, passing the specific LLM instance
                     logger.debug(f"DEBUG: Analyzer LLM Instance Type: {type(analyzer_llm_instance)}")
                     logger.debug(f"DEBUG: Analyzer LLM Instance Model: {getattr(analyzer_llm_instance, 'model', 'N/A')}")

                     crew = Crew(
                         agents=[describer_agent],
                         tasks=[describe_task],
                         process=Process.sequential,
                         llm=analyzer_llm_instance,
                         verbose=1 # Or configure via self.config
                     )

                     details_json = None
                     raw_output_string = None
                     for attempt in range(max_retries + 1):
                         try:
                             logger.debug(f"Attempt {attempt + 1}/{max_retries + 1} to kickoff description crew for {package_name}")
                             result = crew.kickoff()

                             # --- START: Modified Result Handling ---
                             if hasattr(result, 'raw') and isinstance(result.raw, str):
                                 raw_output_string = result.raw
                                 logger.debug(f"Extracted raw string from CrewOutput.raw for {package_name}")
                             elif isinstance(result, str):
                                 raw_output_string = result
                                 logger.debug(f"Crew kickoff returned a string directly for {package_name}")
                             else:
                                 logger.warning(f"Crew kickoff for {package_name} returned unexpected result type on attempt {attempt + 1}. Type: {type(result)}, Value: {result}")
                                 raw_output_string = None
                             # --- END: Modified Result Handling ---

                             if raw_output_string:
                                 # --- Use the utility function ---
                                 details_json = parse_json_from_string(raw_output_string)
                                 # --- End use the utility function ---

                                 if details_json:
                                     # Basic validation (can stay here or move to utility if needed)
                                     if 'package_description' in details_json and 'file_roles' in details_json and isinstance(details_json.get('file_roles'), list):
                                          logger.debug(f"Successfully parsed and validated LLM details for {package_name} on attempt {attempt + 1}")
                                          break # Success
                                     else:
                                          logger.warning(f"Parsed JSON structure invalid for {package_name} on attempt {attempt + 1}. Parsed JSON: {details_json}")
                                          details_json = None # Reset on invalid structure
                             else:
                                 details_json = None

                         except Exception as e:
                             logger.error(f"Crew kickoff failed for {package_name} on attempt {attempt + 1}: {e}", exc_info=True)
                             details_json = None

                         # If failed and retries remain, wait
                         if details_json is None and attempt < max_retries:
                             sleep_time = 1.5 ** attempt # Exponential backoff
                             logger.info(f"Waiting {sleep_time:.2f}s before retry for {package_name} description...")
                             time.sleep(sleep_time)
                         elif details_json is not None:
                              break # Success

                     # Process results after retries and update the main dictionary
                     if details_json:
                         # Optional: Validate file list consistency
                         llm_files = set(item.get('file_path') for item in details_json.get('file_roles', []) if isinstance(item, dict))
                         input_files_set = set(files)
                         if llm_files != input_files_set:
                              logger.warning(f"File list mismatch between input ({len(input_files_set)}) and LLM file_roles ({len(llm_files)}) for {package_name}. Using input file list for 'files' key.")

                         # Update the entry in current_packages_data
                         current_packages_data[package_name] = {
                             "description": details_json.get("package_description", "Error: Description missing"),
                             "file_roles": details_json.get("file_roles", []), # Use LLM roles
                             "files": files, # Use the balanced partition file list
                             "total_tokens": sum(node_weights.get(f, 0) for f in files)
                         }
                     else:
                         logger.error(f"Failed to get LLM description for {package_name} after {max_retries + 1} attempts. Marking as error.")
                         # Update the entry in current_packages_data with error
                         current_packages_data[package_name] = {
                             "description": "Error: Failed to generate description",
                             "file_roles": [{"file_path": f, "role": "Error"} for f in files],
                             "files": files,
                             "total_tokens": sum(node_weights.get(f, 0) for f in files)
                         }
                     # <<< End of if/else block for processing details_json

                     # --- Save intermediate state via ContextManager AFTER processing each package ---
                     logger.debug(f"Saving intermediate package state via ContextManager after processing {package_name}...")
                     self.context_manager.save_packages_data(current_packages_data)

                     # --- Accumulate the result for the final state manager update ---
                     # This ensures final_packages_output reflects the latest saved state from the file
                     final_packages_output[package_name] = current_packages_data[package_name]
                 # <<< End of the 'for package_name, package_data in packages_to_process:' loop

                 # --- START: Refinement Step ---
                 # Only run refinement if the initial description LLM was available and we have packages
                 if final_packages_output:
                     logger.info("--- Starting Package Description Refinement Step ---")
                     try:
                         # --- Instantiate Refiner LLM ---
                         # Use the helper method again, potentially with a different schema
                         # User edit specified GENERATOR_REFINER_MODEL role
                         refiner_llm_instance = self._create_llm_instance(
                             llm_role='GENERATOR_REFINER_MODEL',
                             response_schema_class=RefinedDescriptionsOutput
                         )

                         # Proceed only if refiner LLM was successfully created
                         if not refiner_llm_instance:
                              logger.error("Skipping refinement step because the refiner LLM instance could not be created.")
                         else:
                              # --- Refinement Crew Setup ---

                              # Prepare context - format the accumulated package data as a JSON string
                              try:
                                  # Import json if not already imported at the top
                                  import json
                                  refinement_context_str = json.dumps(final_packages_output, indent=2)
                                  logger.debug(f"Prepared refinement context string ({len(refinement_context_str)} chars).")
                              except TypeError as e:
                                  logger.error(f"Failed to serialize final_packages_output to JSON for refinement context: {e}")
                                  refinement_context_str = "// Error: Could not format package data for refinement."

                              # The RefineDescriptionsTask sets output_json=RefinedDescriptionsOutput for validation/fallback
                              refiner_agent = get_package_refinement_agent(llm_instance=refiner_llm_instance)
                              # Fetch general instructions again for the refinement task
                              instruction_context = self.context_manager.get_instruction_context()
                              refine_task = create_refined_descriptions_task(
                                  agent=refiner_agent,
                                  all_packages_data=final_packages_output, # Pass the dict directly
                                  instructions=instruction_context
                              )

                              refinement_crew = Crew(
                                  agents=[refiner_agent],
                                  tasks=[refine_task],
                                  process=Process.sequential,
                                  llm=refiner_llm_instance,
                                  verbose=True
                              )

                              logger.info("Kicking off refinement crew...")
                              refinement_result = refinement_crew.kickoff()
                              raw_refinement_output = None

                              if hasattr(refinement_result, 'raw') and isinstance(refinement_result.raw, str):
                                  raw_refinement_output = refinement_result.raw
                              elif isinstance(refinement_result, str):
                                  raw_refinement_output = refinement_result
                              else:
                                  logger.warning(f"Refinement crew returned unexpected result type: {type(refinement_result)}")

                              if raw_refinement_output:
                                  parsed_refinement_json = parse_json_from_string(raw_refinement_output)

                                  if isinstance(parsed_refinement_json, dict) and 'package_descriptions' in parsed_refinement_json:
                                      refined_list = parsed_refinement_json['package_descriptions']
                                      if isinstance(refined_list, list):
                                          logger.info(f"Successfully parsed refinement results ({len(refined_list)} entries). Applying updates...")
                                          update_count = 0
                                          for item in refined_list:
                                              if isinstance(item, dict) and 'package_id' in item and 'package_description' in item:
                                                  pkg_id = item['package_id']
                                                  refined_desc = item['package_description']

                                                  if pkg_id in final_packages_output and isinstance(refined_desc, str):
                                                      if final_packages_output[pkg_id]['description'] != refined_desc:
                                                          logger.debug(f"Updating description for {pkg_id}")
                                                          final_packages_output[pkg_id]['description'] = refined_desc
                                                          update_count += 1
                                                      else:
                                                          logger.debug(f"Refined description for {pkg_id} is the same as initial. Skipping update.")
                                                  else:
                                                      logger.warning(f"Skipping refinement update for '{pkg_id}': Package not found in original data or refined description is not a string.")
                                              else:
                                                  logger.warning(f"Skipping invalid item in refined_descriptions list: {item}")

                                          logger.info(f"Applied {update_count} refined descriptions.")
                                          # Save the refined data immediately
                                          logger.info("Saving updated package data with refined descriptions via ContextManager...")
                                          self.context_manager.save_packages_data(final_packages_output)
                                      else:
                                          logger.error("Parsed refinement JSON has 'package_descriptions' but it's not a list.")
                                  else:
                                      logger.error("Failed to parse refinement result dictionary or 'package_descriptions' key missing from LLM output.")
                              else:
                                  logger.error("Refinement crew did not return a usable string output.")
                              # --- End Refinement Crew Execution ---

                     except Exception as e:
                         logger.error(f"An error occurred during the refinement step: {e}", exc_info=True)
                         # Continue without refinement if it fails, error is logged
                 elif not final_packages_output:
                      logger.info("Skipping refinement step as there are no packages to refine.")
                 # --- END: Refinement Step ---

            # 6. Calculate Package Processing Order
            processing_order = None
            if final_packages_output and self.context_manager.include_graph:
                 try:
                      package_dep_graph = self._build_package_dependency_graph(
                          final_packages_output,
                          self.context_manager.include_graph
                      )
                      processing_order = self._calculate_package_order(package_dep_graph)
                      if processing_order is None:
                           logger.error("Failed to calculate package processing order due to detected cycle.")
                           # Decide how to handle cycles - fail step? Continue without order?
                           # For now, log error and continue without order.
                 except Exception as order_err:
                      logger.error(f"Error calculating package order: {order_err}", exc_info=True)
                      processing_order = None # Ensure order is None on error
            elif not self.context_manager.include_graph:
                 logger.warning("Skipping package order calculation as include graph is not available.")
            else: # No packages
                 logger.info("Skipping package order calculation as no packages were identified/processed.")


            # 7. Save Final State (Packages + Order) and Update State Manager
            # Save the final package data (potentially refined) along with the calculated order
            logger.info("Saving final package data and processing order via ContextManager...")
            self.context_manager.save_packages_data(final_packages_output, processing_order)

            # Update state manager with packages and order separately
            logger.info("Updating State Manager with final package data...")
            self.state_manager.set_packages(final_packages_output) # Use the method that handles structure
            if processing_order is not None:
                 logger.info("Updating State Manager with package processing order...")
                 self.state_manager.set_package_processing_order(processing_order)
            else:
                 logger.warning("No processing order calculated or cycle detected. State Manager order not updated.")


            package_count = len(final_packages_output)
            order_status = f"with processing order ({len(processing_order)} steps)" if processing_order else "without a processing order (cycle detected or error)"
            logger.info(f"Step 2 identified, described, and ordered {package_count} final work packages {order_status}.")
            self.state_manager.update_workflow_status('step2_complete')
            success = True

        except Exception as e:
            logger.error(f"An error occurred during Step 2 execution: {e}", exc_info=True)
            self.state_manager.update_workflow_status('failed_step2', f"Error in Step 2: {e}")
            success = False
        finally:
            logger.info("--- Finished Step 2 Execution ---")
            return success

    # --- Helper Functions for Package Ordering ---

    def _build_package_dependency_graph(self,
                                        packages_data: Dict[str, Dict[str, Any]],
                                        file_include_graph: Dict[str, List[Dict[str, Any]]]) -> nx.DiGraph:
        """
        Builds a dependency graph between packages based on file includes.
        An edge from P_A to P_B means P_A depends on P_B.
        """
        logger.info("Building package dependency graph...")
        pkg_graph = nx.DiGraph()
        file_to_package_map: Dict[str, str] = {}

        # 1. Create file -> package mapping
        for pkg_name, pkg_info in packages_data.items():
            pkg_graph.add_node(pkg_name) # Add node for each package
            files = pkg_info.get("files", [])
            if isinstance(files, list):
                for file_path in files:
                    normalized_file = file_path.replace('\\', '/')
                    if normalized_file in file_to_package_map:
                        # This shouldn't happen with non-overlapping partitions, but log if it does
                        logger.warning(f"File '{normalized_file}' found in multiple packages: '{file_to_package_map[normalized_file]}' and '{pkg_name}'. Using first encountered.")
                    else:
                        file_to_package_map[normalized_file] = pkg_name
            else:
                 logger.warning(f"Package '{pkg_name}' has invalid 'files' data: {files}")


        # 2. Iterate through file dependencies to create package dependencies
        edges_added = 0
        if not file_include_graph:
             logger.warning("File include graph is empty. Cannot build package dependency graph.")
             return pkg_graph

        for source_file, includes in file_include_graph.items():
            normalized_source = source_file.replace('\\', '/')
            source_pkg = file_to_package_map.get(normalized_source)
            if not source_pkg:
                # logger.debug(f"Source file '{normalized_source}' not found in any package. Skipping its dependencies.")
                continue # Skip files not assigned to a package

            if isinstance(includes, list):
                for include_item in includes:
                     if isinstance(include_item, dict) and 'path' in include_item:
                          target_file = include_item['path'].replace('\\', '/')
                          target_pkg = file_to_package_map.get(target_file)

                          if target_pkg and source_pkg != target_pkg:
                              # Add edge from source_pkg to target_pkg (source depends on target)
                              if not pkg_graph.has_edge(source_pkg, target_pkg):
                                   pkg_graph.add_edge(source_pkg, target_pkg)
                                   edges_added += 1
                                   # logger.debug(f"Added package dependency edge: {source_pkg} -> {target_pkg} (due to {normalized_source} -> {target_file})")
                     # else: logger.warning(f"Invalid include item format for {source_file}: {include_item}") # Can be verbose
            # else: logger.warning(f"Invalid includes format for {source_file}: {includes}") # Can be verbose


        logger.info(f"Built package dependency graph with {pkg_graph.number_of_nodes()} nodes and {pkg_graph.number_of_edges()} edges.")
        return pkg_graph

    def _calculate_package_order(self, pkg_graph: nx.DiGraph) -> Optional[List[str]]:
        """
        Calculates a processing order for packages prioritizing high importance (PageRank)
        and low number of dependencies (out-degree). Handles cycles.

        Args:
            pkg_graph: The package dependency graph (DiGraph) where an edge P_A -> P_B
                       means P_A depends on P_B.

        Returns:
            A list of package names in the calculated processing order,
            or a fallback order (e.g., alphabetical) if an error occurs.
        """
        logger.info("Calculating package processing order using combined PageRank and Out-Degree heuristic...")
        if not pkg_graph or not pkg_graph.nodes():
            logger.warning("Package graph is empty or has no nodes. Cannot calculate order.")
            return []

        try:
            # Calculate PageRank
            pagerank_scores = nx.pagerank(pkg_graph)

            # Calculate combined score for each package
            package_scores = {}
            for pkg in pkg_graph.nodes():
                out_degree = pkg_graph.out_degree(pkg)
                # Ensure pagerank score exists, default to 0 if not (shouldn't happen)
                pr_score = pagerank_scores.get(pkg, 0.0)
                # Score: Higher PageRank is better, lower Out-Degree is better
                # Add 1.0 to out_degree to avoid division by zero and ensure float division
                package_scores[pkg] = pr_score / (float(out_degree) + 1.0)

            # Sort packages by the combined score in descending order
            ordered_packages = sorted(package_scores, key=package_scores.get, reverse=True)

            logger.info(f"Successfully calculated package processing order using combined heuristic: {ordered_packages}")
            # Log scores for debugging if needed
            # for pkg in ordered_packages:
            #     logger.debug(f"  - {pkg}: Score={package_scores[pkg]:.4f} (PR={pagerank_scores.get(pkg, 0.0):.4f}, OutDegree={pkg_graph.out_degree(pkg)})")

            return ordered_packages

        except Exception as e:
            logger.error(f"Error calculating combined heuristic score for package order: {e}", exc_info=True)
            logger.warning("Falling back to alphabetical package order due to calculation error.")
            return sorted(list(pkg_graph.nodes()))
