# src/core/step_executor.py
# src/core/step_executor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

# Local application imports
from .state_manager import StateManager
from .context_manager import ContextManager
from .tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator
from src.logger_setup import get_logger
import src.config as config # Import config for GEMINI_TIMEOUT


try:
    from crewai import LLM as CrewAI_LLM
except ImportError:
    CrewAI_LLM = None # Define as None if import fails


logger = get_logger(__name__)

class StepExecutor(ABC):
    """Abstract base class for workflow step executors."""

    def __init__(self,
                 state_manager: StateManager,
                 context_manager: ContextManager,
                 config: Dict[str, Any],
                 llm_configs: Dict[str, Dict[str, Any]], # Map role to LLM config dict
                 tools: Dict[Type, Any]): # Map interface type to tool instance
        """
        Initializes the StepExecutor.

        Args:
            state_manager (StateManager): Manages workflow state.
            context_manager (ContextManager): Manages context assembly.
            config (Dict[str, Any]): Application configuration dictionary.
            llm_configs (Dict[str, Dict[str, Any]]): Dictionary mapping role names
                                                     (e.g., 'analyzer', 'mapper')
                                                     to their configuration dictionaries for LiteLLM.
            tools (Dict[Type, Any]): Dictionary mapping tool interface types (e.g., IFileWriter)
                                     to their concrete implementations.
        """
        self.state_manager = state_manager
        self.context_manager = context_manager
        self.config = config
        self.llm_configs = llm_configs # Store the config map
        self.tools = tools
        logger.debug(f"Initialized {self.__class__.__name__} with LLM Configs: {list(llm_configs.keys())}, Tools: {[t.__name__ for t in tools.keys()]}")

    @abstractmethod
    def execute(self, package_ids: Optional[List[str]] = None, force: bool = False, **kwargs) -> bool:
        """
        Executes the specific workflow step.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process (if applicable).
                                                If None, process all eligible packages for the step.
            force (bool): If True, attempts to force reprocessing of packages even if they are in a completed or failed state for this step. Defaults to False.
            **kwargs: Additional arguments specific to the step.

        Returns:
            bool: True if the step execution was successful overall, False otherwise.
        """
        pass

    def _get_eligible_packages(self, target_status: str, specific_ids: Optional[List[str]] = None, force: bool = False) -> List[str]:
        """
        Helper method to find eligible packages based on their status.
        """
        eligible = []
        packages = self.state_manager.get_all_packages()
        if not packages:
            logger.warning(f"No work packages found in state to check eligibility for status '{target_status}'.")
            return []

        # Define the set of statuses considered "terminal" for a step, which might be overridden by 'force'
        # This might need adjustment based on the specific step's logic
        terminal_statuses = {'processed', 'completed'} # Add step-specific completed statuses like 'structure_defined', 'mapping_defined' etc. in subclasses if needed
        failed_statuses = {status for status in self.state_manager.get_all_packages().values() if status and 'failed' in status} # Dynamically get all failed statuses

        for pkg_id, pkg_data in packages.items():
            if specific_ids and pkg_id not in specific_ids:
                continue # Skip if specific IDs are given and this isn't one of them

            current_status = pkg_data.get('status')

            is_target = (current_status == target_status)
            is_failed = (current_status in failed_statuses)
            # Add more checks here if steps have multiple valid starting statuses or terminal statuses

            if is_target:
                 eligible.append(pkg_id)
                 logger.debug(f"Package '{pkg_id}' is eligible (status: '{current_status}').")
            elif force and is_failed:
                 # If force is True and the package failed this step previously, consider it eligible for retry
                 logger.info(f"Forcing eligibility for previously failed package '{pkg_id}' (status: '{current_status}').")
                 # Optionally reset the status here or let the step executor handle it
                 # self.state_manager.update_package_state(pkg_id, target_status, error=None) # Example reset
                 eligible.append(pkg_id)
            elif specific_ids and not is_target:
                 # If specific IDs were requested, warn if they aren't in the right state (and not forced)
                 logger.warning(f"Requested package '{pkg_id}' is not in the required status '{target_status}' (current: '{current_status}') and force=False. Skipping.")
            # else: # Package is neither target status nor forced failed status
                 # logger.debug(f"Package '{pkg_id}' is not eligible (status: '{current_status}', target: '{target_status}', force: {force}).")


        logger.debug(f"Found {len(eligible)} packages eligible for status '{target_status}' (Specific IDs requested: {specific_ids}, Force: {force})")
        return eligible

    def _get_llm_config(self, llm_role: str) -> Optional[Dict[str, Any]]:
        """Helper to get a specific LLM configuration dictionary from the injected map."""
        llm_config = self.llm_configs.get(llm_role)
        if not llm_config:
            logger.error(f"LLM configuration for role '{llm_role}' not found in provided llm_configs map.")
            # Return None if config not found
            return None
        # Return the config dictionary if found
        return llm_config

    def _get_tool(self, tool_interface: Type) -> Optional[Any]:
        """Helper to get a tool instance based on its interface type."""
        tool_instance = self.tools.get(tool_interface)
        if not tool_instance:
             logger.error(f"Tool implementing interface '{tool_interface.__name__}' not found in provided tools map.")
        # Further check if the found instance actually implements the interface?
        # isinstance check might be redundant if the map keys are types, but good practice.
        elif not isinstance(tool_instance, tool_interface):
             logger.error(f"Found tool for '{tool_interface.__name__}' but it does not implement the interface.")
             return None
        return tool_instance

    # --- LLM Instantiation Helper ---
    def _get_llm_instance(self,
                          llm_config: Optional[Dict],
                          config_key_name: str,
                          response_schema_class: Optional[Type] = None) -> Optional[Any]:
         """
         Helper to instantiate an LLM object given its configuration dictionary.
         Optionally applies a Pydantic response schema for Gemini models.

         Args:
             llm_config (Optional[Dict]): The configuration dictionary for the LLM.
             config_key_name (str): The name of the config key (e.g., 'MANAGER_MODEL') for logging.
             response_schema_class (Optional[Type]): The Pydantic class to use for response schema validation (if applicable).

         Returns:
             Optional[Any]: An instantiated LLM object or None.
         """
         if not llm_config or not isinstance(llm_config, dict):
              logger.error(f"Invalid or missing LLM config dictionary provided for '{config_key_name}'. Type: {type(llm_config)}")
              return None

         # Now, instantiate based on the provided llm_config dictionary
         try:
             model_identifier = llm_config.get("model", "[Unknown Model]")
             logger.debug(f"Attempting to instantiate LLM for '{config_key_name}' using model '{model_identifier}' with config: {llm_config}")

             llm_instance = CrewAI_LLM(**llm_config)
             logger.info(f"Successfully instantiated default crewai.LLM for '{config_key_name}': {model_identifier}")
             return llm_instance # Return the instantiated object

         except ImportError as ie: # Should be caught by initial checks, but keep for safety
              logger.error(f"Import error during LLM instantiation for '{config_key_name}': {ie}. Check dependencies.")
              return None
         except TypeError as te:
              logger.error(f"Type error during LLM instantiation for '{config_key_name}': {te}. Check config parameters for model '{model_identifier}'. Config: {llm_config}")
              return None
         except Exception as e:
             logger.error(f"Unexpected error during LLM instantiation for '{config_key_name}' with model '{model_identifier}': {e}", exc_info=True)
             return None

    def _create_llm_instance(self,
                             llm_role: str,
                             response_schema_class: Optional[Type] = None) -> Optional[Any]:
        """
        Utility function to get config and instantiate LLM in one step.

        Args:
            llm_role (str): The role name (key) for the LLM configuration.
            response_schema_class (Optional[Type]): Pydantic class for response schema (if needed).

        Returns:
            Optional[Any]: An instantiated LLM object or None.
        """
        logger.debug(f"Creating LLM instance for role: '{llm_role}'...")
        llm_config = self._get_llm_config(llm_role)
        if not llm_config:
            # Error already logged by _get_llm_config
            return None
        # Call the existing instantiation method with the fetched config
        return self._get_llm_instance(llm_config, llm_role, response_schema_class)
