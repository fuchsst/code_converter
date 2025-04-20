# src/core/step_executor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from .state_manager import StateManager
from .context_manager import ContextManager
from .tool_interfaces import IFileWriter, IFileReplacer, IFileReader, ISyntaxValidator
from src.logger_setup import get_logger

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
    def execute(self, package_ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Executes the specific workflow step.

        Args:
            package_ids (Optional[List[str]]): Specific package IDs to process (if applicable).
                                                If None, process all eligible packages for the step.
            **kwargs: Additional arguments specific to the step.

        Returns:
            bool: True if the step execution was successful overall, False otherwise.
        """
        pass

    def _get_eligible_packages(self, target_status: str, specific_ids: Optional[List[str]] = None) -> List[str]:
        """
        Helper method to find eligible packages based on their status.
        """
        eligible = []
        packages = self.state_manager.get_all_packages()
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

    def _get_llm_config(self, llm_role: str) -> Optional[Dict[str, Any]]:
        """Helper to get a specific LLM configuration dictionary from the injected map."""
        llm_config = self.llm_configs.get(llm_role)
        if not llm_config:
            logger.error(f"LLM configuration for role '{llm_role}' not found in provided llm_configs map.")
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
