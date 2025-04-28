# src/core/state_manager.py
import os
import json
from pathlib import Path # Use Pathlib
from src.logger_setup import get_logger
from typing import Dict, Any, List, Optional, Union # Added Union, Path

logger = get_logger(__name__)

DEFAULT_STATE_FILE = "orchestrator_state.json"

class StateManager:
    """Handles loading, saving, and updating the workflow state."""

    def __init__(self, analysis_dir: str, state_filename: str = DEFAULT_STATE_FILE):
        """
        Initializes the StateManager.

        Args:
            analysis_dir (str): The directory where the state file is stored.
            state_filename (str): The name of the state file.
        """
        self.analysis_dir = os.path.abspath(analysis_dir)
        # Use Path objects internally
        self.analysis_dir = Path(analysis_dir).resolve()
        self.state_file_path = self.analysis_dir / state_filename
        self.state: Dict[str, Any] = self._load_state()
        logger.info(f"StateManager initialized. State file: {self.state_file_path}. Analysis dir: {self.analysis_dir}")

    def _load_state(self) -> Dict[str, Any]:
        """Loads the orchestrator state from the state file using Pathlib."""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"Loaded existing state from {self.state_file_path}")
                # Basic validation/migration could happen here if state format changes
                if 'work_packages' not in state: state['work_packages'] = {}
                if 'workflow_status' not in state: state['workflow_status'] = 'pending'
                return state
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load state file {self.state_file_path}: {e}. Initializing new state.", exc_info=True)
                return self._initialize_state()
        else:
            logger.info("No existing state file found. Initializing new state.")
            return self._initialize_state()

    def _initialize_state(self) -> Dict[str, Any]:
        """Returns a dictionary representing the initial state."""
        return {
            "workflow_status": "pending", # e.g., pending, running, completed, failed
            "work_packages": {}, # Stores package_id -> {description, files, status, artifacts: {}, remapping_attempts: 0, last_error: None}
            "package_processing_order": None, # Stores the calculated order list
            "last_error": None,
            # Add other global state info if needed
        }

    def save_state(self):
        """Saves the current orchestrator state to the state file using Pathlib."""
        try:
            # Ensure analysis directory exists before saving
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4)
            logger.debug(f"Saved state to {self.state_file_path}")
        except IOError as e:
            logger.error(f"Failed to save state to {self.state_file_path}: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"Unexpected error saving state: {e}", exc_info=True)

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state dictionary."""
        return self.state

    def update_workflow_status(self, status: str, error: str = None):
        """Updates the global workflow status and optionally the last error."""
        self.state['workflow_status'] = status
        self.state['last_error'] = error
        logger.debug(f"Workflow status updated to: {status}, Error: {error}")
        self.save_state()

    def update_package_state(self, package_id: str, status: str, artifacts: Dict[str, Any] = None, error: str = None, increment_remap_attempt: bool = False):
        """
        Updates the state of a specific work package.

        Args:
            package_id (str): The ID of the work package.
            status (str): The new status for the package.
            artifacts (Dict[str, Any], optional): Dictionary of artifacts to add/update. Defaults to None.
            error (str, optional): Error message to store. Defaults to None. Clears existing error if None.
            increment_remap_attempt (bool): If True, increments the remapping attempt counter.
        """
        if package_id not in self.state.get('work_packages', {}):
            # If package doesn't exist, maybe initialize it? Or log error?
            # Let's initialize it minimally if it's missing.
            logger.warning(f"Package ID '{package_id}' not found in state. Initializing entry.")
            self.state.setdefault('work_packages', {})[package_id] = {
                'description': 'N/A - Added during update',
                'files': [],
                'status': 'unknown',
                'artifacts': {},
                'remapping_attempts': 0,
                'last_error': None
            }

        package_data = self.state['work_packages'][package_id]
        package_data['status'] = status

        if artifacts:
            package_data.setdefault('artifacts', {}).update(artifacts)

        if error is not None:
            package_data['last_error'] = error
        else:
            # Clear last error if error argument is None
            package_data.pop('last_error', None)

        if increment_remap_attempt:
             current_attempts = package_data.get('remapping_attempts', 0)
             package_data['remapping_attempts'] = current_attempts + 1
             logger.debug(f"Incremented remapping attempts for package '{package_id}' to {package_data['remapping_attempts']}")

        logger.debug(f"Updated state for package '{package_id}': status='{status}', artifacts updated: {bool(artifacts)}, error set: {error is not None}, remap incremented: {increment_remap_attempt}")
        self.save_state()

    def get_package_info(self, package_id: str) -> Dict[str, Any] | None:
        """Retrieves the state information for a specific package."""
        return self.state.get('work_packages', {}).get(package_id)

    def get_all_packages(self) -> Dict[str, Any]:
        """Retrieves the state information for all packages."""
        return self.state.get('work_packages', {})

    def set_packages(self, packages_data: Dict[str, Any]):
        """
        Sets the entire work_packages dictionary in the state.
        Used after Step 2 identifies packages. Ensures structure consistency.
        """
        validated_packages = {}
        for pkg_id, pkg_data in packages_data.items():
            if isinstance(pkg_data, dict) and 'description' in pkg_data and 'files' in pkg_data:
                validated_packages[pkg_id] = {
                    'description': pkg_data.get('description', 'N/A'),
                    'files': pkg_data.get('files', []),
                    'status': 'identified', # Initial status after identification
                    'artifacts': pkg_data.get('artifacts', {}), # Carry over any initial artifacts? Unlikely for step 2.
                    'remapping_attempts': pkg_data.get('remapping_attempts', 0),
                    'last_error': pkg_data.get('last_error', None),
                    'total_tokens': pkg_data.get('total_tokens', 0) # Ensure total_tokens is preserved
                }
            else:
                logger.warning(f"Skipping invalid package data structure for ID '{pkg_id}' during set_packages.")

        self.state['work_packages'] = validated_packages
        logger.info(f"Set {len(validated_packages)} work packages in state.")
        # Reset order when packages are set, it needs recalculation
        if 'package_processing_order' in self.state:
             self.state['package_processing_order'] = None
             logger.info("Reset package processing order as work packages were updated.")
        self.save_state()

    def set_package_processing_order(self, order: Optional[List[str]]):
        """Sets the calculated package processing order in the state."""
        if order is None or isinstance(order, list):
            self.state['package_processing_order'] = order
            logger.info(f"Set package processing order in state ({len(order) if order else 'None'} packages).")
            self.save_state()
        else:
            logger.error(f"Attempted to set invalid package processing order (type: {type(order)}). State not updated.")

    def get_package_processing_order(self) -> Optional[List[str]]:
        """Retrieves the calculated package processing order from the state."""
        return self.state.get('package_processing_order')

    # --- Artifact Management Methods ---

    def save_artifact(self, artifact_filename: str, content: Union[str, Dict, List], is_json: bool = True) -> bool:
        """
        Saves content to an artifact file within the analysis directory.

        Args:
            artifact_filename (str): The filename (relative to analysis_dir).
            content (Union[str, Dict, List]): The content to save.
            is_json (bool): If True and content is dict/list, serialize as JSON. Defaults to True.

        Returns:
            bool: True on success, False on failure.
        """
        artifact_path = self.analysis_dir / artifact_filename
        logger.debug(f"Attempting to save artifact to: {artifact_path}")
        try:
            # Ensure analysis directory exists
            self.analysis_dir.mkdir(parents=True, exist_ok=True)

            with open(artifact_path, 'w', encoding='utf-8') as f:
                if is_json and isinstance(content, (dict, list)):
                    json.dump(content, f, indent=4)
                    logger.debug(f"Saved JSON artifact: {artifact_filename}")
                elif isinstance(content, str):
                    f.write(content)
                    logger.debug(f"Saved text artifact: {artifact_filename}")
                else:
                    logger.error(f"Invalid content type for artifact '{artifact_filename}'. Expected str, dict, or list, got {type(content)}.")
                    return False
            return True
        except (IOError, TypeError) as e: # Separate IO/Type errors from the suspicious JSONDecodeError
            logger.error(f"Failed to save artifact '{artifact_filename}' due to {type(e).__name__}: {e}", exc_info=True)
            return False
        except json.JSONDecodeError as e: # Catch JSONDecodeError separately
            # This is highly unusual during save. Log carefully.
            logger.error(f"Caught unexpected JSONDecodeError during save operation for '{artifact_filename}'. This might indicate a deeper issue.")
            # Log the type and args if possible, before logging the full exception
            try:
                 logger.error(f"Exception type: {type(e)}, Args: {e.args}")
            except Exception as log_err:
                 logger.error(f"Could not log exception details: {log_err}")
            logger.error(f"Full exception info:", exc_info=True) # Log full traceback
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving artifact '{artifact_filename}' of type {type(e).__name__}: {e}", exc_info=True)
            return False

    def load_artifact(self, artifact_filename: str, expect_json: bool = True) -> Optional[Union[str, Dict, List]]:
        """
        Loads content from an artifact file within the analysis directory.

        Args:
            artifact_filename (str): The filename (relative to analysis_dir).
            expect_json (bool): If True, attempt to parse the content as JSON. Defaults to True.

        Returns:
            Optional[Union[str, Dict, List]]: The loaded content (string or parsed JSON),
                                              or None if the file doesn't exist or loading/parsing fails.
        """
        artifact_path = self.analysis_dir / artifact_filename
        if not artifact_path.exists():
            logger.debug(f"Artifact file not found: {artifact_path}")
            return None

        logger.debug(f"Attempting to load artifact from: {artifact_path}")
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if expect_json:
                try:
                    parsed_json = json.loads(content)
                    logger.debug(f"Loaded and parsed JSON artifact: {artifact_filename}")
                    return parsed_json
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse artifact '{artifact_filename}' as JSON: {json_err}")
                    # Optionally return the raw string content if JSON parsing fails but reading succeeded?
                    # For now, return None to indicate failure to get expected type.
                    return None
            else:
                logger.debug(f"Loaded text artifact: {artifact_filename}")
                return content # Return raw string content

        except IOError as e:
            logger.error(f"Failed to read artifact file '{artifact_filename}' from {artifact_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading artifact '{artifact_filename}': {e}", exc_info=True)
            return None
