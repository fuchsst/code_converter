# src/core/state_manager.py
import os
import json
from src.logger_setup import get_logger
from typing import Dict, Any

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
        self.state_file_path = os.path.join(self.analysis_dir, state_filename)
        self.state: Dict[str, Any] = self._load_state()
        logger.info(f"StateManager initialized. State file: {self.state_file_path}")

    def _load_state(self) -> Dict[str, Any]:
        """Loads the orchestrator state from the state file."""
        if os.path.exists(self.state_file_path):
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
            "last_error": None,
            # Add other global state info if needed
        }

    def save_state(self):
        """Saves the current orchestrator state to the state file."""
        try:
            # Ensure analysis directory exists before saving
            os.makedirs(self.analysis_dir, exist_ok=True)
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
                    'last_error': pkg_data.get('last_error', None)
                }
            else:
                logger.warning(f"Skipping invalid package data structure for ID '{pkg_id}' during set_packages.")

        self.state['work_packages'] = validated_packages
        logger.info(f"Set {len(validated_packages)} work packages in state.")
        self.save_state()
