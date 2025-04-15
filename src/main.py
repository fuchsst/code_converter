# main.cli.py
import fire
import os
import sys
from logger_setup import setup_logging, get_logger
import config # Load configuration defaults
from src.core.orchestrator import Orchestrator

# Setup logging at the very beginning
setup_logging() # Configure root logger
logger = get_logger(__name__) # Get logger for this module

class ConversionCLI:
    """
    AI-Powered C++ to Godot Conversion Tool CLI using Python Fire.
    """

    def _get_orchestrator(self, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None) -> Orchestrator:
        """Helper method to instantiate the orchestrator with provided or default args."""
        # Use provided args or fall back to config defaults
        cpp_dir = cpp_dir or config.CPP_PROJECT_DIR
        godot_dir = godot_dir or config.GODOT_PROJECT_DIR
        analysis_dir = analysis_dir or config.ANALYSIS_OUTPUT_DIR
        target_language = target_language or config.TARGET_LANGUAGE

        logger.debug(f"Instantiating Orchestrator with:")
        logger.debug(f"  cpp_dir: {cpp_dir}")
        logger.debug(f"  godot_dir: {godot_dir}")
        logger.debug(f"  analysis_dir: {analysis_dir}")
        logger.debug(f"  target_language: {target_language}")

        try:
            return Orchestrator(
                cpp_project_dir=cpp_dir,
                godot_project_dir=godot_dir,
                analysis_dir=analysis_dir,
                target_language=target_language
            )
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            sys.exit(1)

    def _handle_result(self, orchestrator: Orchestrator, command_name: str):
        """Checks orchestrator status after command execution."""
        logger.info(f"Command '{command_name}' finished.")
        final_status = orchestrator.state_manager.get_state().get('workflow_status', 'unknown')
        if 'failed' in final_status:
             logger.error(f"Workflow ended with status: {final_status}")
             # fire doesn't have explicit exit codes like argparse handlers,
             # but we can raise an exception or print error and let script exit non-zero
             # For simplicity, just log the error. Consider raising SystemExit(1) if needed.
             # sys.exit(1)
        else:
             logger.info(f"Workflow ended with status: {final_status}")


    def analyze_deps(self, cpp_dir=None, analysis_dir=None):
        """
        Step 1: Run C++ include dependency analysis using Clang.

        Args:
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
        """
        command_name = "analyze-deps"
        logger.info(f"Executing {command_name}...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, analysis_dir=analysis_dir)
        try:
            orchestrator.run_step("step1")
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            # orchestrator.save_state() # Consider saving state on error
            sys.exit(1) # Exit with error code

    def identify_packages(self, cpp_dir=None, analysis_dir=None):
        """
        Step 2: Identify logical work packages from the dependency graph.

        Args:
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
        """
        command_name = "identify-packages"
        logger.info(f"Executing {command_name}...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, analysis_dir=analysis_dir)
        try:
            orchestrator.run_step("step2")
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)

    def define_structure(self, package_id: str | tuple[str] = None, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None):
        """
        Step 3: Propose Godot structure for specific or all eligible packages.

        Args:
            package_id (str | tuple[str], optional): Specify one or more package IDs to process.
                                                     If omitted, processes all eligible packages.
                                                     Use --package-id=ID1 --package-id=ID2 ...
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            godot_dir (str, optional): Path to the Godot project directory. Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
            target_language (str, optional): Target language (e.g., GDScript). Defaults to config.
        """
        command_name = "define-structure"
        # Convert single string package_id to list if needed
        package_ids_list = list(package_id) if isinstance(package_id, tuple) else ([package_id] if package_id else None)
        logger.info(f"Executing {command_name} (Packages: {package_ids_list or 'All Eligible'})...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, godot_dir=godot_dir, analysis_dir=analysis_dir, target_language=target_language)
        try:
            orchestrator.run_step("step3", package_ids=package_ids_list)
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)

    def define_mapping(self, package_id: str | tuple[str] = None, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None):
        """
        Step 4: Define C++ to Godot mapping for specific or all eligible packages.

        Args:
            package_id (str | tuple[str], optional): Specify one or more package IDs to process.
                                                     Use --package-id=ID1 --package-id=ID2 ...
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            godot_dir (str, optional): Path to the Godot project directory. Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
            target_language (str, optional): Target language (e.g., GDScript). Defaults to config.
        """
        command_name = "define-mapping"
        package_ids_list = list(package_id) if isinstance(package_id, tuple) else ([package_id] if package_id else None)
        logger.info(f"Executing {command_name} (Packages: {package_ids_list or 'All Eligible'})...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, godot_dir=godot_dir, analysis_dir=analysis_dir, target_language=target_language)
        try:
            orchestrator.run_step("step4", package_ids=package_ids_list)
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)

    def process_code(self, package_id: str | tuple[str] = None, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None):
        """
        Step 5: Generate/modify Godot code based on mapping for specific or all eligible packages.

        Args:
            package_id (str | tuple[str], optional): Specify one or more package IDs to process.
                                                     Use --package-id=ID1 --package-id=ID2 ...
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            godot_dir (str, optional): Path to the Godot project directory (used for context and output). Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
            target_language (str, optional): Target language (e.g., GDScript). Defaults to config.
        """
        command_name = "process-code"
        package_ids_list = list(package_id) if isinstance(package_id, tuple) else ([package_id] if package_id else None)
        logger.info(f"Executing {command_name} (Packages: {package_ids_list or 'All Eligible'})...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, godot_dir=godot_dir, analysis_dir=analysis_dir, target_language=target_language)
        try:
            orchestrator.run_step("step5", package_ids=package_ids_list)
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)

    def run_all(self, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None):
        """
        Run the full conversion pipeline sequentially (Steps 1-5).

        Args:
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            godot_dir (str, optional): Path to the Godot project directory (used for context and output). Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
            target_language (str, optional): Target language (e.g., GDScript). Defaults to config.
        """
        command_name = "run-all"
        logger.info(f"Executing {command_name}...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, godot_dir=godot_dir, analysis_dir=analysis_dir, target_language=target_language)
        try:
            orchestrator.run_full_pipeline()
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)

    def resume(self, cpp_dir=None, godot_dir=None, analysis_dir=None, target_language=None):
        """
        Attempt to resume the pipeline from the last saved state.

        Args:
            cpp_dir (str, optional): Path to the C++ project directory. Defaults to config.
            godot_dir (str, optional): Path to the Godot project directory (used for context and output). Defaults to config.
            analysis_dir (str, optional): Directory for analysis output. Defaults to config.
            target_language (str, optional): Target language (e.g., GDScript). Defaults to config.
        """
        command_name = "resume"
        logger.info(f"Executing {command_name}...")
        orchestrator = self._get_orchestrator(cpp_dir=cpp_dir, godot_dir=godot_dir, analysis_dir=analysis_dir, target_language=target_language)
        try:
            orchestrator.resume_pipeline()
            self._handle_result(orchestrator, command_name)
        except Exception as e:
            logger.error(f"An error occurred executing command '{command_name}': {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting C++ to Godot Conversion Tool CLI (using Fire)...")
    fire.Fire(ConversionCLI)
    logger.info("CLI finished.")
