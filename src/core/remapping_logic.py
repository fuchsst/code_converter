# src/core/remapping_logic.py
from typing import List, Dict, Any
from logger_setup import get_logger
import config # Import config to potentially use thresholds

logger = get_logger(__name__)

class RemappingLogic:
    """Encapsulates logic for deciding when to remap and generating feedback."""

    @staticmethod
    def should_remap_package(failed_tasks: List[Dict[str, Any]]) -> bool:
        """
        Analyzes failed tasks to determine if they suggest mapping issues that would benefit from remapping.

        Args:
            failed_tasks (List[Dict[str, Any]]): List of task result dictionaries that failed.

        Returns:
            bool: True if the pattern of failures suggests mapping issues, False otherwise.
        """
        if not failed_tasks:
            return False

        # Count different types of failures
        search_block_missing_count = 0
        search_block_not_found_count = 0
        agent_validation_failure_count = 0 # Validation reported by agent (if any)
        orchestrator_validation_failure_count = 0 # Validation run by orchestrator/executor
        file_op_failure_count = 0
        other_failure_count = 0

        for task in failed_tasks:
            error_msg = (task.get('error_message', '') or '').lower()
            file_op_msg = (task.get('file_operation_message', '') or '').lower()
            # Check both agent's validation and orchestrator's validation status
            agent_validation_status = task.get('validation_status')
            orch_validation_status = task.get('orchestrator_validation_status') # Assuming Step5Executor adds this key

            # Categorize failures based on agent report and orchestrator validation
            if 'search_block is missing' in error_msg or 'search_block was missing' in error_msg:
                search_block_missing_count += 1
            elif 'search_block not found' in file_op_msg or 'search_block not found' in error_msg:
                search_block_not_found_count += 1
            elif agent_validation_status == 'failure':
                 # Count agent validation failures only if not masked by a search block issue
                 if 'search_block' not in error_msg and 'search_block' not in file_op_msg:
                      agent_validation_failure_count += 1
            elif orch_validation_status == 'failure':
                 # Count orchestrator validation failures if not masked by other issues
                 if 'search_block' not in error_msg and 'search_block' not in file_op_msg and agent_validation_status != 'failure':
                      orchestrator_validation_failure_count += 1
            elif task.get('file_operation_status') == 'failure':
                 # Count file op failures if not search block related
                 if 'search_block' not in error_msg and 'search_block' not in file_op_msg:
                      file_op_failure_count += 1
            elif task.get('status') == 'failed': # Catch-all for other agent-reported failures
                 other_failure_count += 1

        total_failures = len(failed_tasks)
        search_related_failures = search_block_missing_count + search_block_not_found_count
        any_validation_failures = agent_validation_failure_count + orchestrator_validation_failure_count

        logger.debug(f"Remap check: Total Failures={total_failures}, Search Missing={search_block_missing_count}, Search Not Found={search_block_not_found_count}, Agent Validation Fails={agent_validation_failure_count}, Orch Validation Fails={orchestrator_validation_failure_count}, File Op Fails={file_op_failure_count}, Other={other_failure_count}")

        # --- Decision Logic ---
        # Thresholds could potentially come from config
        search_fail_threshold_abs = 2
        search_fail_threshold_rel = 0.3
        validation_fail_threshold_abs = 3
        validation_fail_threshold_rel = 0.5

        # 1. High number of search block related issues strongly suggests mapping problems
        if search_related_failures >= search_fail_threshold_abs or \
           (total_failures > 0 and search_related_failures / total_failures >= search_fail_threshold_rel):
            logger.info(f"Remapping triggered: High search block failure rate ({search_related_failures}/{total_failures}).")
            return True

        # 2. High number of *any* validation failures might indicate mapping issues
        if any_validation_failures >= validation_fail_threshold_abs or \
           (total_failures > 5 and any_validation_failures / total_failures >= validation_fail_threshold_rel):
             logger.info(f"Remapping triggered: High validation failure rate ({any_validation_failures}/{total_failures}).")
             return True

        # 3. If most errors are generic file operation errors or others, less likely a mapping issue.
        if file_op_failure_count + other_failure_count > search_related_failures + any_validation_failures:
             logger.info(f"Remapping not triggered: Failures seem less related to mapping/validation logic (FileOp/Other: {file_op_failure_count+other_failure_count}, Search/Validation: {search_related_failures+any_validation_failures}).")
             return False

        # Default: Don't remap if no strong signal
        logger.debug("Remapping not triggered: No strong failure pattern detected.")
        return False

    @staticmethod
    def generate_mapping_feedback(failed_tasks: List[Dict[str, Any]]) -> str:
        """
        Generates structured feedback about failed tasks for the mapping agent.

        Args:
            failed_tasks (List[Dict[str, Any]]): List of task result dictionaries that failed.
        Returns:
            str: Formatted feedback string.
        """
        feedback_parts = ["## Feedback from Previous Code Processing Attempt\n"]
        feedback_parts.append("The following issues were encountered during the previous attempt to apply the generated code based on the mapping. Please refine the mapping strategy and task list to address these issues:\n")

        # Group failures by type for clarity
        search_missing = []
        search_not_found = []
        validation_failed = [] # Includes agent internal and orchestrator post-op
        file_op_failed = []
        other_failed = []

        for task in failed_tasks:
            error_msg = (task.get('error_message', '') or '').lower()
            file_op_msg = (task.get('file_operation_message', '') or '').lower()
            agent_validation_status = task.get('validation_status')
            orch_validation_status = task.get('orchestrator_validation_status') # Key added by Step5Executor

            # Prioritize critical failures
            if 'search_block is missing' in error_msg or 'search_block was missing' in error_msg:
                search_missing.append(task)
            elif 'search_block not found' in file_op_msg or 'search_block not found' in error_msg:
                search_not_found.append(task)
            elif agent_validation_status == 'failure' or orch_validation_status == 'failure':
                 validation_failed.append(task)
            elif task.get('file_operation_status') == 'failure':
                 file_op_failed.append(task)
            elif task.get('status') == 'failed': # Catch-all for other failures
                 other_failed.append(task)

        def format_task_details(task):
            task_id = task.get('task_id', 'unknown')
            target_file = task.get('target_godot_file', 'unknown')
            target_element = task.get('target_element', 'unknown')
            error = task.get('error_message', 'No specific error')
            file_op_error = task.get('file_operation_message', '')
            # Include validation errors from both sources if available
            agent_val_err = task.get('validation_errors', '') # Agent's internal validation (if any)
            orch_val_err = task.get('orchestrator_validation_errors', '') # Orchestrator's validation

            details = [f"- **Task ID:** `{task_id}` (Target: `{target_file}`::{target_element})"]
            if agent_val_err: details.append(f"  - **Agent Validation Error:** {agent_val_err}")
            if orch_val_err: details.append(f"  - **Orchestrator Validation Error:** {orch_val_err}")
            if file_op_error and 'search_block' not in file_op_error.lower(): # Show file op error if not search related
                 details.append(f"  - **File Operation Message:** {file_op_error}")
            if error: details.append(f"  - **Reported Error:** {error}") # General error message
            return "\n".join(details)

        if search_missing:
            feedback_parts.append("\n### Issues: Missing `search_block` for Modifications")
            feedback_parts.append("The agent failed to provide the required `search_block` when its report indicated a 'CODE_BLOCK' modification. This often means the mapping task was unclear about modifying existing code vs. creating new code, or the agent failed to extract the original block from context.")
            feedback_parts.append("**Recommendation:** Ensure mapping tasks clearly specify if code should be replaced. If replacing, ensure the target element exists in the proposed structure and the agent is instructed to extract the `search_block` from the provided context.")
            for task in search_missing: feedback_parts.append(format_task_details(task))

        if search_not_found:
            feedback_parts.append("\n### Issues: `search_block` Not Found in Target File")
            feedback_parts.append("The `search_block` provided by the agent for the `Replace Content In File` tool could not be found in the target file by the orchestrator. This indicates the agent generated an incorrect `search_block` (doesn't match the actual file content) or the mapping task targeted a non-existent element for modification.")
            feedback_parts.append("**Recommendation:** Review the mapping for these tasks. Ensure the C++ elements correctly map to *existing* elements in the proposed Godot structure if modification is intended. Verify the agent's context includes the correct target file content.")
            for task in search_not_found: feedback_parts.append(format_task_details(task))

        if validation_failed:
            feedback_parts.append("\n### Issues: Generated Code Failed Syntax Validation")
            feedback_parts.append(f"The code generated for these tasks was syntactically incorrect according to the Godot {config.TARGET_LANGUAGE} validator. This might stem from incorrect API usage, typos, or misunderstanding Godot syntax based on the mapping notes.")
            feedback_parts.append(f"**Recommendation:** Review the mapping notes for these tasks. Ensure they guide the agent towards correct Godot {config.TARGET_LANGUAGE} syntax and API usage. Consider adding more specific examples or constraints.")
            for task in validation_failed: feedback_parts.append(format_task_details(task))

        if file_op_failed:
            feedback_parts.append("\n### Issues: File Operation Failures (Non-Search Related)")
            feedback_parts.append("Applying the generated code failed due to file system errors or tool issues unrelated to `search_block` problems (e.g., permission errors, invalid paths reported by the tool). While potentially not a mapping issue, review the task definition for clarity regarding file paths.")
            for task in file_op_failed: feedback_parts.append(format_task_details(task))

        if other_failed:
             feedback_parts.append("\n### Issues: Other Task Failures")
             feedback_parts.append("These tasks failed during processing for other reasons (e.g., agent invocation error, unexpected agent status, context assembly failure).")
             for task in other_failed: feedback_parts.append(format_task_details(task))

        feedback_parts.append("\n**General Recommendations for Remapping:**")
        feedback_parts.append("1.  **Task Granularity:** Break down complex C++ functions/classes into smaller, more focused mapping tasks.")
        feedback_parts.append("2.  **Clarity:** Ensure mapping notes are unambiguous and provide specific Godot API/pattern guidance.")
        feedback_parts.append("3.  **Structure Alignment:** Double-check that `target_godot_file` and `target_element` in tasks accurately reflect the proposed JSON structure.")
        feedback_parts.append("4.  **Context:** Ensure the necessary C++ context (and existing Godot code context) is being provided for the agent to understand the source code accurately and generate correct `search_block`s.")

        return "\n".join(feedback_parts)
