# src/tasks/process_code.py
from crewai import Task
from logger_setup import get_logger
import config
# Import agent definition if needed for type hinting or direct reference
# from agents.code_processor import CodeProcessorAgent # Example

logger = get_logger(__name__)

class ProcessCodeTask:
    """
    CrewAI Task definition for executing a single C++ to Godot conversion task item.
    """
    def create_task(self, agent, context: str, task_item_json: str):
        """
        Creates the CrewAI Task instance for processing a single conversion task item.

        Args:
            agent (Agent): The CodeProcessorAgent instance responsible for this task.
            context (str): The context string containing relevant C++ source code,
                           potentially existing Godot code for the target file, and mapping notes,
                           assembled by the Orchestrator specifically for this task item.
            task_item_json (str): A string containing the JSON object for the single task item to be processed.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating ProcessCodeTask for agent: {agent.role}")
        return Task(
            description=(
                f"Process the **single conversion task item** detailed below, using the provided context. The target language is {config.TARGET_LANGUAGE}.\n\n"
                f"**Input Task Item Details (JSON):**\n```json\n{task_item_json}\n```\n\n"
                "**Your Process (as the CodeProcessorAgent):**\n"
                "1.  Analyze the task details (from the JSON above) and the provided context (which includes relevant C++ code, mapping notes, and potentially existing Godot code for the target file).\n"
                f"2.  Generate the required {config.TARGET_LANGUAGE} code based on the task, context, and mapping notes, adhering to SOLID principles.\n"
                "3.  **Determine Output Format:** Decide if the generated code represents a 'FULL_FILE' (for new files or complete overwrites) or a 'CODE_BLOCK' (for modifying existing files).\n"
                "4.  **Extract Search Block (if modifying):** If the output format is 'CODE_BLOCK', you MUST extract the exact original code block (`search_block`) from the existing Godot code provided in the context that the `generated_code` should replace. This `search_block` must be precise to allow the orchestrator's `Replace Content In File` tool to work correctly.\n"
                "5.  **Report Result:** Structure your final output as a **single JSON object** containing the results of your processing, including `task_id`, `status` (completed/failed based on *your* ability to generate code and required info), `output_format`, `generated_code`, `search_block` (if applicable), `target_godot_file`, `target_element`, and optionally `validation_status` (your internal assessment) and `error_message`. **You do not perform file operations or final validation yourself.**"
            ),
            expected_output=(
                "A **single JSON object** summarizing the outcome of processing the task item. This object MUST include:\n"
                "- `task_id`: (string) The ID from the input task item.\n"
                "- `status`: (string) 'completed' if code generation and analysis (e.g., extracting search_block if needed) were successful from the agent's perspective, 'failed' otherwise.\n"
                "- `output_format`: (string) 'FULL_FILE' or 'CODE_BLOCK'.\n"
                "- `generated_code`: (string) The generated {config.TARGET_LANGUAGE} code snippet or full file content.\n"
                "- `search_block`: (string | null) The exact original code block to search for if `output_format` is 'CODE_BLOCK', otherwise null. **Must be accurate!**\n"
                "- `target_godot_file`: (string) The target file path from the task item.\n"
                "- `target_element`: (string) The target element from the task item.\n"
                "- `validation_status`: (string) Optional: Indicate 'attempted_fix' if internal validation/fixing was tried, otherwise 'not_validated' or 'success' if confident. The orchestrator performs the definitive validation.\n"
                "- `error_message`: (string | null) Description of any error during code generation or analysis.\n\n"
                "Example Output (Success - Full File):\n"
                "```json\n"
                "{\n"
                "  \"task_id\": \"create_player_script_001\",\n"
                "  \"status\": \"completed\",\n"
                "  \"output_format\": \"FULL_FILE\",\n"
                "  \"generated_code\": \"extends CharacterBody3D\\n\\nfunc _physics_process(delta):\\n    pass\\n\",\n"
                "  \"search_block\": null,\n"
                "  \"target_godot_file\": \"src/player/player.gd\",\n"
                "  \"target_element\": \"player.gd\",\n"
                "  \"validation_status\": \"success\",\n"
                "  \"error_message\": null\n"
                "}\n"
                "```\n"
                "Example Output (Success - Code Block):\n"
                "```json\n"
                "{\n"
                "  \"task_id\": \"add_jump_func_002\",\n"
                "  \"status\": \"completed\",\n"
                "  \"output_format\": \"CODE_BLOCK\",\n"
                "  \"generated_code\": \"func jump():\\n    velocity.y = JUMP_VELOCITY\",\n"
                "  \"search_block\": \"func _physics_process(delta):\\n    pass\",\n"
                "  \"target_godot_file\": \"src/player/player.gd\",\n"
                "  \"target_element\": \"jump function\",\n"
                "  \"validation_status\": \"success\",\n"
                "  \"error_message\": null\n"
                "}\n"
                "```\n"
                "Example Output (Failure - Agent Error):\n"
                "```json\n"
                "{\n"
                "  \"task_id\": \"complex_state_machine_003\",\n"
                "  \"status\": \"failed\",\n"
                "  \"output_format\": null,\n"
                "  \"generated_code\": null,\n"
                "  \"search_block\": null,\n"
                "  \"target_godot_file\": \"src/enemy/enemy_fsm.gd\",\n"
                "  \"target_element\": \"update_state\",\n"
                "  \"validation_status\": \"not_validated\",\n"
                "  \"error_message\": \"Failed to understand complex C++ state logic.\"\n"
                "}\n"
                "```"
            ),
            agent=agent,
            context=context,
            output_json=True # Expecting the final report as a JSON object
        )
