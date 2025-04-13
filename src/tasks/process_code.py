# src/tasks/process_code.py
from crewai import Task
from logger_setup import get_logger
import config
# Import agent definition if needed for type hinting or direct reference
# from agents.code_processor import CodeProcessorAgent # Example

logger = get_logger(__name__)

class ProcessCodeTask:
    """
    CrewAI Task definition for executing the C++ to Godot conversion tasks
    based on a provided JSON task list.
    """
    def create_task(self, agent, context: str, json_task_list: str):
        """
        Creates the CrewAI Task instance for processing the conversion tasks.

        Args:
            agent (Agent): The CodeProcessorAgent instance responsible for this task.
            context (str): The context string containing relevant C++ source code,
                           potentially existing Godot code, and mapping strategy notes,
                           assembled by the ContextManager based on the JSON task list.
            json_task_list (str): A string containing the JSON list of tasks generated in Step 4.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating ProcessCodeTask for agent: {agent.role}")
        # This task guides the agent to process the *entire* list internally.
        # The actual implementation of the internal loop depends on agent/tool design.
        return Task(
            description=(
                "You are tasked with converting C++ code to Godot based on a detailed JSON task list. "
                "The context provided contains relevant C++ source code and potentially existing Godot code snippets or mapping notes. "
                f"The target language is {config.TARGET_LANGUAGE}.\n\n"
                f"**Input JSON Task List:**\n```json\n{json_task_list}\n```\n\n"
                "**Your Process:**\n"
                "1.  Iterate through **each task** defined in the JSON list above.\n"
                "2.  For **each task**, analyze its details (`description`, `target_godot_file`, `target_element`, `source_cpp_files`, `source_cpp_elements`, `mapping_notes`).\n"
                "3.  Use the provided context (C++ code, existing Godot code, etc.) and the task's `mapping_notes` to generate the required {config.TARGET_LANGUAGE} code.\n"
                "4.  **Determine Output Format:** Decide if you need to output the **entire content** for `target_godot_file` (e.g., for new files or significant changes) OR just a **specific code block/function** (e.g., for modifying an existing function). **Clearly state your choice ('FULL_FILE' or 'CODE_BLOCK')**.\n"
                "5.  **If outputting a `CODE_BLOCK`:**\n"
                "    - You **MUST** also provide the exact original code block (`search_block`) that the `generated_code` should replace.\n"
                "    - Ensure `search_block` matches the original file content character-for-character, including indentation and line endings.\n"
                "    - Ensure `generated_code` includes enough context (like the full function signature and body) for accurate replacement.\n"
                "6.  **Validate (Optional but Recommended):** Request syntax validation for the `generated_code` using the `validate_gdscript_syntax` tool.\n"
                "7.  **Attempt Fixes:** If validation fails, analyze the error message. Attempt to fix the syntax error (max 2 attempts per task) and re-validate.\n"
                "8.  **Format Output for Each Task:** For each task processed, structure your response clearly, including:\n"
                "    - The `task_id`.\n"
                "    - The chosen output format (`output_format`: 'FULL_FILE' or 'CODE_BLOCK').\n"
                "    - The generated code (`generated_code`: string containing the full file or code block).\n"
                "    - **If `output_format` is `CODE_BLOCK`, include `search_block`: (string) The exact original code block to search for.**\n"
                "    - Validation status (`validation_status`: 'success', 'failure', or 'skipped').\n"
                "    - Validation errors (`validation_errors`: string or null).\n\n"
                "Compile the results for **all** processed tasks into a single final output."
            ),
            expected_output=(
                "A structured report detailing the outcome for each task processed from the input JSON list. "
                "This report should ideally be a **JSON list** where each object corresponds to a processed task and contains the following keys:\n"
                "- `task_id`: (string) The ID of the task from the input list.\n"
                "- `status`: (string) 'completed' or 'failed'.\n"
                "- `output_format`: (string) 'FULL_FILE' or 'CODE_BLOCK'.\n"
                "- `generated_code`: (string) The generated Godot code (either full file content or the specific block).\n"
                "- `search_block`: (string | null) **Required if `output_format` is 'CODE_BLOCK'**. The exact original code block to search for. Null otherwise.\n"
                "- `target_godot_file`: (string) The target file path from the task definition.\n"
                "- `target_element`: (string) The target element (function, etc.) from the task definition.\n"
                "- `validation_status`: (string) 'success', 'failure', or 'skipped'.\n"
                "- `validation_errors`: (string | null) Error messages if validation failed, otherwise null.\n"
                "- `error_message`: (string | null) Any error message if the task processing itself failed.\n\n"
                "Example of the expected JSON list output:\n"
                "```json\n"
                "[\n"
                "  {\n"
                "    \"task_id\": \"map_player_movement_001\",\n"
                "    \"status\": \"completed\",\n"
                "    \"output_format\": \"CODE_BLOCK\",\n"
                "    \"search_block\": \"func _physics_process(delta):\\n    # Original movement logic here\\n    pass\\n\",\n"
                "    \"generated_code\": \"func _physics_process(delta):\\n    var input_dir = Input.get_vector(\\\"left\\\", \\\"right\\\", \\\"up\\\", \\\"down\\\")\\n    velocity = input_dir * speed\\n    move_and_slide()\\n\",\n"
                "    \"target_godot_file\": \"src/player/player.gd\",\n"
                "    \"target_element\": \"_physics_process\",\n"
                "    \"validation_status\": \"success\",\n"
                "    \"validation_errors\": null,\n"
                "    \"error_message\": null\n"
                "  },\n"
                "  {\n"
                "    \"task_id\": \"map_player_jump_002\",\n"
                "    \"status\": \"completed\",\n"
                "    \"output_format\": \"CODE_BLOCK\",\n"
                "    \"search_block\": \"func _input(event):\\n    pass\\n\",\n"
                "    \"generated_code\": \"func _input(event):\\n    if event.is_action_pressed(\\\"jump\\\") and is_on_floor():\\n        velocity.y = JUMP_VELOCITY\\n\",\n"
                "    \"target_godot_file\": \"src/player/player.gd\",\n"
                "    \"target_element\": \"_input\",\n"
                "    \"validation_status\": \"failure\",\n"
                "    \"validation_errors\": \"player.gd:5: Parse Error: Unexpected token: Identifier:JUMP_VELOCITY\",\n"
                "    \"error_message\": null\n"
                "  },\n"
                "  {\n"
                "    \"task_id\": \"create_player_scene_003\",\n"
                "    \"status\": \"completed\",\n"
                "    \"output_format\": \"FULL_FILE\",\n"
                "    \"search_block\": null,\n"
                "    \"generated_code\": \"[gd_scene load_steps=2 format=3 uid=\\\"uid://abc123def456\\\"]\\n\\n[ext_resource type=\\\"Script\\\" path=\\\"res://src/player/player.gd\\\" id=\\\"1_xyz\\\"]\\n\\n[node name=\\\"Player\\\" type=\\\"CharacterBody3D\\\"]\\nscript = ExtResource(\\\"1_xyz\\\")\\n\",\n"
                "    \"target_godot_file\": \"src/player/player.tscn\",\n"
                "    \"target_element\": null,\n"
                "    \"validation_status\": \"skipped\",\n"
                "    \"validation_errors\": null,\n"
                "    \"error_message\": null\n"
                "  }\n"
                "]\n"
                "```"
            ),
            agent=agent,
            context=context, # Pass the assembled context directly to the task
            output_json=True # Expecting the final compiled report as a JSON list
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.code_processor import CodeProcessorAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = CodeProcessorAgent()
#     processor_agent = agent_creator.get_agent()
#
#     # Dummy context and task list for testing
#     test_context = """
#     **File:** `src/player/player.h`
#     ```cpp
#     class CppPlayer {
#         Vector3 position;
#         Vector3 velocity;
#         void updateMovement(float delta);
#         void handleJump();
#     };
#     ```
#     """
#     test_task_list = json.dumps([
#         {
#             "task_id": "map_player_movement_001",
#             "description": "Implement basic player movement in _physics_process based on CppPlayer::updateMovement",
#             "target_godot_file": "src/player/player.gd",
#             "target_element": "_physics_process",
#             "source_cpp_files": ["src/player/player.cpp", "src/input/input_handler.h"],
#             "source_cpp_elements": ["CppPlayer::updateMovement"],
#             "mapping_notes": "Use Input.get_vector for direction. Apply velocity to CharacterBody3D.velocity. Call move_and_slide()."
#         }
#     ])
#
#     task_creator = ProcessCodeTask()
#     process_task = task_creator.create_task(processor_agent, test_context, test_task_list)
#     print("ProcessCodeTask created:")
#     print(f"Description: {process_task.description}")
#     print(f"Expected Output: {process_task.expected_output}")
