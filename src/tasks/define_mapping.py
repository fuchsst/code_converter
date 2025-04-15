# src/tasks/define_mapping.py
from crewai import Task
from logger_setup import get_logger
import config
# Import agent definition if needed for type hinting or direct reference
# from agents.mapping_definer import MappingDefinerAgent # Example

logger = get_logger(__name__)

class DefineMappingTask:
    """
    CrewAI Task definition for defining the C++ to Godot mapping strategy
    and generating an actionable task list.
    """
    def create_task(self, agent, context: str):
        """
        Creates the CrewAI Task instance for defining the mapping.

        Args:
            agent (Agent): The MappingDefinerAgent instance responsible for this task.
            context (str): The context string containing work package info, C++ code,
                           and the proposed Godot structure (Markdown), assembled by ContextManager.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating DefineMappingTask for agent: {agent.role}")
        # Note: This task requests two distinct outputs (Markdown + JSON) in one go.
        # This is ambitious and relies heavily on the LLM's ability to follow complex instructions.
        # Consider splitting into two tasks/agents if this proves unreliable.
        return Task(
            description=(
                "Analyze the provided context, which includes:\n"
                "- The C++ work package definition (files, description).\n"
                "- Relevant C++ source code snippets from the package.\n"
                "- The proposed Godot project structure (as a **JSON object**, designed with SOLID principles) for this package.\n"
                f"- The target language: {config.TARGET_LANGUAGE}.\n\n"
                "Your primary goal is to create a detailed plan for converting the C++ code "
                "within this package to the proposed Godot structure (provided as JSON). You must generate **TWO** distinct outputs:\n\n"
                "1.  **Mapping Strategy (Markdown):**\n"
                "    - Write a concise Markdown document outlining the high-level conversion strategy.\n"
                "    - Describe how key C++ classes, functions, data structures, and patterns identified in the code snippets will map to the proposed Godot nodes, scenes, and **specifically the decoupled scripts** defined in the input JSON structure.\n"
                "    - Emphasize how the mapping maintains the separation of concerns established in the structure.\n"
                "    - Reference specific elements from the input JSON structure (e.g., scene file paths, script names) where appropriate.\n"
                "    - Mention potential challenges or areas requiring careful attention.\n\n"
                "2.  **Actionable Task List (JSON):**\n"
                "    - Generate a JSON list where each item represents a single, granular conversion task.\n"
                "    - **Ensure each task targets the correct, specific Godot script/node according to the SOLID-based structure proposed in Step 3.**\n"
                "    - Each task object must contain keys like:\n"
                "        - `task_id`: A unique identifier (e.g., 'map_func_001').\n"
                "        - `description`: A brief description of the specific action (e.g., 'Implement Player movement logic in player_movement.gd based on CppPlayer::updateMovement').\n"
                "        - `target_godot_file`: The relative path to the Godot script/scene to be created/modified (must align with paths defined in the input JSON structure, e.g., 'res://src/player/player_movement.gd').\n"
                "        - `target_element`: The specific function, method, or node within the Godot file (e.g., '_physics_process', 'setup_animations').\n"
                "        - `source_cpp_files`: A list of relevant C++ source/header file paths for reference (e.g., ['src/player/player.cpp', 'src/core/vector.h']).\n"
                "        - `source_cpp_elements`: Specific C++ functions, classes, or members to reference (e.g., ['CppPlayer::updateMovement', 'Vector3::normalize']).\n"
                "        - `mapping_notes`: Brief instructions on the required logic or specific API mappings (e.g., 'Map CppPlayer velocity vector to CharacterBody3D velocity property. Use Input.get_vector for input.').\n"
                "    - Ensure tasks are small, focused, and provide enough detail for a code generation agent to attempt implementation, respecting the intended separation of concerns.\n\n"
                "**Crucially, format your entire response so that the Markdown strategy comes first, followed by a clear separator (`--- JSON TASK LIST BELOW ---`), and then the complete JSON task list.**"
            ),
            expected_output=(
                "A single string containing:\n"
                "1. The Mapping Strategy document in Markdown format.\n"
                "2. A clear separator line (`--- JSON TASK LIST BELOW ---`).\n"
                "3. The complete Actionable Task List formatted as a valid JSON list of objects.\n\n"
                "Example:\n"
                "```markdown\n"
                "## Mapping Strategy for Package: [Package Name]\n\n"
                "The C++ `Player` class will be mapped to a `CharacterBody3D` node using the `player.gd` script...\n"
                "Input handling will transition from C++ callbacks to Godot's `_input()` or `Input.get_action_strength()`...\n"
                "Potential challenge: Complex physics interactions in C++ need careful mapping to Godot's physics engine...\n"
                "```\n"
                "--- JSON TASK LIST BELOW ---\n"
                "```json\n"
                "[\n"
                "  {\n"
                "    \"task_id\": \"map_player_movement_001\",\n"
                "    \"description\": \"Implement basic player movement in _physics_process based on CppPlayer::updateMovement\",\n"
                "    \"target_godot_file\": \"src/player/player.gd\",\n"
                "    \"target_element\": \"_physics_process\",\n"
                "    \"source_cpp_files\": [\"src/player/player.cpp\", \"src/input/input_handler.h\"],\n"
                "    \"source_cpp_elements\": [\"CppPlayer::updateMovement\"],\n"
                "    \"mapping_notes\": \"Use Input.get_vector for direction. Apply velocity to CharacterBody3D.velocity. Call move_and_slide().\"\n"
                "  },\n"
                "  {\n"
                "    \"task_id\": \"map_player_jump_002\",\n"
                "    \"description\": \"Implement player jump logic in _input based on CppPlayer::handleJump\",\n"
                "    \"target_godot_file\": \"src/player/player.gd\",\n"
                "    \"target_element\": \"_input\",\n"
                "    \"source_cpp_files\": [\"src/player/player.cpp\"],\n"
                "    \"source_cpp_elements\": [\"CppPlayer::handleJump\"],\n"
                "    \"mapping_notes\": \"Check for Input.is_action_just_pressed('jump') and is_on_floor(). Apply vertical impulse to velocity.y.\"\n"
                "  }\n"
                "]\n"
                "```"
            ),
            agent=agent,
            context=context, # Pass the assembled context directly to the task
            # Note: CrewAI doesn't inherently support splitting a single string output into distinct formats.
            # The orchestrator (main.cli.py) will need logic to parse this combined output string
            # based on the separator '--- JSON TASK LIST BELOW ---'.
            # output_json=True # Cannot use output_json=True as the output contains Markdown first.
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.mapping_definer import MappingDefinerAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = MappingDefinerAgent()
#     mapping_agent = agent_creator.get_agent()
#
#     # Dummy context for testing
#     test_context = """
#     **Work Package Definition (JSON):**
#     ```json
#     {
#       "description": "Handles player movement and input.",
#       "files": ["src/player/player.cpp", "src/player/player.h", "src/input/input_handler.h"]
#     }
#     ```
#
#     **Proposed Godot Structure:**
#     ```markdown
#     ## Proposed Godot Structure for Package: Player Control
#     **Directory:** `res://src/player/`
#     **Scenes:**
#     *   `player.tscn` (Root: CharacterBody3D, Script: `player.gd`)
#     **Scripts:**
#     *   `player.gd`: Handles movement, jumping, input.
#     ```
#
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
#
#     task_creator = DefineMappingTask()
#     define_task = task_creator.create_task(mapping_agent, test_context)
#     print("DefineMappingTask created:")
#     print(f"Description: {define_task.description}")
#     print(f"Expected Output: {define_task.expected_output}")
