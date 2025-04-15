# src/tasks/define_structure.py
from crewai import Task
import config
from logger_setup import get_logger
# Import agent definition if needed for type hinting or direct reference
# from agents.structure_definer import StructureDefinerAgent # Example

logger = get_logger(__name__)

class DefineStructureTask:
    """
    CrewAI Task definition for proposing a Godot project structure based on
    a C++ work package.
    """
    def create_task(self, agent, context: str):
        """
        Creates the CrewAI Task instance for defining the Godot structure.

        Args:
            agent (Agent): The StructureDefinerAgent instance responsible for this task.
            context (str): The context string containing work package info and C++ code snippets,
                           assembled by the ContextManager.

        Returns:
            Task: The CrewAI Task object.
        """
        logger.info(f"Creating DefineStructureTask for agent: {agent.role}")
        return Task(
            description=(
                "Analyze the provided context, which includes information about a C++ work package "
                "(file list, description) and potentially relevant C++ code snippets. Your goal is to "
                "propose a logical Godot 4.x project structure for converting this package, ensuring the design **adheres to SOLID principles and promotes good separation of concerns**. "
                f"The target language for scripts should be {config.TARGET_LANGUAGE}."
                "\n\nYour proposal should cover:\n"
                "1.  **Scene Structure:** Suggest main scenes and potentially reusable sub-scenes relevant to the package.\n"
                "2.  **Node Hierarchy:** For key scenes, outline the main nodes and their types (e.g., Node2D, Control, CharacterBody3D, custom types).\n"
                "3.  **Scripting:** Suggest names for new scripts, associate them with nodes in the hierarchy, and describe their primary responsibilities. **Crucially, ensure these responsibilities are focused and decoupled (e.g., separate scripts for input, state, logic) to align with SOLID.**\n"
                "4.  **Mapping Ideas:** Briefly mention how key C++ concepts or classes from the snippets might translate to the proposed Godot nodes/scripts, keeping separation of concerns in mind.\n"
                "5.  **Directory Structure:** Suggest where the new scenes and scripts should be placed within the Godot project structure (e.g., 'res://src/feature_x/').\n\n"
                "Focus on creating a structure that is idiomatic to Godot 4.x, maintainable, testable, and clearly reflects the functionality of the original C++ package while respecting SOLID principles."
            ),
            expected_output=(
                "A **JSON object** describing the proposed Godot structure, designed with SOLID principles in mind. This JSON should be the *only* content in your output. "
                "The JSON object should have top-level keys like `target_language`, `base_directory`, `scenes`, and `scripts`. "
                "`scenes` should be a list of objects, each describing a scene file (`file_path`), its root node (`root_node_name`, `root_node_type`), associated script (`script_path`), and potentially its children (`children`: list of node objects with `name`, `type`, `script_path`). "
                "`scripts` should be a list of objects, each describing a script file (`file_path`), its primary associated node (`attached_to_node` in `attached_to_scene`), and its key `responsibilities` (list of strings). "
                "Include brief `mapping_notes` within scene or script objects where relevant, explaining the connection to C++ concepts.\n\n"
                "Example JSON structure:\n"
                "```json\n"
                "{\n"
                "  \"target_language\": \"GDScript\",\n"
                "  \"base_directory\": \"res://src/audio_system/\",\n"
                "  \"scenes\": [\n"
                "    {\n"
                "      \"file_path\": \"res://src/audio_system/audio_manager.tscn\",\n"
                "      \"root_node_name\": \"AudioManager\",\n"
                "      \"root_node_type\": \"Node\",\n"
                "      \"script_path\": \"res://src/audio_system/audio_manager.gd\",\n"
                "      \"children\": [\n"
                "        {\"name\": \"MusicPlayer\", \"type\": \"AudioStreamPlayer\", \"script_path\": null},\n"
                "        {\"name\": \"SfxPlayers\", \"type\": \"Node\", \"script_path\": null}\n"
                "      ],\n"
                "      \"mapping_notes\": \"Maps to the C++ AudioManager singleton concept.\"\n"
                "    },\n"
                "    {\n"
                "      \"file_path\": \"res://src/audio_system/sound_emitter_2d.tscn\",\n"
                "      \"root_node_name\": \"SoundEmitter2D\",\n"
                "      \"root_node_type\": \"Node2D\",\n"
                "      \"script_path\": \"res://src/audio_system/sound_emitter_2d.gd\",\n"
                "      \"children\": [\n"
                "        {\"name\": \"AudioPlayer\", \"type\": \"AudioStreamPlayer2D\", \"script_path\": null}\n"
                "      ],\n"
                "      \"mapping_notes\": \"Represents instances of C++ SoundSource.\"\n"
                "    }\n"
                "  ],\n"
                "  \"scripts\": [\n"
                "    {\n"
                "      \"file_path\": \"res://src/audio_system/audio_manager.gd\",\n"
                "      \"attached_to_node\": \"AudioManager\",\n"
                "      \"attached_to_scene\": \"res://src/audio_system/audio_manager.tscn\",\n"
                "      \"responsibilities\": [\n"
                "        \"Loading sounds\",\n"
                "        \"Managing playback channels\",\n"
                "        \"Handling global volume\"\n"
                "      ],\n"
                "      \"mapping_notes\": \"Interfaces with C++ AudioManager concepts.\"\n"
                "    },\n"
                "    {\n"
                "      \"file_path\": \"res://src/audio_system/sound_emitter_2d.gd\",\n"
                "      \"attached_to_node\": \"SoundEmitter2D\",\n"
                "      \"attached_to_scene\": \"res://src/audio_system/sound_emitter_2d.tscn\",\n"
                "      \"responsibilities\": [\n"
                "        \"Playing sounds at a specific 2D position\",\n"
                "        \"Managing attenuation\"\n"
                "      ],\n"
                "      \"mapping_notes\": \"Based on C++ SoundSource properties.\"\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "```"
            ),
            agent=agent,
            context=context, # Pass the assembled context directly to the task
            output_json=True # Expect the output to be a valid JSON object
            # output_file="analysis_results/package_structure_proposal.json" # Optional: Save output directly
        )

# Example instantiation (for testing or direct use if needed)
# if __name__ == '__main__':
#     from agents.structure_definer import StructureDefinerAgent # Need agent for task
#     # Assume agent is initialized properly
#     agent_creator = StructureDefinerAgent()
#     structure_agent = agent_creator.get_agent()
#
#     # Dummy context for testing
#     test_context = """
#     **Work Package Definition (JSON):**
#     ```json
#     {
#       "description": "Handles sound loading and playback.",
#       "files": ["src/audio/manager.cpp", "src/audio/sound.h"]
#     }
#     ```
#
#     **File:** `src/audio/manager.h`
#     ```cpp
#     class AudioManager {
#     public:
#         static AudioManager& instance();
#         void playSound(const std::string& soundId);
#         void setVolume(float volume);
#     };
#     ```
#     """
#
#     task_creator = DefineStructureTask()
#     define_task = task_creator.create_task(structure_agent, test_context)
#     print("DefineStructureTask created:")
#     print(f"Description: {define_task.description}")
#     print(f"Expected Output: {define_task.expected_output}")
