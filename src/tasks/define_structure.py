# src/tasks/define_structure.py
from crewai import Task
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
                "propose a logical Godot 4.x project structure for converting this package. "
                f"The target language for scripts should be {config.TARGET_LANGUAGE}."
                "\n\nYour proposal should cover:\n"
                "1.  **Scene Structure:** Suggest main scenes and potentially reusable sub-scenes relevant to the package.\n"
                "2.  **Node Hierarchy:** For key scenes, outline the main nodes and their types (e.g., Node2D, Control, CharacterBody3D, custom types).\n"
                "3.  **Scripting:** Suggest names for new scripts, associate them with nodes in the hierarchy, and briefly describe their primary responsibilities.\n"
                "4.  **Mapping Ideas:** Briefly mention how key C++ concepts or classes from the snippets might translate to the proposed Godot nodes/scripts.\n"
                "5.  **Directory Structure:** Suggest where the new scenes and scripts should be placed within the Godot project structure (e.g., 'res://src/feature_x/').\n\n"
                "Focus on creating a structure that is idiomatic to Godot 4.x, maintainable, and clearly reflects the functionality of the original C++ package."
            ),
            expected_output=(
                "A detailed description of the proposed Godot structure in **Markdown format**. "
                "Use headings, bullet points, and code formatting (`like_this`) for clarity. "
                "Example structure:\n"
                "```markdown\n"
                "## Proposed Godot Structure for Package: [Package Name]\n\n"
                "**Target Language:** GDScript\n\n"
                "**Directory:** `res://src/audio_system/`\n\n"
                "**Scenes:**\n"
                "*   `audio_manager.tscn` (Main scene for managing audio)\n"
                "    *   Root Node: `AudioManager` (Node)\n"
                "        *   Script: `audio_manager.gd`\n"
                "        *   Children:\n"
                "            *   `MusicPlayer` (AudioStreamPlayer)\n"
                "            *   `SfxPlayers` (Node containing multiple AudioStreamPlayer instances)\n"
                "*   `sound_emitter_2d.tscn` (Reusable scene for positional sounds)\n"
                "    *   Root Node: `SoundEmitter2D` (Node2D)\n"
                "        *   Script: `sound_emitter_2d.gd`\n"
                "        *   Children:\n"
                "            *   `AudioPlayer` (AudioStreamPlayer2D)\n\n"
                "**Scripts:**\n"
                "*   `audio_manager.gd` (Attached to `AudioManager` node in `audio_manager.tscn`)\n"
                "    *   Responsibilities: Loading sounds, managing playback channels, handling global volume, interfacing with C++ `AudioManager` concepts.\n"
                "*   `sound_emitter_2d.gd` (Attached to `SoundEmitter2D` node in `sound_emitter_2d.tscn`)\n"
                "    *   Responsibilities: Playing sounds at a specific 2D position, managing attenuation based on C++ `SoundSource` properties.\n\n"
                "**Mapping Notes:**\n"
                "*   The C++ `AudioManager` class seems to map well to the singleton pattern often used in Godot, potentially managed by `audio_manager.gd`.\n"
                "*   C++ `SoundSource` instances could be represented by instances of `sound_emitter_2d.tscn`.\n"
                "```\n"
                "**(Note: While Markdown is requested, consider if a structured JSON output might be more reliable for automated parsing in the subsequent mapping step. For now, generate Markdown.)**"
            ),
            agent=agent,
            context=context # Pass the assembled context directly to the task
            # output_file="analysis_results/package_structure_proposal.md" # Optional: Save output directly
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
