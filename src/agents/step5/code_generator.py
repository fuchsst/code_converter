# src/agents/code_generator.py
# src/agents/step5/code_generator.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)

def get_code_generator_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for code generation.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing tools the agent can use. Ensure 'Godot Project File Reader' is included.
    """
    # This agent specifically deals with reading Godot files for context.
    godot_reader_tool_name = "Godot Project File Reader"
    if not any(t.name == godot_reader_tool_name for t in tools if hasattr(t, 'name')):
        logger.warning(f"'{godot_reader_tool_name}' not found in provided tools for CodeGeneratorAgent. Agent may not function correctly.")
        # Fallback or error handling could be added here if necessary

    return Agent(
        role=f"Expert C++ to {config.TARGET_LANGUAGE} Full File Content Generator",
        goal=(
            f"Based on the provided C++ code elements (described in the task item details and broader context), "
            f"your primary goal is to generate the **complete and final {config.TARGET_LANGUAGE} code content** for the specified target Godot file. "
            f"1. **Analyze Task & Context**: Understand the C++ source, the mapping task details, and any existing Godot project context provided. "
            f"2. **Check Target File**: The context might indicate if the target Godot file (e.g., `{'{target_godot_file}'}`) already exists and may contain its current content. If not, or to ensure you have the latest version, you **MUST use the '{godot_reader_tool_name}' tool** to read the current content of the target Godot file if it exists. "
            f"3. **Generate Full Content**: If the file is new, generate its complete content from scratch. If the file exists, integrate the required changes/additions into its existing content, producing a new version of the **entire file content**. "
            f"Ensure the generated code is clean, idiomatic, and syntactically plausible {config.TARGET_LANGUAGE} for Godot Engine 4.x, adhering to SOLID principles and Godot best practices. "
            f"Your final output **MUST BE ONLY the raw, complete {config.TARGET_LANGUAGE} code string for the entire file**. Do not include any explanations, comments outside the code, or markdown formatting like ```."
        ),
        backstory=(
            f"You are a highly skilled software engineer specializing in game engine code conversion and generation. "
            f"You excel at understanding C++ game logic and translating or implementing it accurately as complete {config.TARGET_LANGUAGE} files for Godot 4.x. "
            f"You are meticulous about producing the full content for a file, whether it's new or an update to an existing one. "
            f"You always use the '{godot_reader_tool_name}' tool to fetch existing content before modification to ensure context is not lost and changes are integrated correctly. "
            f"You understand that your output will directly overwrite the target file."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=tools
    )
