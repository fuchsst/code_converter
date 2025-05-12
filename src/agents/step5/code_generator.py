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
        tools: A list containing tools the agent can use (e.g., FileReaderTool).
    """
    tool_names = [t.name for t in tools if hasattr(t, 'name')]
    return Agent(
        role=f"Expert C++ to {config.TARGET_LANGUAGE} Translator",
        goal=(
            f"Translate the provided C++ code elements (described in the task item and context) "
            f"into clean, idiomatic, and syntactically plausible {config.TARGET_LANGUAGE} code for Godot Engine 4.x. "
            f"Ensure the generated code matches the output file type (gs, tres, tscn). "
            f"Focus purely on generating the code string based on the task requirements and context. "
            f"Adhere to SOLID principles and Godot best practices. "
            f"Use the '{tool_names[0] if tool_names else 'File Reader'}' tool if necessary to read related Godot files (e.g., base classes, resources) for context to ensure accurate translation. "
            f"Your final output should be ONLY the generated {config.TARGET_LANGUAGE} code string, without any explanations, comments outside the code, or formatting."
        ),
        backstory=(
            f"You are a highly skilled software engineer specializing in game engine code conversion. "
            f"You excel at understanding C++ game logic and translating it accurately into {config.TARGET_LANGUAGE} for Godot 4.x. "
            f"You focus solely on generating high-quality, readable, and maintainable code based on the specific instructions and context provided for each task item. "
            f"You do not determine file operation modes or extract original code blocks; you only generate the target code itself. "
            f"You can use tools like a file reader to gather additional context from related project files if needed for accuracy."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False, # Focus on generation and formatting
        tools=tools # Assign the provided tools (e.g., FileReaderTool)
    )
