# src/agents/code_generator.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)

def get_code_generator_agent(llm_instance: BaseLLM):
    """
    Creates and returns the configured CrewAI Agent instance for code generation.

    Args:
        llm_instance: The pre-configured LLM instance to use.
    """
    return Agent(
        role=f"Expert C++ to {config.TARGET_LANGUAGE} Translator",
        goal=(
            f"Translate the provided C++ code elements (described in the task item and context) "
            f"into clean, idiomatic, and syntactically plausible {config.TARGET_LANGUAGE} code for Godot Engine 4.x. "
            f"Focus purely on generating the code snippet or full file content as requested by the task. "
            f"Adhere to SOLID principles and Godot best practices. "
            f"The output should be ONLY the generated code string."
        ),
        backstory=(
            f"You are a highly skilled software engineer specializing in game engine code conversion. "
            f"You excel at understanding C++ game logic and translating it accurately into {config.TARGET_LANGUAGE} for Godot 4.x. "
            f"You focus on generating high-quality, readable, and maintainable code, following the specific instructions provided for each task item. "
            f"You do not handle file operations, validation, or deciding between full files/code blocks; you only generate the code itself."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=[] # This agent does not use external tools directly
    )
