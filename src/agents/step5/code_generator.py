# src/agents/code_generator.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
from src.tasks.step5.process_code import CodeGenerationResult
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
        role=f"Expert C++ to {config.TARGET_LANGUAGE} Translator & Formatter",
        goal=(
            f"Translate the provided C++ code elements (described in the task item and context) "
            f"into clean, idiomatic, and syntactically plausible {config.TARGET_LANGUAGE} code for Godot Engine 4.x. "
            f"Adhere to SOLID principles and Godot best practices. "
            f"Analyze the task and target file content (provided in context) to determine the output format ('FULL_FILE' or 'CODE_BLOCK'). "
            f"If 'CODE_BLOCK', extract the exact original code block (`search_block`) from the target file content that your generated code should replace. "
            f"Use the '{tool_names[0] if tool_names else 'File Reader'}' tool if necessary to read related Godot files (e.g., base classes, resources) for context. "
            f"Your final output MUST be ONLY a raw JSON object string conforming precisely to the `CodeGenerationResult` model (containing `generated_code`, `output_format`, `search_block`). "
            f"Do NOT include any introductory text, explanations, or markdown formatting like ```json."
        ),
        backstory=(
            f"You are a highly skilled software engineer specializing in game engine code conversion and formatting. "
            f"You excel at understanding C++ game logic and translating it accurately into {config.TARGET_LANGUAGE} for Godot 4.x. "
            f"You meticulously analyze the task requirements and existing code context to decide whether to generate a full file or a code block replacement. "
            f"If replacing a block, you precisely extract the original code to be replaced. "
            f"You can use tools like a file reader to gather additional context from related project files if needed. "
            f"Your output is always a perfectly formatted JSON object containing the generated code and formatting details, ready for the next step in the pipeline."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False, # Focus on generation and formatting
        tools=tools # Assign the provided tools (e.g., FileReaderTool)
        # Ensure the LLM instance is configured to output JSON if possible,
        # or rely on strong prompting for the JSON output format.
        # output_pydantic=CodeGenerationResult # Set this on the Task, not the Agent
    )
