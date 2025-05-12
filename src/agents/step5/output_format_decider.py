# src/agents/step5/output_format_decider.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
from typing import List, Optional
from crewai.tools import BaseTool

logger = get_logger(__name__)

def get_output_format_decider_agent(llm_instance: BaseLLM, tools: Optional[List[BaseTool]] = None):
    """
    Creates and returns the agent responsible for deciding the file output format.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: Optional list of tools (likely not needed).
    """
    return Agent(
        role="File Output Format Decider",
        goal=(
            "Analyze the provided generated code snippet, the original task details, and the existing target file content (if available in context). "
            "Based on whether the generated code represents a complete file or just a modification/addition to existing content, decide the output format. "
            "Your final, entire output **MUST BE ONLY** the string 'FULL_FILE' or **ONLY** the string 'CODE_BLOCK'. "
            "Do not include any other text, explanations, or formatting. "
            "Consider the task description: does it imply creating a new file or modifying an existing one? "
            "Consider the generated code: does it look like a complete script or just a function/method/block? "
            "Consider the existing file content: is it empty or does the generated code seem intended to replace a part of it?"
        ),
        backstory=(
            "You are a specialized analyst focused on determining how generated code should be applied to a file system. "
            "You examine the code itself, the instructions that led to its generation, and the context of the target file. "
            "Your sole responsibility is to classify the operation as either overwriting the entire file ('FULL_FILE') or replacing a specific section ('CODE_BLOCK'). "
            "You provide only this classification string as your output."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools or [],
    )
