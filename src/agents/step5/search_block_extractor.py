# src/agents/step5/search_block_extractor.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
from typing import List, Optional
import src.config as config
from crewai.tools import BaseTool

logger = get_logger(__name__)

def get_search_block_extractor_agent(llm_instance: BaseLLM, tools: Optional[List[BaseTool]] = None):
    """
    Creates and returns the agent responsible for extracting the search block for code replacement.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: Optional list of tools (likely not needed).
    """
    return Agent(
        role="Code Block Extractor",
        goal=(
            "Given the original task details, a generated code snippet (for context), and the full content of the existing target file (provided in your task description's context), "
            "your primary task is to identify and extract the **exact, original, unfenced code block** within the existing target file content that the generated code is intended to replace. "
            "Pay close attention to the task description and any `target_element` specified in the task details to accurately locate the correct block. "
            "The extracted block must match the original file content character-for-character, including all whitespace and line endings. "
            "If, after careful analysis, you determine that no specific block needs replacement (e.g., the task implies adding entirely new code, the target file is empty, or you cannot confidently identify the precise block for replacement), "
            "your entire output **MUST BE the exact literal string 'NULL'**. "
            "Otherwise, your entire output **MUST BE ONLY the extracted code block string itself**, without any surrounding text, explanations, or markdown fences."
        ),
        backstory=(
            "You are a highly precise code analysis agent. Your expertise lies in identifying specific code sections within a larger file based on contextual clues like task descriptions and generated code snippets. "
            "You understand the critical importance of extracting the *exact* original code block for replacement operations. "
            "You meticulously compare the task requirements with the provided file content to find the precise match. "
            "If no replacement is indicated or possible, you clearly signal this by outputting 'NULL'."
        ),
        llm=llm_instance,
        verbose=True,
        max_execution_time=config.VERTEX_TIMEOUT,
        allow_delegation=False,
        tools=tools or [],
    )
