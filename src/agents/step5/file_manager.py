# src/agents/file_manager.py
from crewai import Agent, BaseLLM
from src.logger_setup import get_logger
import src.config as config
from crewai.tools import BaseTool
from typing import List

logger = get_logger(__name__)

def get_file_manager_agent(llm_instance: BaseLLM, tools: List[BaseTool]):
    """
    Creates and returns the configured CrewAI Agent instance for file management.

    Args:
        llm_instance: The pre-configured LLM instance to use.
        tools: A list containing the instantiated file operation tool(s)
                (e.g., FileWriterTool, FileReplacerTool).
    """
    # Explicitly find tool names for clarity in prompts
    writer_tool_name = next((t.name for t in tools if "Writer" in t.name), "FileWriterTool")
    replacer_tool_name = next((t.name for t in tools if "Replacer" in t.name), "FileReplacerTool")

    return Agent(
        role="File System Operations Specialist",
        goal=(
            f"Your SOLE task is to execute ONE file operation using ONE of your tools: '{writer_tool_name}' or '{replacer_tool_name}'. "
            f"You will receive `file_path`, `content`, and `output_format` ('FULL_FILE' or 'CODE_BLOCK'). You might also receive `search_block`. "
            f"**Decision Logic:** "
            f"1. If `output_format` is 'FULL_FILE', you MUST use the '{writer_tool_name}' tool. Pass `file_path` and `content` to it. "
            f"2. If `output_format` is 'CODE_BLOCK', you MUST use the '{replacer_tool_name}' tool. Construct the exact `diff` string ('<<<<<<< SEARCH\\n[search_block]\\n=======\\n[content]\\n>>>>>>> REPLACE') and pass `file_path` and `diff` to it. "
            f"Execute the chosen tool EXACTLY ONCE with the correct parameters. Report the result message from the tool."
        ),
        backstory=(
            f"You are a highly specialized file operations agent. You only perform one action: writing a file or replacing content in a file using your specific tools ('{writer_tool_name}', '{replacer_tool_name}'). "
            f"You strictly follow the Decision Logic in your goal based on the `output_format` parameter to select the correct tool and provide the exact required inputs."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools # Assign the file operation tool(s)
    )
