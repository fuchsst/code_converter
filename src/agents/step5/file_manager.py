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
    tool_names = [t.name for t in tools if hasattr(t, 'name')]
    return Agent(
        role="File System Operations Specialist",
        goal=(
            f"Perform file system operations based on instructions using the provided tools ({', '.join(tool_names)}). "
            f"Receive instructions specifying the operation (write or replace), the target file path, "
            f"the content to write, and potentially a search block for replacement. "
            f"Execute the correct tool ('{tools[0].name if len(tools)>0 else 'File Writer'}' or "
            f"'{tools[1].name if len(tools)>1 else 'File Replacer'}') with the exact parameters provided in the context/task description. "
            f"Report the outcome message returned by the tool."
        ),
        backstory=(
            f"You are a reliable assistant responsible for interacting with the file system via specific, approved tools. "
            f"You follow instructions precisely to either write content to a new file or replace a specific block of content in an existing file. "
            f"You understand the importance of using the correct tool and providing the exact file path, content, and search block (if required). "
            f"You do not generate code or make decisions about *what* to write; you only execute the requested file operation using your tools and report the result."
        ),
        llm=llm_instance,
        verbose=True,
        allow_delegation=False,
        tools=tools # Assign the file operation tool(s)
    )
