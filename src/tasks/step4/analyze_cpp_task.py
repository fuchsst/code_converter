# src/tasks/step4/analyze_cpp_task.py
from crewai import Task, Agent # Added Agent for type hinting
from typing import Dict, Any # Added for type hinting if context were a dict
from src.logger_setup import get_logger

logger = get_logger(__name__)

def create_analyze_cpp_task(agent: Agent, context: str, package_id: str) -> Task:
    """
    Creates a task for analyzing C++ source code of a specific package.

    Args:
        agent: The CppCodeAnalystAgent that will perform this task.
        context: A string containing the C++ source code, file list, and roles
                 for the specified package.
        package_id: The ID of the work package being processed.

    Returns:
        A CrewAI Task object.
    """
    logger.info(f"Creating AnalyzeCppTask for agent {agent.role} (Package: {package_id})")
    
    description = (
        f"**Task: Analyze C++ Source Code for Package '{package_id}'**\n\n"
        f"**Objective:**\n"
        f"Thoroughly analyze the provided C++ source code, including the list of files and their designated roles. "
        f"Your goal is to understand the purpose, structure, key classes, functions, data structures, "
        f"logic flow, and overall architecture of the C++ code within this package. "
        f"Focus on extracting information that will be crucial for planning its conversion to Godot Engine.\n\n"
        f"**Provided C++ Context for Package '{package_id}':**\n"
        f"```cpp\n{context}\n```\n" # Assuming context includes code and file list as formatted by CppCodeAnalysisTool
        f"--- END OF PROVIDED CONTEXT ---\n\n"
        f"**Your Analysis Should Identify and Summarize:**\n"
        f"1.  **Overall Purpose:** What is the primary function or responsibility of this C++ package/module?\n"
        f"2.  **Key Classes/Structs:** List the most important classes and data structures, briefly describing their roles.\n"
        f"3.  **Main Functions/Methods:** Identify critical functions or methods and their core logic.\n"
        f"4.  **Logic Flow & Algorithms:** Describe any significant algorithms or complex logic sequences.\n"
        f"5.  **Data Management:** How is data stored, accessed, and manipulated within this code?\n"
        f"6.  **Dependencies:** Note any apparent internal dependencies within the package or (if discernible) external dependencies this code might imply for a game engine context.\n"
        f"7.  **Suitability for Conversion:** Briefly assess which parts seem straightforward to map to Godot and which might require more complex translation or redesign.\n\n"
        f"Your output should be a comprehensive, structured textual analysis. This analysis will be used by other agents to define a conversion strategy and decompose tasks."
    )
    
    expected_output = (
        "A detailed and structured textual analysis of the C++ code, covering:\n"
        "1.  Overall purpose of the package.\n"
        "2.  Key classes/structs and their roles.\n"
        "3.  Main functions/methods and their core logic.\n"
        "4.  Significant algorithms or logic flows.\n"
        "5.  Data management approaches.\n"
        "6.  Apparent dependencies.\n"
        "7.  Initial thoughts on conversion complexity/approach for different parts."
    )

    return Task(
        name=f"AnalyzeCppCode_{package_id}", # Unique task name per package
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False
        # This task consumes C++ code context and produces textual analysis.
        # Its output is vital for the ConversionStrategistAgent.
    )
