# src/tasks/step4/define_strategy_task.py
from crewai import Task, Agent # Added Agent for type hinting
from typing import Dict, Any, Optional # Added Optional for existing_mapping
from src.logger_setup import get_logger

logger = get_logger(__name__)

def create_define_strategy_task(
    agent: Agent, 
    cpp_analysis: str, 
    godot_analysis: str, 
    package_id: str, 
    existing_mapping: Optional[str] = None,
    feedback: Optional[str] = None # Added feedback parameter
) -> Task:
    """
    Creates a task for defining the C++ to Godot conversion strategy.

    Args:
        agent: The ConversionStrategistAgent that will perform this task.
        cpp_analysis: The textual analysis output from the CppCodeAnalystAgent.
        godot_analysis: The textual analysis output from the GodotStructureAnalystAgent.
        package_id: The ID of the work package being processed.
        existing_mapping: (Optional) JSON string of a previous mapping definition for refinement.
        feedback: (Optional) Textual feedback on a previous mapping attempt.

    Returns:
        A CrewAI Task object.
    """
    logger.info(f"Creating DefineConversionStrategyTask for agent {agent.role} (Package: {package_id})")
    
    # Base context includes C++ and Godot analyses
    context_parts = [
        f"**Task: Define High-Level Conversion Strategy for Package '{package_id}'**\n\n"
        f"**Objective:**\n"
        f"Based on the provided analyses of the C++ source code and the proposed Godot project structure, "
        f"develop a concise, high-level strategy for converting the C++ functionalities to Godot. "
        f"Your strategy should outline how key C++ concepts (classes, systems, data flow) will map to "
        f"Godot patterns (nodes, scenes, scripts, resources, signals, etc.) within the proposed Godot architecture.\n\n"
        f"--- C++ CODE ANALYSIS (Input) ---\n{cpp_analysis}\n--- END C++ ANALYSIS ---\n\n"
        f"--- GODOT STRUCTURE ANALYSIS (Input) ---\n{godot_analysis}\n--- END GODOT ANALYSIS ---"
    ]
    
    # Add existing mapping context if provided
    if existing_mapping:
        context_parts.append(
            f"\n\n--- EXISTING MAPPING DEFINITION (for refinement) ---\n"
            f"```json\n{existing_mapping}\n```\n"
            f"--- END EXISTING MAPPING ---"
        )
        context_parts.append(
            "\n**Refinement Note:** Review the existing mapping above. Your new strategy should aim to improve upon it, "
            "addressing any implied shortcomings or incorporating new insights from the analyses."
        )

    # Add feedback context if provided
    if feedback:
        context_parts.append(
            f"\n\n--- FEEDBACK ON PREVIOUS ATTEMPT ---\n{feedback}\n--- END FEEDBACK ---"
        )
        context_parts.append(
            "\n**Feedback Note:** Carefully consider the feedback provided above. Your strategy must address these points."
        )
        
    context_parts.append(
        f"\n\n**Your Strategy Should Explain:**\n"
        f"1.  How major C++ classes, systems, or modules will be represented in Godot (e.g., as specific Godot nodes, custom scenes, script-driven logic, or custom resources).\n"
        f"2.  Key Godot APIs, nodes, or features that should be leveraged (e.g., `CharacterBody3D`, `AnimationPlayer`, `SignalBus` singleton, custom `Resource` types).\n"
        f"3.  Major implementation patterns to follow (e.g., component-based design for entities, state machines for AI, event-driven communication via signals).\n"
        f"4.  If applicable, how the converted C++ logic will integrate with any existing Godot code or systems mentioned in the analyses.\n"
        f"5.  The overall approach to data management and persistence, if relevant from the C++ code.\n\n"
        f"The output should be a clear, well-reasoned mapping strategy description, typically 1-3 paragraphs, "
        f"focusing on the 'big picture' before detailed tasks are decomposed."
    )
    
    description = "\n".join(context_parts)
    
    expected_output = (
        "A clear, concise, and well-reasoned high-level mapping strategy description (1-3 paragraphs) that explains:\n"
        "1.  How C++ classes and systems will map to Godot nodes, scenes, and scripts.\n"
        "2.  Key Godot APIs or features to leverage (e.g., CharacterBody3D, Signals, Resources).\n"
        "3.  Major implementation patterns to follow for the conversion.\n"
        "4.  The approach for integrating with existing Godot code (if applicable).\n"
        "5.  Data management strategy if significant in the C++ source."
    )

    return Task(
        name=f"DefineConversionStrategy_{package_id}",
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False
        # The output of this task is a string (the strategy), which will be used by the TaskDecomposerAgent.
    )
