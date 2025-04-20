# src/agents/code_processor.py
from crewai import Agent, BaseLLM
# Removed tool imports: FileReadTool, FileWriteTool, validate_gdscript_syntax, replace_content_in_file
from src.logger_setup import get_logger
import src.config as config

logger = get_logger(__name__)


class CodeProcessorAgent:
    """
    CrewAI Agent responsible for processing a single conversion task item.
    It generates/modifies Godot code based on context and task details,
    determines the output format (full file or code block), and returns
    a structured JSON report for the Orchestrator to handle file operations and validation.
    """
    def __init__(self):
        logger.info(f"Initializing CodeProcessorAgent (LLM instance will be provided)")

    def get_agent(self, llm_instance: BaseLLM = None):
        """
        Creates and returns the CrewAI Agent instance.

        Args:
            llm_instance: An optional pre-configured LLM instance to use.
        """
        # No tools are passed to the agent itself.
        return Agent(
            role=f"SOLID-Focused C++ to {config.TARGET_LANGUAGE} Code Translator",
            goal=(
                f"Process a **single** conversion task item provided via context. The task details (description, target file/element, source C++/elements, mapping notes) and relevant context (C++ code, potentially existing Godot code) are given.\n"
                f"Your goal is to generate clean, idiomatic, and syntactically plausible {config.TARGET_LANGUAGE} code for this single task, adhering to SOLID principles.\n"
                f"1. Analyze the task details and context (including any existing Godot code provided).\n"
                f"2. Generate the required {config.TARGET_LANGUAGE} code based on the task, context, and mapping notes.\n"
                f"3. **Determine Output Format:** Decide if the generated code represents a 'FULL_FILE' (for new files or complete overwrites) or a 'CODE_BLOCK' (for modifying existing files).\n"
                f"4. **Extract Search Block (if modifying):** If the output format is 'CODE_BLOCK', you MUST extract the exact original code block (`search_block`) from the existing Godot code provided in the context that the `generated_code` should replace.\n"
                f"5. **Report Result:** Structure your final output as a **single JSON object** containing:\n"
                f"   - `task_id`: (string) The ID from the input task item.\n"
                f"   - `status`: (string) 'completed' if code generation was successful, 'failed' otherwise.\n"
                f"   - `output_format`: (string) 'FULL_FILE' or 'CODE_BLOCK'.\n"
                f"   - `generated_code`: (string) The generated {config.TARGET_LANGUAGE} code snippet or full file content.\n"
                f"   - `search_block`: (string | null) The exact original code block to search for if `output_format` is 'CODE_BLOCK', otherwise null. **Must be accurate!**\n"
                f"   - `target_godot_file`: (string) The target file path from the task item.\n"
                f"   - `target_element`: (string) The target element from the task item.\n"
                f"   - `validation_status`: (string) Optional: Indicate 'attempted_fix' if internal validation/fixing was tried, otherwise 'not_validated' or 'success' if confident. The orchestrator performs the definitive validation.\n"
                f"   - `error_message`: (string | null) Description of any error during code generation or analysis."
            ),
            backstory=(
                f"You are a meticulous programmer specialized in translating C++ code into idiomatic {config.TARGET_LANGUAGE} for Godot Engine 4.x, focusing on clean code and SOLID principles. "
                f"You follow instructions precisely for a single task, referencing provided context (including existing target code if available). "
                f"You generate clean, functional {config.TARGET_LANGUAGE} code. You determine if the output is a full file or a modification block. If it's a modification, you carefully extract the exact code block to be replaced (`search_block`) from the provided context. "
                f"You **do not** interact with the file system directly. You report the generated code and necessary metadata (like `output_format` and `search_block`) in a structured JSON format for another process (the Orchestrator) to handle file writing, replacement, and final validation."
            ),
            llm=llm_instance,
            verbose=True,
            allow_delegation=False, # Focuses on executing the defined tasks
            tools=[] # Agent does not use external tools directly
        )
