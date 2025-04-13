# src/utils/parser_utils.py
import json
from logger_setup import get_logger

logger = get_logger(__name__)

STEP4_SEPARATOR = "--- JSON TASK LIST BELOW ---"

def parse_step4_output(output_string: str) -> tuple[str | None, list | None]:
    """
    Parses the combined Markdown strategy and JSON task list output from Step 4.

    Args:
        output_string (str): The raw string output from the DefineMappingTask.

    Returns:
        tuple[str | None, list | None]: A tuple containing:
            - The extracted Markdown strategy string (or None if separator not found).
            - The parsed JSON task list (or None if separator not found or JSON parsing fails).
    """
    logger.debug(f"Attempting to parse Step 4 output (length: {len(output_string)} chars).")

    if not isinstance(output_string, str) or not output_string.strip():
        logger.error("Received empty or non-string input for Step 4 parsing.")
        return None, None

    if STEP4_SEPARATOR not in output_string:
        logger.error(f"Step 4 output separator '{STEP4_SEPARATOR}' not found. Cannot parse.")
        # Return the whole string as strategy, assuming JSON failed? Or None?
        # Let's return None for both if separator is missing, indicating a format error.
        return None, None

    try:
        parts = output_string.split(STEP4_SEPARATOR, 1)
        markdown_strategy = parts[0].strip()
        json_part = parts[1].strip()

        # Clean potential markdown fences around the JSON part
        if json_part.startswith("```json"):
            json_part = json_part[7:]
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()

        if not json_part:
             logger.error("JSON part of Step 4 output is empty after stripping separator/fences.")
             return markdown_strategy, None # Return strategy, but None for JSON

        # Parse the JSON part
        parsed_task_list = json.loads(json_part)

        # Basic validation: Check if it's a list
        if not isinstance(parsed_task_list, list):
             logger.error(f"Parsed JSON task list is not a list (type: {type(parsed_task_list)}).")
             # Maybe add more validation here later (check structure of items)
             return markdown_strategy, None # Return strategy, but None for JSON

        logger.info("Successfully parsed Step 4 Markdown strategy and JSON task list.")
        return markdown_strategy, parsed_task_list

    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to parse JSON task list from Step 4 output: {json_err}")
        logger.debug(f"Problematic JSON string part: {json_part}")
        # Return the strategy part even if JSON parsing fails
        return markdown_strategy, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Step 4 output parsing: {e}", exc_info=True)
        return None, None # Return None for both on unexpected errors
