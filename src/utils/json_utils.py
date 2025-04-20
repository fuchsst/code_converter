# src/utils/json_utils.py
import json
from src.logger_setup import get_logger
from typing import Optional

logger = get_logger(__name__)

def extract_json_block(text: str) -> Optional[str]:
    """
    Extracts the first JSON block (from first '{' to last '}') found in a string.

    Handles cases where the JSON might be embedded within other text or markdown fences.

    Args:
        text: The raw string potentially containing a JSON block.

    Returns:
        The extracted JSON block as a string, or None if a valid block
        (starting with '{' and ending with '}') is not found.
    """
    if not isinstance(text, str):
        logger.warning(f"Input to extract_json_block was not a string (type: {type(text)}).")
        return None

    try:
        # Find the start of the first JSON object
        json_start_index = text.find('{')
        if json_start_index == -1:
            logger.debug("Could not find starting '{' in the text.")
            return None

        # Find the end of the last JSON object
        json_end_index = text.rfind('}')
        if json_end_index == -1:
            logger.debug("Could not find ending '}' in the text.")
            return None

        # Ensure the closing brace comes after the opening brace
        if json_end_index < json_start_index:
            logger.debug(f"Found '}}' before '{{' (start: {json_start_index}, end: {json_end_index}). Invalid structure.")
            return None

        # Extract the potential JSON block
        json_string_block = text[json_start_index : json_end_index + 1]
        logger.debug(f"Extracted potential JSON block (length: {len(json_string_block)} chars).")
        return json_string_block

    except Exception as e:
        # Catch potential errors during string manipulation, though unlikely here
        logger.error(f"Unexpected error during JSON block extraction: {e}", exc_info=True)
        return None

def parse_json_from_string(text: str) -> Optional[dict | list]:
    """
    Attempts to extract and parse a JSON block from a string.

    Args:
        text: The raw string potentially containing JSON.

    Returns:
        The parsed JSON object (dict or list), or None if extraction or parsing fails.
    """
    json_block = extract_json_block(text)
    if json_block is None:
        logger.warning("Could not extract a JSON block from the input string.")
        return None

    try:
        parsed_json = json.loads(json_block)
        logger.debug("Successfully parsed extracted JSON block.")
        return parsed_json
    except json.JSONDecodeError as json_err:
        logger.warning(f"Failed to parse extracted JSON block: {json_err}. Block: '{json_block[:200]}...'") # Log start of block
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        return None
