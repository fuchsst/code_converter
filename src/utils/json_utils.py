# src/utils/json_utils.py
import json
import re
from typing import Optional, Union, Dict, List, Any
from src.logger_setup import get_logger

logger = get_logger(__name__)

def _remove_trailing_commas(json_string: str) -> str:
    """Removes trailing commas from objects and arrays in a JSON string."""
    # Remove trailing commas before closing braces/brackets
    # Handles cases like { "key": "value", } and [ "item1", ]
    # Makes multiple passes to handle nested structures potentially
    cleaned_string = json_string
    for _ in range(5): # Limit passes to prevent infinite loops on weird input
        prev_string = cleaned_string
        # Remove trailing comma before }
        cleaned_string = re.sub(r",\s*}", "}", cleaned_string)
        # Remove trailing comma before ]
        cleaned_string = re.sub(r",\s*]", "]", cleaned_string)
        if cleaned_string == prev_string:
            break # No changes made in this pass
    if cleaned_string != json_string:
        logger.debug("Removed trailing commas from JSON string.")
    return cleaned_string

def parse_json_from_string(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Attempts to extract and parse a JSON object or list from a string.

    Handles potential markdown fences, leading/trailing garbage, and
    common LLM errors like trailing commas.

    Args:
        text: The raw string potentially containing JSON.

    Returns:
        The parsed JSON object (dict or list), or None if extraction or parsing fails.
    """
    if not text or not isinstance(text, str):
        logger.warning(f"Input to parse_json_from_string was empty or not a string (type: {type(text)}).")
        return None

    # 1. Basic cleaning: remove markdown fences and strip whitespace
    cleaned_text = re.sub(r'^```json\s*', '', text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        logger.debug("String is empty after removing markdown fences.")
        return None

    # 2. Find the start of the first JSON object or array
    start_brace = cleaned_text.find('{')
    start_bracket = cleaned_text.find('[')

    start_index = -1
    if start_brace == -1 and start_bracket == -1:
        logger.debug("No JSON start character ({ or [) found in cleaned text.")
        return None # No JSON object/list start found

    if start_brace != -1 and start_bracket != -1:
        start_index = min(start_brace, start_bracket)
    elif start_brace != -1:
        start_index = start_brace
    else: # start_bracket != -1
        start_index = start_bracket

    # 3. Find the end of the last JSON object or array
    # Find the corresponding closing '}' or ']' - This is tricky and imperfect.
    # A simple last '}' or ']' might work for basic cases but fails with nesting.
    # We rely on json.loads() to handle the structure, but try trimming trailing garbage.
    end_brace = cleaned_text.rfind('}')
    end_bracket = cleaned_text.rfind(']')
    end_index = max(end_brace, end_bracket)

    if end_index == -1 or end_index < start_index:
        # No closing character found, or it's before the opening one.
        # Try parsing from the start index onwards.
        json_candidate = cleaned_text[start_index:]
        logger.debug("No valid closing brace/bracket found after start, using text from start index.")
    else:
        # Extract the substring from the first opening char to the last closing char
        json_candidate = cleaned_text[start_index : end_index + 1]
        logger.debug("Extracted JSON candidate between first opening and last closing brace/bracket.")

    # 4. Attempt to remove trailing commas (common LLM error)
    json_candidate_no_commas = _remove_trailing_commas(json_candidate)

    # 5. Attempt to parse the cleaned candidate string
    try:
        parsed_json = json.loads(json_candidate_no_commas)
        logger.debug("Successfully parsed JSON object/list from cleaned candidate string.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed for cleaned candidate string: {e}")
        logger.debug(f"Cleaned candidate JSON string tried:\n{json_candidate_no_commas[:500]}...") # Log start

        # Fallback 1: Try parsing the original cleaned text (without start/end trimming)
        # after removing trailing commas
        cleaned_text_no_commas = _remove_trailing_commas(cleaned_text)
        try:
            parsed_json_orig = json.loads(cleaned_text_no_commas)
            logger.debug("Successfully parsed JSON from original cleaned text (no commas) as fallback.")
            return parsed_json_orig
        except json.JSONDecodeError:
            logger.error("JSON decode failed for both candidate and original cleaned text (even after removing trailing commas).")
            logger.debug(f"Original cleaned text tried:\n{cleaned_text_no_commas[:500]}...")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        return None
