# src/utils/parser_utils.py
import json
import re # Added re
from typing import Tuple, Optional, List, Dict, Any, Union # Added typing imports
from src.logger_setup import get_logger
from src.tasks.define_mapping import MappingOutput # Import the Pydantic model

logger = get_logger(__name__)

# Separator used by the MappingDefinerAgent between Markdown and JSON
MAPPING_SEPARATOR = "--- JSON TASK LIST BELOW ---" # Renamed constant

def parse_step4_output(combined_output: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]: # Updated signature
    """
    Parses the combined output string from the MappingDefinerAgent (Step 4).

    Args:
        combined_output (str): The raw string output from the agent, expected
                               to contain Markdown followed by the separator,
                               followed by a JSON object conforming to MappingOutput.

    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]]]:
            - The extracted Markdown strategy (or None if parsing fails).
            - The parsed JSON mapping data as a dictionary (or None if parsing fails).
    """
    if not combined_output or not isinstance(combined_output, str):
        logger.error("Invalid input: combined_output must be a non-empty string.")
        return None, None

    if MAPPING_SEPARATOR not in combined_output: # Use new separator name
        logger.error(f"Parsing failed: Separator '{MAPPING_SEPARATOR}' not found in the output.")
        logger.debug(f"Full output received:\n{combined_output}")
        # Attempt to parse the whole thing as JSON in case the agent missed the separator but provided JSON
        json_data = parse_json_from_string(combined_output)
        if json_data and isinstance(json_data, dict):
             logger.warning("Separator missing, but parsed the full output as JSON. Returning JSON only.")
             return None, json_data
        return None, None # Truly failed

    try:
        parts = combined_output.split(MAPPING_SEPARATOR, 1)
        markdown_strategy = parts[0].strip()
        json_part_raw = parts[1].strip()

        # Clean potential markdown fences around the JSON part
        json_part_cleaned = re.sub(r'^```json\s*', '', json_part_raw, flags=re.IGNORECASE)
        json_part_cleaned = re.sub(r'\s*```$', '', json_part_cleaned)
        json_part_cleaned = json_part_cleaned.strip()

        if not markdown_strategy:
            logger.warning("Parsing warning: Markdown part is empty.")

        if not json_part_cleaned:
            logger.error("Parsing failed: JSON part is empty after cleaning.")
            return markdown_strategy or None, None # Return markdown if it exists

        # Parse the cleaned JSON string into a Python dictionary
        mapping_data_json = json.loads(json_part_cleaned)

        # Basic validation: Check if it's a dictionary
        if not isinstance(mapping_data_json, dict):
            logger.error(f"Parsing failed: Parsed JSON is not a dictionary (type: {type(mapping_data_json)}).")
            logger.debug(f"Cleaned JSON part:\n{json_part_cleaned}")
            return markdown_strategy or None, None

        # **Crucial Validation against Pydantic Model**
        try:
            MappingOutput(**mapping_data_json) # Validate structure and types
            logger.info("Successfully parsed Step 4 output: Found Markdown and valid structured JSON mapping data.")
            return markdown_strategy, mapping_data_json
        except Exception as pydantic_error:
            logger.error(f"Parsing failed: Parsed JSON does not conform to MappingOutput model. Error: {pydantic_error}")
            logger.debug(f"Parsed JSON data:\n{json.dumps(mapping_data_json, indent=2)}")
            # Return markdown part, but indicate JSON failure by returning None for the data
            return markdown_strategy or None, None

    except json.JSONDecodeError as e:
        logger.error(f"Parsing failed: Invalid JSON in the output. Error: {e}")
        logger.debug(f"Cleaned JSON part attempted to parse:\n{json_part_cleaned}")
        return markdown_strategy or None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Step 4 output parsing: {e}", exc_info=True)
        return markdown_strategy or None, None


def parse_json_from_string(text: str) -> Optional[Union[Dict, List]]:
    """
    Attempts to extract and parse a JSON object or list from a string,
    potentially cleaning markdown fences.
    """
    if not text:
        return None

    # Basic cleaning: remove markdown fences and strip whitespace
    cleaned_text = re.sub(r'^```json\s*', '', text.strip(), flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        return None

    try:
        # Find the first '{' or '[' to potentially trim leading garbage
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

        json_candidate = cleaned_text[start_index:]

        # Find the corresponding closing '}' or ']' - This is tricky and imperfect.
        # A simple last '}' or ']' might work for basic cases but fails with nesting.
        # For robustness, we rely on json.loads() to handle the structure.
        # We can try finding the last one as a heuristic for trimming trailing garbage.
        end_brace = json_candidate.rfind('}')
        end_bracket = json_candidate.rfind(']')
        end_index = max(end_brace, end_bracket)

        if end_index != -1:
             # Trim potential trailing garbage after the last brace/bracket
             # Add 1 because slicing is exclusive of the end index
             json_candidate_trimmed = json_candidate[:end_index + 1]
        else:
             # If no closing char found, maybe it's truncated? Try parsing anyway.
             json_candidate_trimmed = json_candidate


        # Attempt to parse the candidate string
        parsed_json = json.loads(json_candidate_trimmed)
        logger.debug("Successfully parsed JSON object/list from string.")
        return parsed_json

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode failed for candidate string: {e}")
        logger.debug(f"Candidate JSON string tried:\n{json_candidate_trimmed}")
        # Try parsing the original cleaned text as a fallback
        try:
             parsed_json_orig = json.loads(cleaned_text)
             logger.debug("Successfully parsed JSON from original cleaned text as fallback.")
             return parsed_json_orig
        except json.JSONDecodeError:
             logger.error("JSON decode failed for both candidate and original cleaned text.")
             return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing from string: {e}", exc_info=True)
        return None
