# src/utils/parser_utils.py
import json
import re
from typing import Tuple, Optional, Dict, Any
from src.logger_setup import get_logger
from src.tasks.step4.define_mapping import MappingOutput
from src.utils.json_utils import parse_json_from_string

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
             logger.warning("Separator missing, but parsed the full output as JSON dictionary. Returning JSON only.")
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

        # Use the consolidated parser from json_utils
        mapping_data_json = parse_json_from_string(json_part_cleaned)

        if mapping_data_json is None:
            logger.error("Parsing failed: Consolidated parser could not extract valid JSON from the JSON part.")
            logger.debug(f"Cleaned JSON part provided to parser:\n{json_part_cleaned}")
            return markdown_strategy or None, None

        # Basic validation: Check if it's a dictionary (expected for MappingOutput)
        if not isinstance(mapping_data_json, dict):
            logger.error(f"Parsing failed: Parsed JSON is not a dictionary (type: {type(mapping_data_json)}). Expected for MappingOutput.")
            logger.debug(f"Parsed JSON data:\n{json.dumps(mapping_data_json, indent=2)}")
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

    except Exception as e:
        logger.error(f"An unexpected error occurred during Step 4 output parsing: {e}", exc_info=True)
        # Attempt to return markdown part if available even on unexpected errors
        markdown_part = locals().get('markdown_strategy', None)
        return markdown_part, None
