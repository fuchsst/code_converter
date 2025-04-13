# core/api_utils.py
import google.generativeai as genai
import google.api_core.exceptions
import time
import json
from logger_setup import get_logger
import config

logger = get_logger(__name__)

# Configure the Gemini client globally or pass it around
# Using global configuration for simplicity here, ensure API key is loaded
try:
    if config.GEMINI_API_KEY:
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("Gemini API configured successfully.")
    else:
        logger.error("GEMINI_API_KEY is not set. API calls will fail.")
        # Optionally raise an error here to prevent proceeding without a key
        # raise ValueError("GEMINI_API_KEY is not configured.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)

def call_gemini_api(prompt: str, model_name: str, expect_json: bool = False, retry_count: int = config.MAX_RETRIES):
    """
    Calls the Gemini API with the given prompt and model, handling retries and errors.

    Args:
        prompt (str): The prompt to send to the model.
        model_name (str): The name of the Gemini model to use (e.g., 'gemini-1.5-flash-latest').
        expect_json (bool): If True, attempts to parse the response as JSON.
        retry_count (int): The number of times to retry on failure.

    Returns:
        Union[str, dict, None]: The model's response text, parsed JSON dictionary, or None if failed.
    """
    if not config.GEMINI_API_KEY:
        logger.error("Cannot call Gemini API: API key not configured.")
        return None

    model = genai.GenerativeModel(model_name)
    current_retry = 0
    delay = config.INITIAL_BACKOFF_DELAY

    while current_retry <= retry_count:
        try:
            logger.debug(f"Attempt {current_retry + 1}/{retry_count + 1}: Calling Gemini model {model_name}")
            # TODO: Add generation config (temperature, safety settings etc.) from config.py if needed
            generation_config = genai.types.GenerationConfig(
                # temperature=config.DEFAULT_TEMPERATURE, # Example
                # Add other relevant configs
            )
            if expect_json:
                generation_config.response_mime_type = "application/json"

            response = model.generate_content(
                prompt,
                generation_config=generation_config
                # safety_settings=... # Add safety settings if needed
            )

            # Check for blocked content or empty response
            if not response.candidates:
                 prompt_feedback = response.prompt_feedback
                 block_reason = prompt_feedback.block_reason if prompt_feedback else "Unknown"
                 block_message = prompt_feedback.block_reason_message if prompt_feedback else "No candidates returned"
                 logger.warning(f"Gemini API call blocked or returned no candidates. Reason: {block_reason}. Message: {block_message}")
                 # Decide if this is retryable - safety blocks usually aren't
                 if block_reason == 'SAFETY':
                     logger.error("Content blocked due to safety settings. Not retrying.")
                     return None # Or raise a specific exception
                 # Other reasons might be retryable, continue loop for now

            response_text = response.text
            logger.debug(f"Gemini API call successful. Response received.")

            if expect_json:
                try:
                    # Clean potential markdown code block fences
                    if response_text.strip().startswith("```json"):
                        response_text = response_text.strip()[7:-3].strip()
                    elif response_text.strip().startswith("```"):
                         response_text = response_text.strip()[3:-3].strip()

                    parsed_json = json.loads(response_text)
                    logger.info("Successfully parsed JSON response.")
                    return parsed_json
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse JSON response: {json_err}")
                    logger.debug(f"Raw response text: {response_text}")
                    # Treat as failure, maybe retry if it looks like a transient formatting issue
                    # For now, we'll let the retry logic handle it if it's not the last attempt
                    if current_retry == retry_count:
                        return None # Failed after all retries
                    # Fall through to retry logic
                except Exception as parse_err:
                     logger.error(f"Unexpected error parsing JSON response: {parse_err}", exc_info=True)
                     if current_retry == retry_count:
                         return None
                     # Fall through to retry logic

            else:
                # Return raw text if JSON wasn't expected
                return response_text

        except google.api_core.exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit (ResourceExhausted). Retrying in {delay:.2f} seconds... ({e})")
        except google.api_core.exceptions.RetryError as e:
             logger.warning(f"RetryError encountered. Retrying in {delay:.2f} seconds... ({e})")
        except google.api_core.exceptions.ServiceUnavailable as e:
            logger.warning(f"Service unavailable. Retrying in {delay:.2f} seconds... ({e})")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Gemini API call: {e}", exc_info=True)
            # Decide if this error is retryable or fatal
            if current_retry == retry_count:
                logger.error("Max retries reached after unexpected error.")
                return None # Failed after all retries
            # Fall through to retry logic for potentially transient errors

        # If we reached here, it means an error occurred and we should retry
        current_retry += 1
        if current_retry <= retry_count:
            time.sleep(delay)
            delay = min(delay * 2, config.MAX_BACKOFF_DELAY) # Exponential backoff
            # Add jitter? delay += random.uniform(0, delay * 0.1)

    logger.error(f"Gemini API call failed after {retry_count + 1} attempts.")
    return None

def get_gemini_model(model_name: str) -> genai.GenerativeModel | None:
    """
    Retrieves a configured GenerativeModel instance for the given model name.

    Args:
        model_name (str): The name of the Gemini model (e.g., 'gemini-1.5-flash-latest').

    Returns:
        genai.GenerativeModel | None: The configured model instance or None on failure.
    """
    if not config.GEMINI_API_KEY:
        logger.error(f"Cannot get Gemini model '{model_name}': API key not configured.")
        return None
    try:
        # Ensure API key is configured (might be redundant if called after global configure)
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name)
        logger.info(f"Retrieved configured Gemini model instance: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to get Gemini model instance '{model_name}': {e}", exc_info=True)
        return None
