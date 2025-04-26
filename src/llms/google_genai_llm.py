# src/llms/google_genai_llm.py
import os
import time
from typing import Any, Dict, List, Optional, Union, cast, Callable
import json

from google import genai
from typing import Tuple
from google.genai.errors import APIError # Import Google API error class
from google.genai.types import (
    GenerationConfig, GenerateContentConfig, ContentDict, PartDict, # Added GenerateContentConfig
    Tool, ThinkingConfig, FunctionDeclaration, GoogleSearch
)

from pydantic import BaseModel, RootModel # Import Pydantic base classes
from crewai.llms.base_llm import BaseLLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

# Mapping from CrewAI roles to Gemini roles
ROLE_MAP = {"agent": "model", "user": "user", "tool": "function"}

class GoogleGenAI_LLM(BaseLLM):
    """
    Custom CrewAI LLM wrapper for Google Generative AI (Gemini) models.
    Uses the official google-generativeai SDK directly via the Client interface.
    """
    model_name: str # Store the target model name (e.g., gemini-1.5-pro-latest)
    api_key: Optional[str] = None
    generation_config_dict: Optional[Dict[str, Any]] = None
    safety_settings_dict: Optional[List[Dict[str, Any]]] = None
    request_timeout: Optional[int] = None # Timeout in seconds
    _client_instance: Optional[genai.Client] = None # Store the client instance

    def __init__(
        self,
        model: str, # Expects format like "gemini/gemini-1.5-pro-latest" or just "gemini-1.5-pro-latest"
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        response_schema: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any
    ):
        # Store the potentially prefixed model name first
        raw_model_name = model
        # Extract base model name if prefixed (used for logging/display)
        if "/" in model:
            self.model_name = model.split('/')[-1]
        else:
            self.model_name = model

        # Pass the *base* model name to BaseLLM if it expects it,
        # but we'll use the raw_model_name for the actual API calls if needed by the SDK client.
        # For google-genai, the generate_content method takes the model name directly.
        super().__init__(model=self.model_name, temperature=temperature)

        # Store API key if provided, otherwise rely on environment variable
        self.api_key = api_key # Will be used by _get_client

        # Store schema if provided
        self.response_schema = response_schema

        # Build GenerationConfig dictionary
        gen_config_params = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
                "stop_sequences": stop_sequences,
        }
        if self.response_schema:
            gen_config_params["response_schema"] = self.response_schema
        # Thinking config will be added dynamically in call() if enabled

        self.generation_config_dict = {
            key: value for key, value in gen_config_params.items() if value is not None
        }

        # Handle safety settings
        self.safety_settings_dict = kwargs.get("safety_settings")

        # Set timeout
        default_timeout = 600
        try:
            env_timeout = os.getenv("GEMINI_TIMEOUT")
            self.request_timeout = timeout if timeout is not None else (int(env_timeout) if env_timeout else default_timeout)
        except ValueError:
            logger.warning(f"Invalid GEMINI_TIMEOUT value '{env_timeout}'. Using default: {default_timeout}s")
            self.request_timeout = default_timeout

        logger.info(f"Initialized GoogleGenAI_LLM wrapper for model: {self.model_name} (raw: {raw_model_name})")
        # Log base config, thinking/grounding flags will be logged in call()
        logger.debug(f"Base Generation Config: {self.generation_config_dict}")
        logger.debug(f"Safety Settings: {self.safety_settings_dict}")
        logger.debug(f"Request Timeout: {self.request_timeout}s")

    def _get_client(self) -> genai.Client:
        """Initializes and returns the google-generativeai client instance."""
        if self._client_instance is None:
            try:
                # Determine API key source
                current_api_key = self.api_key or os.getenv("GEMINI_API_KEY")
                if not current_api_key:
                    raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor.")

                # Instantiate the client, passing the API key directly
                # TODO: Add support for Vertex AI client options if needed later
                self._client_instance = genai.Client(api_key=current_api_key)
                if self.api_key:
                    logger.debug("Instantiated google.generativeai Client with provided API key.")
                else:
                    logger.debug("Instantiated google.generativeai Client with API key from environment variable.")

            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI Client: {e}", exc_info=True)
                raise RuntimeError(f"Could not initialize Google Generative AI Client: {e}") from e
        return self._client_instance

    def _convert_messages(self, messages: Union[str, List[Dict[str, str]]]) -> Tuple[List[ContentDict], Optional[ContentDict]]:
        """Converts CrewAI message format to google-generativeai format."""
        system_instruction: Optional[ContentDict] = None
        converted_messages: List[ContentDict] = []

        if isinstance(messages, str):
            converted_messages.append({"role": "user", "parts": [{"text": messages}]})
            return converted_messages, system_instruction

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                logger.warning(f"Skipping invalid message format: {msg}")
                continue
            role_lower = role.lower()
            if role_lower == "system":
                if system_instruction:
                    logger.warning("Multiple system messages found. Using the last one.")
                system_instruction = {"parts": [{"text": content}]}
                continue
            gemini_role = ROLE_MAP.get(role_lower)
            if not gemini_role:
                logger.warning(f"Unsupported role '{role}' found. Mapping to 'user'. Message: {content[:50]}...")
                gemini_role = "user"
            parts: List[PartDict] = [{"text": content}]
            converted_messages.append({"role": gemini_role, "parts": parts})
        return converted_messages, system_instruction

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Union[Dict, Callable]]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Calls the Google Generative AI model using the Client."""
        client = self._get_client()
        converted_messages, system_instruction_content = self._convert_messages(messages)

        # --- Prepare Tools ---
        final_tools: List[Tool] = []
        if tools:
            for tool_input in tools:
                try:
                    if isinstance(tool_input, dict) and 'function' in tool_input:
                        func_decl = FunctionDeclaration(**tool_input['function'])
                        final_tools.append(Tool(function_declarations=[func_decl]))
                    elif callable(tool_input):
                         logger.warning(f"Callable tools are not directly supported. Skipping tool: {tool_input.__name__}")
                    else:
                        logger.warning(f"Unsupported tool format encountered: {type(tool_input)}. Skipping.")
                except Exception as e:
                    logger.warning(f"Failed to convert provided tool to Gemini format: {e}. Tool: {tool_input}")
        genai_tools_for_call = final_tools if final_tools else None

        # --- Prepare Combined Config using GenerateContentConfig ---
        combined_config_params = self.generation_config_dict.copy() if self.generation_config_dict else {}
        if system_instruction_content:
            combined_config_params["system_instruction"] = system_instruction_content
            logger.debug("Adding system_instruction to GenerateContentConfig parameters")
        if genai_tools_for_call:
             combined_config_params["tools"] = genai_tools_for_call # Add tools here
             logger.debug("Adding tools to GenerateContentConfig parameters")

        # Instantiate GenerateContentConfig if there are any params
        final_config_obj = GenerateContentConfig(**combined_config_params) if combined_config_params else None



        logger.debug(f"Calling Gemini model {self.model_name} via Client with {len(converted_messages)} messages.")
        logger.debug(f"Final Combined Config Obj for call: {final_config_obj}")

        try:
            logger.info(f"Attempting to call client.models.generate_content for model: {self.model_name}") # ADDED LOG
            # Use client.models.generate_content with combined config
            response = client.models.generate_content(
                model=self.model_name,
                contents=converted_messages,
                config=final_config_obj
            )
            logger.info(f"Received response from client.models.generate_content for model: {self.model_name}") # ADDED LOG

            # --- Response Handling ---
            if not response.candidates:
                 feedback = response.prompt_feedback
                 block_reason = feedback.block_reason.name if feedback.block_reason else "Unknown"
                 logger.error(f"Gemini call blocked. Reason: {block_reason}. Feedback: {feedback}")
                 return f"Error: Call blocked by API. Reason: {block_reason}"

            first_candidate = response.candidates[0]

            # 1. Check for function calls
            if first_candidate.content and first_candidate.content.parts:
                function_calls = [part.function_call for part in first_candidate.content.parts if hasattr(part, 'function_call') and part.function_call]
                if function_calls:
                    first_call = function_calls[0]
                    args_dict = dict(first_call.args) if hasattr(first_call.args, 'items') else {}
                    tool_call_dict = {"name": first_call.name, "arguments": args_dict}
                    tool_call_json = json.dumps(tool_call_dict)
                    logger.info(f"Detected function call: {first_call.name}. Returning JSON.")
                    logger.debug(f"Function call JSON for CrewAI: {tool_call_json}")
                    return tool_call_json

            # 2. Check for direct text response
            if hasattr(response, 'text') and response.text:
                logger.debug("Using response.text for output.")
                return response.text

            # 3. Fallback to concatenating text parts
            if first_candidate.content and first_candidate.content.parts:
                text_response = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text') and part.text)
                if text_response:
                    logger.debug("Using concatenated text from response parts.")
                    return text_response

            # 4. Handle empty/unexpected response
            logger.warning(f"Gemini response candidate did not contain expected text parts, function calls, or response.text. Response: {response}")
            return ""

        except APIError as e:
            logger.error(f"Google API error during Gemini call: {e}", exc_info=True)
            raise RuntimeError(f"Google API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during Gemini call: {e}") from e

    def get_context_window_size(self) -> int:
        """Retrieves the context window size using the Client."""
        default_size = 32_000
        model_identifier = f'models/{self.model_name}' # Use the models/ prefix
        try:
            client = self._get_client() # Ensures client and API key are configured
            model_info = client.models.get(model=model_identifier) # Use client.models.get

            if model_info and hasattr(model_info, 'input_token_limit'):
                logger.debug(f"Retrieved input_token_limit for {model_identifier}: {model_info.input_token_limit}")
                return model_info.input_token_limit
            else:
                logger.warning(f"Could not retrieve input_token_limit for model {model_identifier}. Response: {model_info}. Defaulting to {default_size}.")
                return default_size
        except APIError as e:
             logger.warning(f"Google API error retrieving model info for {model_identifier}: {e}. Defaulting to {default_size}.")
             return default_size
        except Exception as e:
            logger.warning(f"Failed to retrieve context window size for model {model_identifier} via API: {e}. Defaulting to {default_size}.")
            return default_size

    # Override other methods if needed
    def supports_function_calling(self) -> bool:
        return False
