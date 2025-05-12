# src/llms/litellm_gemini_llm.py
import os
import json
from typing import Any, Dict, List, Optional, Union, Callable

# Use try-except for robustness
try:
    import litellm
    from litellm import completion as litellm_completion
    from litellm.exceptions import APIError as LiteLLMAPIError
except ImportError:
    raise ImportError("LiteLLM is required to use the LiteLLMGeminiLLM wrapper. Please install it using 'pip install litellm'")

from crewai.llms.base_llm import BaseLLM
from src.logger_setup import get_logger
import src.config as config # For defaults like timeout, max_tokens

logger = get_logger(__name__)

# Mapping from CrewAI roles to OpenAI/LiteLLM roles (usually consistent)
# LiteLLM generally expects 'system', 'user', 'assistant', 'tool'
ROLE_MAP = {"agent": "assistant", "user": "user", "tool": "tool", "system": "system"}

class LiteLLMGeminiLLM(BaseLLM):
    """
    Custom CrewAI LLM wrapper for Google Gemini models via the LiteLLM library.
    Handles interaction using litellm.completion.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    model_name: str # Stores the full model identifier (e.g., "gemini/gemini-1.5-pro-latest")
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None # Corresponds to max_output_tokens in Gemini
    stop_sequences: Optional[List[str]] = None
    # Add other litellm supported parameters as needed
    request_timeout: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None # For JSON mode or schema
    safety_settings: Optional[List[Dict[str, Any]]] = None
    # Note: LiteLLM handles API key via environment variable (GEMINI_API_KEY) by default

    def __init__(
        self,
        model: str, # Expects format like "gemini/gemini-1.5-pro-latest"
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None, # Use max_tokens for litellm standard
        stop: Optional[List[str]] = None, # Use 'stop' for litellm standard
        response_format: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[int] = None,
        **kwargs: Any # Catch any other potential litellm params
    ):
        # Pass the full model name to BaseLLM and store it
        super().__init__(model=model, temperature=temperature)
        self.model_name = model # Keep the full name for litellm call

        # Store parameters using litellm standard names where applicable
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens if max_tokens is not None else config.MAX_OUTPUT_TOKENS
        self.stop_sequences = stop # Will be None if 'stop' not in kwargs to __init__
        self.stop = None # Explicitly set self.stop to None to override potential superclass defaults
        self.response_format = response_format
        self.safety_settings = safety_settings

        # Set timeout
        default_timeout = 600 # Or use global_config.GEMINI_TIMEOUT if defined
        try:
            env_timeout = os.getenv("GEMINI_TIMEOUT") # Check env var
            self.request_timeout = timeout if timeout is not None else (int(env_timeout) if env_timeout else default_timeout)
        except ValueError:
            logger.warning(f"Invalid GEMINI_TIMEOUT value '{env_timeout}'. Using default: {default_timeout}s")
            self.request_timeout = default_timeout

        # Store any extra kwargs for potential pass-through
        self.extra_params = kwargs

        logger.info(f"Initialized LiteLLMGeminiLLM wrapper for model: {self.model_name}")
        logger.debug(f"Config - Temp: {self.temperature}, TopP: {self.top_p}, MaxTokens: {self.max_tokens}, Stop: {self.stop_sequences}, Timeout: {self.request_timeout}, ResponseFormat: {self.response_format}, Safety: {self.safety_settings}, Extras: {self.extra_params}")

        # Check for API key in environment during init for early warning
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning("GEMINI_API_KEY environment variable not found. LiteLLM calls may fail.")


    def _convert_messages(self, messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Converts CrewAI message format to LiteLLM/OpenAI format."""
        converted_messages: List[Dict[str, Any]] = []

        if isinstance(messages, str):
            converted_messages.append({"role": "user", "content": messages})
            return converted_messages

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or content is None: # Allow empty content? Check litellm behavior
                logger.warning(f"Skipping message with missing role or content: {msg}")
                continue

            litellm_role = ROLE_MAP.get(role.lower())
            if not litellm_role:
                logger.warning(f"Unsupported role '{role}' found. Mapping to 'user'. Message: {str(content)[:50]}...")
                litellm_role = "user"

            # Handle potential non-string content if needed (e.g., tool calls might be dicts)
            # LiteLLM generally expects string content, but check for multimodal/tool formats
            message_dict = {"role": litellm_role, "content": content}

            # Add tool_call_id if present (for tool responses)
            if litellm_role == "tool" and isinstance(content, str):
                 # Attempt to parse content as JSON to extract tool_call_id if needed
                 # CrewAI might format tool results differently, adjust as necessary
                 try:
                      tool_data = json.loads(content)
                      if isinstance(tool_data, dict) and "tool_call_id" in tool_data:
                           message_dict["tool_call_id"] = tool_data["tool_call_id"]
                           # Potentially reformat content if litellm expects just the result string
                           message_dict["content"] = tool_data.get("result", content)
                 except json.JSONDecodeError:
                      pass # Keep content as is if not valid JSON

            converted_messages.append(message_dict)

        return converted_messages

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[Union[Dict, Callable]]] = None,
        callbacks: Optional[List[Any]] = None, # Callbacks might not be directly usable by litellm
        available_functions: Optional[Dict[str, Any]] = None, # Deprecated in favor of tools
    ) -> str:
        """Calls the Gemini model via LiteLLM."""
        converted_messages = self._convert_messages(messages)

        # --- Prepare LiteLLM parameters ---
        params = {
            "model": self.model_name,
            "messages": converted_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences,
            "timeout": self.request_timeout,
            "response_format": self.response_format,
            "safety_settings": self.safety_settings,
            **self.extra_params # Include any extra params passed during init
        }

        # --- Handle Tools ---
        # Convert CrewAI tool format (often OpenAI-like) to LiteLLM format if needed
        # LiteLLM generally accepts OpenAI tool format directly.
        if tools:
            # Basic validation/conversion if necessary, assume OpenAI format for now
            formatted_tools = []
            tool_choice = "auto" # Default tool choice
            for tool_input in tools:
                 if isinstance(tool_input, dict) and tool_input.get("type") == "function":
                      formatted_tools.append(tool_input)
                 # Check for CrewAI's specific tool structure if different
                 elif isinstance(tool_input, dict) and 'function' in tool_input and 'type' not in tool_input:
                      # Adapt CrewAI's older format if encountered
                      formatted_tools.append({"type": "function", "function": tool_input['function']})
                 elif callable(tool_input):
                      logger.warning(f"Callable tools are not directly supported by LiteLLM in this wrapper. Skipping tool: {tool_input.__name__}")
                 else:
                      logger.warning(f"Unsupported tool format encountered: {type(tool_input)}. Skipping.")

            if formatted_tools:
                params["tools"] = formatted_tools
                params["tool_choice"] = tool_choice # Let litellm/model decide
                logger.debug(f"Passing {len(formatted_tools)} tools to LiteLLM.")


        # Remove None values from params before calling litellm
        api_params = {key: value for key, value in params.items() if value is not None}

        logger.debug(f"Calling LiteLLM completion for model {self.model_name} with {len(converted_messages)} messages.")
        logger.debug(f"LiteLLM API Params: {api_params}") # Be cautious logging sensitive data

        try:
            response = litellm_completion(**api_params)
            logger.debug(f"Received response from LiteLLM for model: {self.model_name}")
            # logger.debug(f"LiteLLM Raw Response Object: {response}") # Detailed logging if needed

            # --- Response Handling ---
            if not response or not response.choices:
                logger.error("LiteLLM response is empty or missing choices.")
                return "Error: Received empty response from LLM."

            first_choice = response.choices[0]
            message = first_choice.message

            # 1. Check for tool calls (LiteLLM follows OpenAI structure)
            if message and message.tool_calls:
                # CrewAI expects a JSON string representing the tool call(s)
                # For simplicity, return the first tool call as JSON string
                # TODO: Handle multiple tool calls if CrewAI supports it
                first_tool_call = message.tool_calls[0]
                tool_call_dict = {
                    "id": first_tool_call.id, # Include ID for potential tool response mapping
                    "type": first_tool_call.type, # Usually 'function'
                    "function": {
                        "name": first_tool_call.function.name,
                        "arguments": first_tool_call.function.arguments # Arguments are already a string
                    }
                }
                tool_call_json = json.dumps(tool_call_dict)
                logger.info(f"Detected tool call: {first_tool_call.function.name}. Returning JSON.")
                logger.debug(f"Tool call JSON for CrewAI: {tool_call_json}")
                return tool_call_json

            # 2. Check for direct text content
            if message and message.content and isinstance(message.content, str):
                logger.debug("Using message.content for output.")
                return message.content

            # 3. Handle potential finish reasons (e.g., length, blocked)
            finish_reason = first_choice.finish_reason
            if finish_reason != "stop":
                 logger.warning(f"LiteLLM call finished with reason: {finish_reason}. Response content: {message.content if message else 'N/A'}")
                 # Return content even if finish reason isn't 'stop', or handle specific reasons
                 if message and message.content:
                      return message.content

            # 4. Handle empty/unexpected response content
            logger.warning(f"LiteLLM response choice did not contain expected text content or tool calls. Finish Reason: {finish_reason}. Response: {response}")
            return "" # Return empty string if no usable content

        except LiteLLMAPIError as e:
            logger.error(f"LiteLLM API error during Gemini call: {e}", exc_info=True)
            # Try to return a more informative error message
            error_message = f"Error: LiteLLM API Error - Status: {e.status_code}, Message: {e.message}"
            # Check for specific Gemini blocking reasons if available in the error details
            # This might require inspecting e.response or e.body depending on litellm version
            return error_message
        except Exception as e:
            logger.error(f"Unexpected error during LiteLLM Gemini call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during LiteLLM call: {e}") from e

    def get_context_window_size(self) -> int:
        """
        Retrieves the context window size using LiteLLM's model info.
        Falls back to a default if retrieval fails.
        """
        default_size = 32_000 # A common default, adjust if needed
        try:
            # LiteLLM provides model_cost which often includes context window
            model_info = litellm.get_model_info(self.model_name)
            if model_info and 'max_input_tokens' in model_info:
                size = model_info['max_input_tokens']
                logger.debug(f"Retrieved context window size for {self.model_name} via LiteLLM: {size}")
                return size
            else:
                 logger.warning(f"Could not retrieve 'max_input_tokens' for model {self.model_name} from LiteLLM info. Defaulting to {default_size}.")
                 return default_size
        except Exception as e:
            logger.warning(f"Failed to retrieve context window size for model {self.model_name} via LiteLLM: {e}. Defaulting to {default_size}.")
            return default_size

    def supports_function_calling(self) -> bool:
        """Indicates that this LLM wrapper supports tool calling via LiteLLM."""
        # Gemini via LiteLLM supports OpenAI-style tool calling
        return True
