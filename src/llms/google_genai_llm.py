# src/llms/google_genai_llm.py
import os
import time
from typing import Any, Dict, List, Optional, Union, cast

import google.generativeai as genai
from typing import Tuple
from google.generativeai.types import GenerationConfig, ContentDict, PartDict
from google.api_core.exceptions import GoogleAPIError

from crewai.llms.base_llm import BaseLLM
from src.logger_setup import get_logger

logger = get_logger(__name__)

# Mapping from CrewAI roles to Gemini roles
ROLE_MAP = {"agent": "model", "user": "user", "tool": "function"}

class GoogleGenAI_LLM(BaseLLM):
    """
    Custom CrewAI LLM wrapper for Google Generative AI (Gemini) models.
    Uses the official google-generativeai SDK directly.
    """
    model_name: str
    api_key: Optional[str] = None
    generation_config_dict: Optional[Dict[str, Any]] = None
    safety_settings_dict: Optional[List[Dict[str, Any]]] = None
    _model_instance: Optional[genai.GenerativeModel] = None

    def __init__(
        self,
        model: str, # Expects format like "gemini/gemini-1.5-pro-latest"
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None, # Note: google SDK uses float for top_k? Check docs. Let's assume int for now.
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        api_key: Optional[str] = None, # Allow explicit key passing
        **kwargs: Any # Catch other potential LiteLLM/CrewAI params
    ):
        # Extract model name without prefix for Google SDK
        if "/" in model:
            self.model_name = model.split('/')[-1]
        else:
            self.model_name = model # Assume it's already the correct format

        super().__init__(model=self.model_name, temperature=temperature) # Pass base params

        # Store API key if provided, otherwise rely on environment variable
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in arguments or environment. Google API calls may fail.")
            # Consider raising an error if key is strictly required?

        # Build GenerationConfig dictionary from relevant parameters
        self.generation_config_dict = {
            key: value
            for key, value in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k, # Google SDK might expect float, needs verification
                "max_output_tokens": max_output_tokens,
                "stop_sequences": stop_sequences,
                # Add other supported genai.GenerationConfig fields if needed from kwargs
            }.items()
            if value is not None
        }

        # Handle safety settings if passed via kwargs (assuming CrewAI might pass them)
        self.safety_settings_dict = kwargs.get("safety_settings")

        logger.info(f"Initialized GoogleGenAI_LLM wrapper for model: {self.model_name}")
        logger.debug(f"Generation Config: {self.generation_config_dict}")
        logger.debug(f"Safety Settings: {self.safety_settings_dict}")

    def _get_model(self) -> genai.GenerativeModel:
        """Initializes and returns the google-generativeai model instance."""
        if self._model_instance is None:
            try:
                # Configure only if API key is explicitly provided here
                # Otherwise, assume it's configured globally or via env var
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    logger.debug("Configured google-generativeai with provided API key.")
                elif not os.getenv("GEMINI_API_KEY"):
                     # Check env var again just before use if not provided initially
                     raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor.")

                self._model_instance = genai.GenerativeModel(model_name=self.model_name)
                logger.debug(f"Successfully created GenerativeModel instance for {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI model '{self.model_name}': {e}", exc_info=True)
                raise RuntimeError(f"Could not initialize Google Generative AI model: {e}") from e
        return self._model_instance

    def _convert_messages(self, messages: Union[str, List[Dict[str, str]]]) -> Tuple[List[ContentDict], Optional[ContentDict]]:
        """
        Converts CrewAI message format to google-generativeai format.
        Extracts the system instruction if present.
        """
        system_instruction: Optional[ContentDict] = None
        converted_messages: List[ContentDict] = []

        if isinstance(messages, str):
            # Single string message is always treated as user input
            converted_messages.append({"role": "user", "parts": [{"text": messages}]})
            return converted_messages, system_instruction

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                logger.warning(f"Skipping invalid message format: {msg}")
                continue

            role_lower = role.lower()

            # Handle system instruction
            if role_lower == "system":
                if system_instruction:
                    logger.warning("Multiple system messages found. Using the last one.")
                # System instruction format based on REST API example
                system_instruction = {"parts": [{"text": content}]}
                continue # Don't add system message to the main conversation history

            # Map other CrewAI roles to Gemini roles
            gemini_role = ROLE_MAP.get(role_lower)
            if not gemini_role:
                logger.warning(f"Unsupported role '{role}' found. Mapping to 'user'. Message: {content[:50]}...")
                gemini_role = "user" # Default fallback

            # Simple conversion assuming text parts for now
            # TODO: Handle multimodal parts if needed later
            parts: List[PartDict] = [{"text": content}]
            converted_messages.append({"role": gemini_role, "parts": parts})

        return converted_messages, system_instruction


    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None, # TODO: Map tools if needed
        callbacks: Optional[List[Any]] = None, # Callbacks not directly used here
            available_functions: Optional[Dict[str, Any]] = None, # Deprecated? Use tools
    ) -> str:
        """
        Calls the Google Generative AI model.

        Args:
            messages: Input messages (string or list of dicts).
            tools: Optional list of tool schemas (currently basic mapping).
            callbacks: Ignored by this implementation.
            available_functions: Ignored by this implementation.

        Returns:
            The text response from the model.

        Raises:
            RuntimeError: If the API call fails.
        """
        model = self._get_model()
        converted_messages, system_instruction_content = self._convert_messages(messages)

        # Set system instruction directly on the model if provided
        if system_instruction_content:
             # Ensure it's in the correct ContentDict format expected by the SDK property
             # _convert_messages already returns this format: {"parts": [{"text": ...}]}
             try:
                 # Attempt to set the property directly on the model instance
                 model.system_instruction = system_instruction_content
                 logger.debug(f"Set system_instruction on model: {system_instruction_content}")
             except AttributeError:
                 logger.error("Failed to set system_instruction directly on the model. This attribute might not be supported or settable this way.", exc_info=True)
                 # Fallback or raise error? For now, log and continue without it.
                 # If this fails, the API call might ignore the system instruction.
             except Exception as e:
                 logger.error(f"Unexpected error setting system_instruction on model: {e}", exc_info=True)


        # Prepare generation config (without system instruction)
        gen_config = GenerationConfig(**self.generation_config_dict) if self.generation_config_dict else None
        safety_settings = self.safety_settings_dict # Pass directly if provided

        # Basic tool mapping (if tools are provided)
        genai_tools = None
        if tools:
             # This is a simplified mapping; real implementation might need more detail
             # based on how CrewAI structures the 'tools' dict vs genai's expectations.
             # Assuming 'tools' follows OpenAI-like schema for now.
             try:
                 genai_tools = [genai.types.FunctionDeclaration(**tool['function']) for tool in tools if 'function' in tool]
                 if not genai_tools: genai_tools = None # Ensure it's None if list is empty
             except Exception as e:
                 logger.warning(f"Failed to convert CrewAI tools to Gemini format: {e}. Proceeding without tools.")
                 genai_tools = None


        logger.debug(f"Calling Gemini model {self.model_name} with {len(converted_messages)} messages.")
        # logger.debug(f"Messages: {converted_messages}") # Can be very verbose
        logger.debug(f"Generation Config for call: {gen_config}")
        logger.debug(f"Safety Settings for call: {safety_settings}")
        logger.debug(f"Tools for call: {genai_tools}")


        try:
            # Use generate_content for potentially multi-turn history
            response = model.generate_content(
                contents=converted_messages,
                generation_config=gen_config, # Pass config without system_instruction
                safety_settings=safety_settings,
                tools=genai_tools
                # system_instruction is set on the model instance now
                # TODO: Add tool_config if needed/provided
            )

            # Handle potential blocking or errors in response
            if not response.candidates:
                 feedback = response.prompt_feedback
                 block_reason = feedback.block_reason.name if feedback.block_reason else "Unknown"
                 logger.error(f"Gemini call blocked. Reason: {block_reason}. Feedback: {feedback}")
                 # Returning the block reason might be more informative than raising error sometimes
                 # raise RuntimeError(f"Gemini call failed due to blocking: {block_reason}")
                 return f"Error: Call blocked by API. Reason: {block_reason}"

            # Extract text from the first candidate
            # TODO: Handle function calls if response contains them
            if response.candidates[0].content and response.candidates[0].content.parts:
                 # Concatenate text parts, ignore others for now
                 text_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                 return text_response
            else:
                 # Handle cases where response might be empty or structured differently (e.g., only function call)
                 logger.warning(f"Gemini response candidate did not contain expected text parts. Response: {response}")
                 return "" # Return empty string or handle appropriately

        except GoogleAPIError as e:
            logger.error(f"Google API error during Gemini call: {e}", exc_info=True)
            raise RuntimeError(f"Google API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Gemini call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during Gemini call: {e}") from e

    def get_context_window_size(self) -> int:
        """Returns context window size based on known Gemini models (approximate)."""
        # TODO: Find a more reliable way to get this, maybe via model listing?
        if "1.5-pro" in self.model_name:
            return 2_000_000 # Or 1M depending on specific version? Check latest docs.
        elif "1.5-flash" in self.model_name:
            return 1_000_000
        elif "flash" in self.model_name: # Older flash
             return 32_000 # Approximation, check specific model
        elif "pro" in self.model_name: # Older pro
             return 32_000 # Approximation
        else:
            logger.warning(f"Unknown context window size for model {self.model_name}. Defaulting to 32000.")
            return 32_000 # Default fallback

    # Override other methods if needed, e.g., supports_function_calling
    # def supports_function_calling(self) -> bool:
    #     # Check if the specific Gemini model supports function calling
    #     # Return True or False based on model capabilities
    #     return True # Assume True for modern Gemini models
