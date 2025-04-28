# src/llms/google_genai_embeddings.py
import os
import google.genai as genai
from google.genai import types # Import types for EmbedContentConfig
from typing import List, Optional
from google.genai.types import EmbedContentResponse, ContentDict
from google.genai.errors import APIError
import numpy as np
from crewai.knowledge.embedder.base_embedder import BaseEmbedder

from src.logger_setup import get_logger

logger = get_logger(__name__)

# Default model - check Google AI documentation for the latest/recommended models
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-exp-03-07"

class GoogleGenAIEmbeddings(BaseEmbedder):
    """
    Wrapper for Google Generative AI embeddings compatible with CrewAI's
    BaseEmbedder interface.
    Supports authentication via API key (passed or env var) or Application Default Credentials (ADC).
    """
    _client: Optional[genai.Client] = None # Store the client instance
    _dimension: Optional[int] = None # Store dimension after first call, or hardcode

    def __init__(self,
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 task_type: str = "RETRIEVAL_DOCUMENT", # Use string type hint
                 api_key: Optional[str] = None): # Add optional api_key parameter
        self.model_name = model_name
        self.task_type = task_type # Store default task type
        self.api_key = api_key # Store explicitly passed key
        # Client initialization moved to _get_client to be lazy
        logger.info(f"Initialized GoogleGenAIEmbeddings wrapper for model: {self.model_name}, task_type: {self.task_type}")

    def _get_client(self) -> genai.Client:
        """
        Initializes and returns the google-generativeai client instance for embeddings.
        Prioritizes explicitly passed API key, then environment variable, then ADC.
        """
        if self._client is None:
            try:
                # 1. Check for explicitly passed API key
                if self.api_key:
                    logger.debug("Using explicitly provided API key for embeddings client.")
                    self._client = genai.Client(api_key=self.api_key)
                # 2. Check for environment variable API key
                elif os.getenv("GEMINI_API_KEY"):
                    logger.debug("Using API key from GEMINI_API_KEY environment variable for embeddings client.")
                    self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                # 3. Fallback to Application Default Credentials (ADC)
                else:
                    logger.debug("No API key found for embeddings (explicit or env var). Using ADC.")
                    self._client = genai.Client()
                    logger.debug("Successfully instantiated embeddings client using ADC.")

            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI Client for embeddings: {e}", exc_info=True)
                raise RuntimeError(f"Could not initialize Google Generative AI Client for embeddings: {e}") from e
        return self._client


    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts using the configured Google model.
        Maps to BaseEmbedder.embed_texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A numpy array of embeddings.

        Raises:
            RuntimeError: If the embedding process fails.
        """
        if not texts or not all(isinstance(t, str) for t in texts):
            logger.warning("embed_documents received empty or invalid input.")
            return []
        try:
            client = self._get_client() # Get the initialized client
            logger.debug(f"Embedding {len(texts)} documents using model {self.model_name} with task_type {self.task_type} via client...")
            # Note: The API might handle batching internally, but check limits if needed.
            # Use client.models.embed_content with config object
            config = types.EmbedContentConfig(task_type=self.task_type)
            result: EmbedContentResponse = client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=config
            )
            logger.debug(f"Successfully embedded {len(texts)} documents.")
            # Access result.embeddings (list[ContentEmbedding]), then item.values (list[float])
            if hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                 extracted_embeddings = []
                 for emb_obj in result.embeddings:
                      if hasattr(emb_obj, 'values') and isinstance(emb_obj.values, list):
                           extracted_embeddings.append(emb_obj.values)
                      else:
                           logger.error(f"ContentEmbedding object missing 'values' list: {emb_obj}")
                           raise RuntimeError("Google embedding API returned unexpected ContentEmbedding structure.")

                 if extracted_embeddings:
                      # Store dimension if not already set
                      if self._dimension is None:
                           self._dimension = len(extracted_embeddings[0])
                           logger.info(f"Detected embedding dimension: {self._dimension}")
                      return np.array(extracted_embeddings, dtype=np.float32)
                 else:
                      # This case might occur if the input texts list was empty but passed initial checks,
                      # or if all ContentEmbedding objects lacked 'values'.
                      logger.warning("No valid embedding values found in the response.")
                      return np.array([], dtype=np.float32) # Return empty array
            else:
                 logger.error(f"Could not find 'embeddings' attribute (list) in the API response: {result}")
                 raise RuntimeError("Google embedding API response missing 'embeddings' attribute.")

        except APIError as e:
             logger.error(f"Google API error during document embedding: {e}", exc_info=True)
             raise RuntimeError(f"Google API error during document embedding: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error embedding documents with Google: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error embedding texts: {e}") from e

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Embeds a list of text chunks. For this implementation, it's identical to embed_texts.
        Maps to BaseEmbedder.embed_chunks.
        """
        logger.debug("embed_chunks called, forwarding to embed_texts.")
        return self.embed_texts(chunks)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single text string using the configured Google model.
        Maps to BaseEmbedder.embed_text.

        Args:
            text: The text string to embed.

        Returns:
            The embedding as a numpy array.

        Raises:
            RuntimeError: If the embedding process fails.
        """
        if not isinstance(text, str) or not text:
             logger.warning("embed_query received empty or invalid input.")
             # Return a zero vector or raise error? Raising is safer.
             raise ValueError("Cannot embed empty or non-string query.")
        try:
            # Determine appropriate task type for a query
            # Use string type for task_type parameter
            query_task_type = "RETRIEVAL_QUERY" # Often different from document embedding
            # Could also use "SEMANTIC_SIMILARITY" or others depending on use case
            logger.debug(f"Embedding query using model {self.model_name} with task_type {query_task_type} via client...")

            client = self._get_client() # Get the initialized client
            # Use client.models.embed_content with config object
            config = types.EmbedContentConfig(task_type=query_task_type)
            result: EmbedContentResponse = client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config
            )
            logger.debug(f"Successfully embedded query.")

            # Response for single text should have one ContentEmbedding in result.embeddings
            if hasattr(result, 'embeddings') and isinstance(result.embeddings, list) and len(result.embeddings) == 1:
                 emb_obj = result.embeddings[0]
                 if hasattr(emb_obj, 'values') and isinstance(emb_obj.values, list):
                      embedding_list = emb_obj.values
                      # Store dimension if not already set
                      if self._dimension is None and embedding_list:
                           self._dimension = len(embedding_list)
                           logger.info(f"Detected embedding dimension: {self._dimension}")
                      return np.array(embedding_list, dtype=np.float32)
                 else:
                      logger.error(f"ContentEmbedding object in single query response missing 'values' list: {emb_obj}")
                      raise RuntimeError("Google embedding API returned unexpected ContentEmbedding structure for single query.")
            elif hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                 logger.error(f"Expected 1 embedding for single query, but got {len(result.embeddings)}: {result.embeddings}")
                 raise RuntimeError("Google embedding API returned unexpected number of embeddings for single query.")
            else:
                 logger.error(f"Could not find 'embeddings' attribute (list) in the query API response: {result}")
                 raise RuntimeError("Google embedding API query response missing 'embeddings' attribute.")

        except APIError as e:
             logger.error(f"Google API error during query embedding: {e}", exc_info=True)
             raise RuntimeError(f"Google API error during query embedding: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error embedding query with Google: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error embedding text: {e}") from e

    @property
    def dimension(self) -> int:
        """
        Returns the dimension of the embeddings.
        Tries to detect dynamically, falls back to default for gemini-embedding-exp-03-07.
        """
        if self._dimension is not None:
            return self._dimension

        # Fallback or attempt detection if needed (could embed dummy text)
        return 3072 # Known dimension for gemini-embedding-exp-03-07
