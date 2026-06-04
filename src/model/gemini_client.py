import os
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.logging_config import setup_logger

logger = setup_logger("gemini_client")


class GeminiClientManager:
    """Manages the connection, configuration, and API validation checks for the Google Gemini LLM via LangChain."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        api_key: Optional[str] = None
    ):
        """Initializes the Gemini LLM client configuration.
        
        Args:
            model_name (str): The specific Gemini model model name (e.g. 'gemini-1.5-flash', 'gemini-1.5-pro').
            temperature (float): Controls response variance. Lower values are more deterministic.
            max_retries (int): Number of automatic request retries on rate limits or service hiccups.
            timeout (Optional[float]): Time limit in seconds to await model responses.
            api_key (Optional[str]): Gemini API key overrides. If None, checks the environment variables.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Load API keys sequentially prioritizing explicit arguments, then standard environments
        self.api_key = (
            api_key 
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("GOOGLE_API_KEY")
        )
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Instantiates and returns the configured ChatGoogleGenerativeAI instance.
        
        Returns:
            ChatGoogleGenerativeAI: A LangChain-compliant Gemini LLM model instance.
            
        Raises:
            ValueError: If no valid API Key can be resolved.
            RuntimeError: If initialization fails due to configuration conflicts.
        """
        if self._llm is not None:
            return self._llm

        if not self.api_key:
            logger.error(
                "Gemini API key is not configured. "
                "Ensure GEMINI_API_KEY or GOOGLE_API_KEY environment variable is defined."
            )
            raise ValueError(
                "Google Gemini API Key is required. Please set GEMINI_API_KEY in your environment."
            )

        logger.info(
            f"Configuring ChatGoogleGenerativeAI connection (Model: '{self.model_name}', "
            f"Temp: {self.temperature}, Timeout: {self.timeout}s, Retries: {self.max_retries})"
        )

        try:
            # Setup LangChain LLM Client wrapper
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_retries=self.max_retries,
                timeout=self.timeout
            )
            return self._llm
        except Exception as e:
            logger.exception(f"Failed to load ChatGoogleGenerativeAI client: {str(e)}")
            raise RuntimeError(f"Gemini client setup failed: {str(e)}") from e

    def validate_connection(self) -> bool:
        """Runs a validation request checking key verification and network connection to Gemini API.
        
        Returns:
            bool: True if test invoke succeeds, False otherwise.
        """
        logger.info("Executing API authorization connection check...")
        try:
            llm = self.get_llm()
            # Send a lightweight test request
            response = llm.invoke("ping")
            if response and response.content:
                logger.info("Credentials check succeeded. Gemini API connection is active.")
                return True
            logger.warning("Gemini API returned empty text packet during authorization test.")
            return False
        except Exception as e:
            logger.error(f"Gemini API connectivity validation failed: {str(e)}")
            return False
