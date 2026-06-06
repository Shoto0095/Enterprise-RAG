"""LLM implementations for the chatbot.

Provides `GeminiLLM` class for Google Gemini integration.
"""
from time import sleep

import google.generativeai as genai
from langchain_core.language_models import LLM
from typing import Optional, List, Iterator
from .logger import get_logger
from .config import settings

_logger = get_logger("llm")

# Configure Gemini API key
genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiLLM(LLM):
    """LangChain-compatible wrapper for Google Gemini LLM.
    
    This class provides integration with Google's Gemini API through
    the LangChain LLM interface.
    """
    
    model: str = settings.GEMINI_MODEL

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text from the given prompt.
        
        Args:
            prompt: The input prompt for text generation
            stop: Optional list of stop sequences (not used by Gemini)
            
        Returns:
            Generated text response
        """
        try:
            response = genai.GenerativeModel(self.model).generate_content(prompt)
            return response.text
        except Exception as e:
            _logger.exception(f"Error calling Gemini API: {e}")
            raise

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            response = genai.GenerativeModel(self.model).generate_content(
                prompt,
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    words = chunk.text.split(" ")
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words) - 1 else "")
                        sleep(0.04)  # ~25 words/sec — tune to taste
        except Exception as e:
            _logger.exception(f"Error streaming from Gemini API: {e}")
            raise

    @property
    def _identifying_params(self):
        """Get identifying parameters for the LLM."""
        return {"model": self.model}

    @property
    def _llm_type(self):
        """Get the type of LLM."""
        return "gemini_llm"
