"""Simple environment-backed settings.

This avoids importing pydantic BaseSettings which may not be available in the
installed pydantic version. Use environment variables to override defaults.
"""
import os

# Try to load .env automatically if python-dotenv is installed (optional).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv not installed; continue and rely on environment variables
    pass


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "project_kb")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    PORT: int = int(os.getenv("PORT", "8000"))
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "60"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    # API Token storage (set dynamically from auth header)
    API_TOKEN: str = ""
    
    @classmethod
    def set_api_token(cls, token: str):
        """Set the API token from the authorization header.
        
        Args:
            token: The bearer token from the Authorization header
        """
        cls.API_TOKEN = token
        
    @classmethod
    def get_api_token(cls) -> str:
        """Get the currently stored API token.
        
        Returns:
            str: The API token, or empty string if not set
        """
        return cls.API_TOKEN

settings = Settings()
