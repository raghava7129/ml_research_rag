from pathlib import Path
import os
import logging
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
DOCUMENTS_DIR = ROOT_DIR / "data/documents"
LEGACY_SOURCE_DIR = ROOT_DIR / "data/papers"
CHROMA_DIR = ROOT_DIR / "data/chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL_NAME = "gemini-3.5-flash"
LLM_TEMPERATURE = 0.3
RETRIEVER_K = 5

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not found in the context, say "I don't have enough information in the provided documents."
"""

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ml-research-rag")
LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)


def setup_langsmith() -> dict:
    """Initialize LangSmith tracing env vars and return current tracing status."""
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    if LANGSMITH_TRACING:
        if api_key:
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
            os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            logger.info(
                "LangSmith tracing enabled (project=%s, endpoint=%s).",
                LANGSMITH_PROJECT,
                LANGSMITH_ENDPOINT,
            )
            return {
                "enabled": True,
                "project": LANGSMITH_PROJECT,
                "endpoint": LANGSMITH_ENDPOINT,
            }

        logger.warning("LANGSMITH_API_KEY is missing.")

    return {
        "enabled": False,
        "project": LANGSMITH_PROJECT,
        "endpoint": LANGSMITH_ENDPOINT,
    }

def validate_model_access(model_name: str = LLM_MODEL_NAME) -> bool:
    """
    Check if the API key has access to the configured model.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

        client = genai.Client(api_key=api_key)
        available_models = [model.name for model in client.models.list()]

        if f"models/{model_name}" in available_models:
            logger.info("Model '%s' is accessible.", model_name)
            return True
        else:
            logger.error(
                "Model '%s' not accessible. Available models: %s",
                model_name,
                available_models,
            )
            return False

    except ValueError as e:
        logger.error("Configuration error: %s", str(e))
        return False

    except Exception:
        logger.exception("Unexpected error during model validation.")
        return False