from pathlib import Path
import os
import logging
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent
PAPERS_DIR = ROOT_DIR / "data/papers"
CHROMA_DIR = ROOT_DIR / "data/chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL_NAME = "gemini-3.5-flash"
LLM_TEMPERATURE = 0.3
RETRIEVER_K = 3


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