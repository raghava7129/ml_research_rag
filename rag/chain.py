import os
from .config import DEFAULT_PROMPT_TEMPLATE
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from google.genai.errors import ServerError

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

REQUIRED_PROMPT_BLOCK = """
Context:
{context}

Question:
{question}

Answer:
"""

# Silence noisy third-party INFO logs; keep retry warnings visible.
for noisy_logger in [
    "httpx",
    "huggingface_hub",
    "sentence_transformers",
    "google_genai",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Hide Hugging Face unauthenticated warning noise in normal runs.
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

load_dotenv()


class RAGChain:
    """
    Connects ChromaDB retriever with Gemini LLM to answer
    questions grounded in ML research papers.
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma_db/",
        model_name: str = "gemini-3.5-flash",
        k: int = 3,  # number of chunks to retrieve per question
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.k = k
        self.prompt_template = prompt_template
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

        self.chain = self.build_chain()

    def load_vectorstore(self):
        """Load the existing ChromaDB vectorstore from disk."""

        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Vectorstore not found at '{self.persist_dir}'. "
                "Run DocumentIngester.ingest() first."
            )

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
        )

        logger.debug("Vectorstore loaded from '%s'", self.persist_dir)
        return vectorstore

    def build_chain(self):
        """
        Build the RAG chain by connecting ChromaDB retriever with Gemini.
        Returns a RetrievalQA chain ready to answer questions.
        """
        vectorstore = self.load_vectorstore()

        # convert vectorstore into a retriever.
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.3, # add some variability to responses
            max_retries=0,
            timeout=30,
        )

        prompt_template = (
            f"{self.prompt_template.strip()}\n\n{REQUIRED_PROMPT_BLOCK.strip()}"
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
        return chain

    @retry(
        retry=retry_if_exception_type((ServerError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke_chain(self, question):
        return self.chain.invoke({"query": question})

    def ask(self, question):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            result = self._invoke_chain(question)

            return {
                "answer": result["result"],
                "sources": result["source_documents"],
            }

        except (ServerError, TimeoutError, ConnectionError) as e:
            logger.error(
                "Gemini unavailable after all retries: %s",
                str(e),
            )

            return {
                "answer": (
                    "The AI service is temporarily unavailable. "
                    "Please try again later."
                ),
                "sources": [],
            }

        except Exception:
            logger.exception("Unexpected error while processing query")
            return {
                "answer": "An unexpected error occurred while processing your question.",
                "sources": [],
            }