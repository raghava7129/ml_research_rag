import os
from .config import DEFAULT_PROMPT_TEMPLATE, CHROMA_DIR, LLM_MODEL_NAME
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

for noisy_logger in [
    "httpx",
    "huggingface_hub",
    "sentence_transformers",
    "google_genai",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

# MultiQueryRetriever logs the generated query variants at INFO level.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

load_dotenv()


def _format_docs(docs) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """
    Connects ChromaDB retriever with Gemini LLM to answer questions
    grounded in uploaded documents, via an LCEL pipeline with
    Multi-Query (query translation) wrapping Self-Query (query construction).
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        model_name: str = LLM_MODEL_NAME,
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

    def build_retriever(self, vectorstore, llm):
        """
        Build the combined retriever:
          1. Self-Query (query construction) — parses the question into a
             metadata filter (e.g. source == 'paper.pdf') + a cleaned
             semantic query, then filters + searches Chroma.
          2. Multi-Query (query translation) — wraps (1), generating several
             paraphrased versions of the question first, running each
             through the self-query retriever, and merging/deduping results.
        """
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})

        try:
            from langchain_classic.retrievers.multi_query import MultiQueryRetriever
        except Exception as e:
            logger.warning(
                "MultiQueryRetriever unavailable (%s). Falling back to base retriever.",
                str(e),
            )
            return base_retriever

        try:
            from langchain_classic.chains.query_constructor.schema import AttributeInfo
            from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
        except Exception as e:
            logger.warning(
                "SelfQueryRetriever unavailable (%s). Falling back to MultiQuery only.",
                str(e),
            )
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )

        try:
            metadata_field_info = [
                AttributeInfo(
                    name="source",
                    description=(
                        "The filename of the PDF document this chunk came from, "
                        "e.g. 'research_paper_2023.pdf'. Use this to filter to a "
                        "specific document when the user names or implies one."
                    ),
                    type="string",
                ),
                AttributeInfo(
                    name="page",
                    description=(
                        "The page number within the source PDF where this chunk "
                        "appears (0-indexed). Use this only if the user explicitly "
                        "references a page number."
                    ),
                    type="integer",
                ),
            ]

            document_content_description = (
                "Excerpts from research/technical PDF documents uploaded by the user."
            )

            self_query_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vectorstore,
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                search_kwargs={"k": self.k},
                enable_limit=False,
            )

            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self_query_retriever,
                llm=llm,
            )

            return multi_query_retriever
        except Exception as e:
            logger.warning(
                "SelfQuery initialization failed (%s). Falling back to MultiQuery only.",
                str(e),
            )
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )

    def build_chain(self):
        """
        Build the RAG pipeline as an LCEL Runnable:
        translate + construct query -> retrieve -> format context
        -> fill prompt -> call LLM -> parse output.
        """
        vectorstore = self.load_vectorstore()

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.3,
            max_retries=0,
            timeout=30,
        )

        retriever = self.build_retriever(vectorstore, llm)

        prompt_template = (
            f"{self.prompt_template.strip()}\n\n{REQUIRED_PROMPT_BLOCK.strip()}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        # Sub-chain: takes {"context": [Document, ...], "question": str}
        # and produces the final answer string.
        answer_chain = (
            RunnablePassthrough.assign(
                context=lambda x: _format_docs(x["context"])
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Full chain: takes {"question": str}, retrieves docs, runs answer_chain,
        # and returns both "context" (source docs) and "answer".
        chain = RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever
        ).assign(answer=answer_chain)

        return chain

    @retry(
        retry=retry_if_exception_type((ServerError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke_chain(self, question):
        return self.chain.invoke({"question": question})

    def ask(self, question):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            result = self._invoke_chain(question)

            return {
                "answer": result["answer"],
                "sources": result["context"],
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