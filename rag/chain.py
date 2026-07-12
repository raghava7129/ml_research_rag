import os
from .config import DEFAULT_PROMPT_TEMPLATE, CHROMA_DIR, LLM_MODEL_NAME
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

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

logger.setLevel(logging.INFO)

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
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

load_dotenv()


def _format_docs(docs) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


class GradeDocuments(BaseModel):
    """Relevance grader: gives yes/no o/p."""
    binary_score: str = Field(
        description="Is the document relevant to the question? Answer 'yes' or 'no'."
    )


class RouterQuery(BaseModel):
    """Query router: decides where this Question should be sent to (Retrieval or General)."""
    datasource: str = Field(
        description=
        "Given a user question, choose which datasource would best answer it. "
        "Return 'vectorstore' if the question could plausibly be answered from "
        "the user's uploaded documents (research papers, technical PDFs, etc). "
        "Return 'general' if the question is general knowledge, small talk, "
        "or clearly unrelated to any document content, e.g. 'what's 2+2', "
        "'who are you', 'what's the weather'."
    )


OFF_TOPIC_MESSAGE = (
    "I'm only able to answer questions grounded in your uploaded documents. "
    "That question doesn't look related to them — try rephrasing it, or "
    "upload a document that covers this topic."
)


class RAGChain:
    """
    Connects ChromaDB retriever with Gemini LLM to answer questions
    grounded in uploaded documents.

    Pipeline:
      1. Retrieve chunks (Multi-Query + Self-Query).
      2. Grade each chunk for relevance to the question.
      3. If nothing is relevant, rewrite the question and retrieve once more.
      4. Generate the final answer from whatever relevant chunks remain.
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

        self._build_components()

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

    def route_question(self, question: str) -> str:
        """Classify the question as 'vectorstore' or 'general'."""
        try:
            result = self.router.invoke({"question": question})
            datasource = result.datasource.strip().lower()
        except Exception:
            logger.exception("Routing failed; defaulting to 'vectorstore'.")
            return "vectorstore"

        if datasource not in ("vectorstore", "general"):
            logger.warning("Unexpected router output '%s'; defaulting to 'vectorstore'.", datasource)
            return "vectorstore"

        logger.info("Router decision: '%s' -> %s", question, datasource)
        return datasource

    def _build_components(self):
        self.vectorstore = self.load_vectorstore()

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0.3,
            max_retries=0,
            timeout=30,
        )

        # Router (decides if retrieval is even needed)
        self.router_llm = self.llm.with_structured_output(RouterQuery)

        router_prompt_template = PromptTemplate(
            template=(
                "You are an expert at routing a user question to the right "
                "datasource.\n\n"
                "The vectorstore contains: research/technical PDF documents "
                "uploaded by the user.\n\n"
                "User question: {question}\n\n"
                "Decide whether this question should go to the 'vectorstore' "
                "or is 'general' (small talk, unrelated general knowledge)."
            ),
            input_variables=["question"],
        )

        self.router = router_prompt_template | self.router_llm

        self.retriever = self.build_retriever(self.vectorstore, self.llm)

        prompt_template = (
            f"{self.prompt_template.strip()}\n\n{REQUIRED_PROMPT_BLOCK.strip()}"
        )
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        self.answer_chain = self.prompt | self.llm | StrOutputParser()

        self.grader_llm = self.llm.with_structured_output(GradeDocuments)

        grade_prompt_template = PromptTemplate(
            template=(
                "You are a grader assessing relevance of a retrieved document "
                "to a user question.\n\n"
                "Retrieved document:\n{document}\n\n"
                "User question: {question}\n\n"
                "If the document contains information related to the question, "
                "grade it as relevant. Give a binary score 'yes' or 'no'."
            ),
            input_variables=["document", "question"],
        )
        self.grader = grade_prompt_template | self.grader_llm

        rewrite_prompt_template = PromptTemplate(
            template=(
                "You are rewriting a search query to improve retrieval from a "
                "vector database. Look at the input question and try to reason "
                "about the underlying semantic intent.\n\n"
                "Original question: {question}\n\n"
                "Rewrite it as a single, improved search query. "
                "Return ONLY the rewritten query, nothing else."
            ),
            input_variables=["question"],
        )
        self.rewriter = rewrite_prompt_template | self.llm | StrOutputParser()

    def grade_documents(self, question: str, docs: list) -> list:
        """Grade each retrieved doc for relevance; keep only the 'yes' ones."""
        relevant_docs = []
        for doc in docs:
            try:
                result = self.grader.invoke(
                    {"document": doc.page_content, "question": question}
                )
                score = result.binary_score.strip().lower()
            except Exception:
                logger.exception("Grading failed for a chunk; keeping it by default.")
                relevant_docs.append(doc)
                continue

            source = doc.metadata.get("source", "unknown")
            if score == "yes":
                logger.info("Grader: RELEVANT   (source=%s)", source)
                relevant_docs.append(doc)
            else:
                logger.info("Grader: NOT relevant (source=%s)", source)

        return relevant_docs

    def rewrite_query(self, question: str) -> str:
        """Rephrase the question for better retrieval."""
        rewritten = self.rewriter.invoke({"question": question})
        logger.info("Rewrote query: '%s' -> '%s'", question, rewritten)
        return rewritten.strip()

    def _retrieve_and_grade(self, question: str):
        docs = self.retriever.invoke(question)
        graded = self.grade_documents(question, docs)
        return graded, question

    @retry(
        retry=retry_if_exception_type((ServerError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_answer(self, question: str, docs: list) -> str:
        return self.answer_chain.invoke(
            {"context": _format_docs(docs), "question": question}
        )

    def ask(self, question: str):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            #step 0: route the question to decide if retrieval is needed.
            datasource = self.route_question(question)
            if datasource == "general":
                logger.info("Router decided this is a general question; skipping retrieval.")
                return {
                    "answer": OFF_TOPIC_MESSAGE,
                    "sources": [],
                }

            # Step 1: retrieve + grade with the original question.
            graded_docs, _ = self._retrieve_and_grade(question)

            # Step 2: if nothing survived grading, rewrite the query and retry once.
            if not graded_docs:
                logger.info("No relevant chunks found. Rewriting query and retrying.")
                rewritten_question = self.rewrite_query(question)
                graded_docs, _ = self._retrieve_and_grade(rewritten_question)

            # Step 3: generate the answer.
            answer = self._generate_answer(question, graded_docs)

            return {
                "answer": answer,
                "sources": graded_docs,
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