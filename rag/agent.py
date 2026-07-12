import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from google.genai.errors import ServerError

from .chain import RAGChain  # reused for retrieval + grading internals
from .tools import make_retriever_tool, calculator
from .config import CHROMA_DIR, LLM_MODEL_NAME

logger = logging.getLogger(__name__)
load_dotenv()

DEFAULT_AGENT_SYSTEM_PROMPT = """
You are a helpful assistant with access to two tools:
1. retrieve_documents — searches the user's uploaded documents.
2. calculator — evaluates math expressions.

Rules:
- If the question could be answered from the user's documents, ALWAYS use
  retrieve_documents before answering. Never answer document-related
  questions from memory.
- If retrieve_documents finds nothing relevant, tell the user you don't have
  enough information in the provided documents — do not make something up.
- Use calculator for arithmetic instead of computing it yourself.
- For general knowledge or small talk that clearly has nothing to do with
  the documents, you may answer directly without any tool.
"""


class RAGAgent:
    """
    A ReAct agent (LangGraph) that decides, turn by turn, whether to call the
    document retriever, the calculator, or answer directly.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_DIR,
        model_name: str = LLM_MODEL_NAME,
        system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
    ):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

        self.system_prompt = system_prompt

        self._retrieval_engine = RAGChain(
            persist_dir=persist_dir,
            model_name=model_name,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
            temperature=0.3,
            max_retries=0,
            timeout=30,
        )

        retriever_tool = make_retriever_tool(self._retrieval_engine)
        self.tools = [retriever_tool, calculator]

        self.graph = create_agent(model=self.llm, tools=self.tools)

    @retry(
        retry=retry_if_exception_type((ServerError, TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _invoke_graph(self, question: str):
        return self.graph.invoke(
            {
                "messages": [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=question),
                ]
            }
        )

    def ask(self, question: str):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            result = self._invoke_graph(question)
            messages = result["messages"]
            answer = messages[-1].content

            sources = []
            for msg in messages:
                if isinstance(msg, ToolMessage) and msg.name == "retrieve_documents":
                    if msg.artifact:
                        sources.extend(msg.artifact)

            return {"answer": answer, "sources": sources}

        except (ServerError, TimeoutError, ConnectionError) as e:
            logger.error("Gemini unavailable after all retries: %s", str(e))
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