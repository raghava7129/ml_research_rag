import logging
from langchain_core.tools import tool
from sympy import sympify, SympifyError

logger=logging.getLogger(__name__)


retriever_description=(
    """
    Search the user's uploaded documents (research/technical PDFs) for
    information relevant to a query. Use this whenever the question could
    plausibly be answered from the user's documents. Do NOT use this for
    general knowledge, small talk, or math.

    Returns a summary of what was found, plus the source chunks.
        """
    )


calculator_description=(
    """
    Evaluate a basic math expression, e.g. '12 * (7 + 3)' or 'sqrt(16)'.
    Use this for arithmetic or simple math questions instead of guessing.
    Returns the numeric result as a string.
    """
)


def make_retriever_tool(ragchain_ins):
    """
    Build the document-retrieval tool, bound to a specific RAGChain instance
    (rag_engine) so it reuses your existing Multi-Query + Self-Query retriever,
    the relevance grader, and the query-rewrite-and-retry logic
    """
    @tool(description=retriever_description, response_format='content_and_artifact')
    def retrieve_documents(query: str):
        graded_docs = ragchain_ins._retrieve_and_grade(query)

        if not graded_docs:
            logger.info("Retriever tool: no relevant chunks, rewriting query.")
            rewritten = ragchain_ins.rewrite_query(query)
            graded_docs = ragchain_ins._retrieve_and_grade(rewritten)

        if not graded_docs:
            logger.info("Retriever tool: no relevant chunks after rewriting query.")
            return {"content": "No relevant documents found.", "artifact": None}

        content = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in graded_docs
        )
        return {"content": content, "artifact": graded_docs}
    
    return retrieve_documents

@tool(description=calculator_description)
def calculator(expression: str)->str:
    try:
        result = sympify(expression)
        return str(result)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Could not evaluate '{expression}': {e}"