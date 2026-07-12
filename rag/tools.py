import logging
from langchain_core.tools import tool
from sympy import sympify, SympifyError

logger=logging.getLogger(__name__)


def make_retriever_tool(ragchain_ins):
    @tool(response_format='content_and_artifact')
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

@tool
def calculator(expression: str)->str:
    try:
        result = sympify(expression)
        return str(result)
    except (SympifyError, TypeError, ValueError) as e:
        return f"Could not evaluate '{expression}': {e}"