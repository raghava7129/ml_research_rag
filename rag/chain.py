import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

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
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.k = k
        self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

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

        print(f"Vectorstore loaded from '{self.persist_dir}'")
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
        )

        prompt_template = """
        You are an expert ML research assistant.
        Answer the question using ONLY the context provided below.
        If the answer is not in the context, say "I don't have enough information in the provided papers."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
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

    def ask(self, question):
        """
        Ask a question and get a grounded answer from the research papers.

        Returns:
            dict with keys:
                - "answer"  : the LLM's response
                - "sources" : list of source documents used
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        chain = self.build_chain()
        result = chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"],
        }
