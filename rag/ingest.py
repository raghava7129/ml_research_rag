import os
from pathlib import Path

import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from .config import (
    DOCUMENTS_DIR,
    LEGACY_SOURCE_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


class DocumentIngester:
    """Handles loading, chunking, and storing PDF documents via an LCEL chain."""

    def __init__(
        self,
        documents_dir: Path = DOCUMENTS_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        persist_dir: Path = CHROMA_DIR,
    ):
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir

        self.chain = (
            RunnableLambda(self._resolve_source_dir)
            | RunnableLambda(self._load_documents)
            | RunnableLambda(self._chunk_documents)
            | RunnableLambda(self._build_vectorstore)
        )

    def _resolve_source_dir(self, _input=None) -> str:
        source_dir = self.documents_dir
        if not os.path.exists(source_dir) and os.path.exists(LEGACY_SOURCE_DIR):
            source_dir = LEGACY_SOURCE_DIR

        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Folder '{source_dir}' not found.")

        return source_dir

    def _load_documents(self, source_dir: str):
        pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise ValueError(f"No PDFs found in '{source_dir}'. Add some documents first!")

        print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

        loader = PyPDFDirectoryLoader(source_dir)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages total.")
        return documents

    def _chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_documents(documents)

        print(f"Split into {len(chunks)} chunks (token-based) from {len(documents)} pages.")
        return chunks

    def _build_vectorstore(self, chunks):
        print("Loading embedding model (this may take a minute the first time)...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cuda"
                if torch.cuda.is_available()
                else "cpu"
            },
        )

        print(f"Embedding {len(chunks)} chunks and saving to '{self.persist_dir}'...")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
        )
        print(f"Vectorstore built and saved to '{self.persist_dir}'")
        return vectorstore

    def ingest(self):
        """Full pipeline: resolve dir → load → chunk → embed → save."""
        return self.chain.invoke(None)