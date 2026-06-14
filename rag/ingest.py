import os

from pathlib import Path
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import (
    DOCUMENTS_DIR,
    LEGACY_SOURCE_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

class DocumentIngester:
    """Handles loading, chunking, and storing PDF documents."""

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

    def load_documents(self):
        """Load all PDFs from the documents directory."""

        source_dir = self.documents_dir
        if not os.path.exists(source_dir) and os.path.exists(LEGACY_SOURCE_DIR):
            source_dir = LEGACY_SOURCE_DIR

        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Folder '{source_dir}' not found.")

        pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise ValueError(f"No PDFs found in '{source_dir}'. Add some documents first!")

        print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

        loader = PyPDFDirectoryLoader(source_dir)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages total.")
        return documents

    def chunk_documents(self, documents):
        """Split documents into smaller overlapping chunks."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        print(f"Split into {len(chunks)} chunks from {len(documents)} pages.")
        return chunks

    def build_vectorstore(self, chunks):
        """Embed chunks using HuggingFace and persist to ChromaDB."""
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
        """Full pipeline: load → chunk → embed → save."""
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        vectorstore = self.build_vectorstore(chunks)
        return vectorstore