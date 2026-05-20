from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
import os
import chromadb
from langchain_core.documents import Document


class VectorStore:
    """
    Production ChromaDB wrapper with:
    - Persistent storage (survives restarts)
    - Deduplication via chunk IDs
    - Configurable embedding model
    - Metadata filtering support
    """

    def __init__(
        self,
        collection_name: str = "production_rag",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Load embedding model locally — no API call needed
        # normalize_embeddings=True improves cosine similarity accuracy
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Persistent client — data survives restarts
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain wrapper for easy document insertion/querying
        self.store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        print(f"✅ Vector store ready: '{collection_name}'")

    def add_documents(self, chunks: List[Document]) -> int:
        # Add chunks to the vector store.
        # Uses chunk_id for deduplication — safe to re-run.

        # Extract IDs for deduplication
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        # Check which IDs already exist
        existing = self.client.get_collection(self.collection_name).get(ids=ids)
        existing_ids = set(existing["ids"])

        # Filter to only new chunks
        new_chunks = [
            chunk for chunk, id_ in zip(chunks, ids) if id_ not in existing_ids
        ]

        if not new_chunks:
            print("⚡ All chunks already indexed. Skipping.")
            return 0

        self.store.add_documents(new_chunks)
        print(
            f"✅ Indexed {len(new_chunks)} new chunks "
            f"({len(chunks) - len(new_chunks)} duplicates skipped)"
        )
        return len(new_chunks)

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Semantic search — returns top-k most similar chunks.
        filter example: {"source": "policy.pdf"}
        """
        return self.store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 10,
    ) -> List[tuple[Document, float]]:
        """
        Same as above but returns similarity scores.
        Useful for debugging retrieval quality.
        """
        return self.store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

    @property
    def count(self) -> int:
        return self.client.get_collection(self.collection_name).count()
