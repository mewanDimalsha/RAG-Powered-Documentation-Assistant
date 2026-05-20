import hashlib
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ProductionChunker:
    """
    Production-grade chunker with overlap, metadata preservation,
    and chunk ID generation for deduplication.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk a list of documents while preserving metadata and generating chunk IDs.

        Args:
            documents (list): List of documents, where each document is a dict with 'text' and 'metadata'.

        Returns:
            list: List of chunked documents with metadata and chunk IDs.
        """
        chunks = []

        for doc in documents:
            splits = self.splitter.split_documents([doc])

            for i, chunk in enumerate(splits):
                # Generate a stable unique ID for each chunk
                # This prevents re-ingesting the same chunk twice
                chunk_id = hashlib.md5(
                    f"{chunk.metadata.get('source', '')}-{chunk.page_content[:50]}".encode()
                ).hexdigest()

                # Enrich metadata — this powers citations later
                chunk.metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "chunk_total": len(splits),
                        "char_count": len(chunk.page_content),
                    }
                )
                chunks.append(chunk)

        print(f"✅ Chunked {len(documents)} docs → {len(chunks)} chunks")
        return chunks
