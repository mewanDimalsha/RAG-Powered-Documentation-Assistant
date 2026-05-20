from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ProductionChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever


class IngestionPipeline:
    """
    End-to-end ingestion: Load → Chunk → Embed → Store
    Idempotent: safe to re-run, won't duplicate data.
    """

    def __init__(self, docs_directory: str = "./docs"):
        self.docs_directory = docs_directory
        self.loader = DocumentLoader()
        self.chunker = ProductionChunker(chunk_size=512, overlap=50)
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever()

    def run(self) -> dict:
        print("\n🚀 Starting ingestion pipeline...\n")

        # Step 1: Load
        print("Step 1/3: Loading documents...")
        documents = self.loader.load_directory(self.docs_directory)

        if not documents:
            return {"status": "error", "message": "No documents found"}

        # Step 2: Chunk
        print("\nStep 2/3: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: Embed + Store
        print("\nStep 3/3: Embedding and storing...")
        indexed = self.vector_store.add_documents(chunks)

        print("\nStep 4/4: Building BM25 keyword index...")
        self.bm25.build_index(chunks)

        result = {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_indexed": indexed,
            "total_in_store": self.vector_store.count,
        }

        print(f"\n✅ Pipeline complete: {result}")
        return result


# Run it directly
if __name__ == "__main__":
    pipeline = IngestionPipeline(docs_directory="./docs")
    pipeline.run()
