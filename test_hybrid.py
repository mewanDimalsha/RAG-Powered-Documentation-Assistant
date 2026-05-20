from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever

# Load existing index
vector_store = VectorStore()
bm25 = BM25Retriever()

# Create hybrid retriever
retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25,
    k=5,
    fetch_k=10,
)

# Test it
query = "what is the refund policy?"
print(f"\n🔎 Query: '{query}'\n")

results = retriever.retrieve(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Text:      {doc.page_content[:150]}...")
    print(f"Source:    {doc.metadata.get('source')}")
    print(f"RRF Score: {doc.metadata.get('rrf_score')}")
    print(f"In vector: rank {doc.metadata.get('vector_rank', 'not found')}")
    print(f"In BM25:   rank {doc.metadata.get('bm25_rank', 'not found')}")
