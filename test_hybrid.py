from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker

# Load existing indexes
vector_store = VectorStore()
bm25 = BM25Retriever()
reranker = CrossEncoderReranker(top_k=3)  # ← final top 3 after reranking

# Wire everything together
retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_retriever=bm25,
    reranker=reranker,  # ← plugged in here
    k=5,
    fetch_k=10,
)

# Test queries — notice the second one tests exact matching
queries = [
    "what is the refund policy?",
    "are digital products refundable?",
    "how long does support take to respond?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"🔎 Query: '{query}'")
    print("=" * 60)

    results = retriever.retrieve(query)

    print(f"\n📋 Final top {len(results)} results after reranking:\n")
    for i, doc in enumerate(results):
        print(f"Rank {i+1}:")
        print(f"  Text:         {doc.page_content[:100]}...")
        print(f"  Rerank score: {doc.metadata.get('rerank_score', 'N/A')}")
        print(f"  RRF score:    {doc.metadata.get('rrf_score', 'N/A')}")
        print(f"  Source:       {doc.metadata.get('filename')}")
        print()
