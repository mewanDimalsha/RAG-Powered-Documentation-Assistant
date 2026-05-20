from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from langchain_core.documents import Document
from typing import List, Optional
from collections import defaultdict


class HybridRetriever:
    """
    Stage 1: Hybrid retrieval (BM25 + Vector + RRF)
    Stage 2: Optional cross-encoder reranking
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        reranker=None,  # ← inject reranker optionally
        k: int = 5,
        fetch_k: int = 10,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self.k = k
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> List[Document]:

        # Stage 1: Hybrid retrieval
        vector_results = self.vector_store.similarity_search(
            query=query, k=self.fetch_k
        )
        bm25_results = self.bm25.search(query=query, k=self.fetch_k)

        print(f"  🔍 Vector: {len(vector_results)} results")
        print(f"  🔍 BM25:   {len(bm25_results)} results")

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)
        print(f"  ✅ RRF fusion: {len(fused)} unique results")

        # Stage 2: Rerank if reranker is available
        if self.reranker:
            return self.reranker.rerank(query=query, documents=fused)

        return fused[: self.k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
    ) -> List[Document]:

        scores = defaultdict(float)
        docs_by_id = {}

        for rank, doc in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[chunk_id] += 1.0 / (rank + self.rrf_k)
            doc.metadata["vector_rank"] = rank + 1
            docs_by_id[chunk_id] = doc

        for rank, doc in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[chunk_id] += 1.0 / (rank + self.rrf_k)
            doc.metadata["bm25_rank"] = rank + 1
            if chunk_id not in docs_by_id:
                docs_by_id[chunk_id] = doc

        ranked_ids = sorted(scores.keys(), key=lambda id_: scores[id_], reverse=True)

        final_results = []
        for chunk_id in ranked_ids:
            doc = docs_by_id[chunk_id]
            doc.metadata["rrf_score"] = round(scores[chunk_id], 6)
            final_results.append(doc)

        return final_results
