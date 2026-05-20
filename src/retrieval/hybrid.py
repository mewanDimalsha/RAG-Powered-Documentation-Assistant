from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from langchain_core.documents import Document
from typing import List
from collections import defaultdict


class HybridRetriever:
    """
    Combines vector search (semantic) + BM25 (keyword)
    using Reciprocal Rank Fusion.

    Why this matters:
    - Vector alone: misses exact terms (codes, names, acronyms)
    - BM25 alone: misses meaning ("refund" vs "money back")
    - Hybrid: catches both — significantly better in production
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        k: int = 5,  # final results to return
        fetch_k: int = 10,  # results to fetch from each retriever
        rrf_k: int = 60,  # RRF smoothing constant
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.k = k
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> List[Document]:
        """
        Main retrieval method.
        1. Search both retrievers
        2. Fuse results with RRF
        3. Return top-k
        """

        # Step 1: Get results from both retrievers
        vector_results = self.vector_store.similarity_search(
            query=query, k=self.fetch_k
        )
        bm25_results = self.bm25.search(query=query, k=self.fetch_k)

        print(f"  🔍 Vector: {len(vector_results)} results")
        print(f"  🔍 BM25:   {len(bm25_results)} results")

        # Step 2: Fuse with RRF
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Step 3: Return top-k
        return fused[: self.k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion algorithm.

        For each result list:
            score = 1 / (rank + rrf_k)

        If same chunk appears in both lists:
            final_score = vector_score + bm25_score
            (gets double boost → rises to top)
        """

        # scores[chunk_id] = accumulated RRF score
        scores = defaultdict(float)

        # Store the actual Document objects by chunk_id
        docs_by_id = {}

        # Score vector results
        for rank, doc in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_score = 1.0 / (rank + self.rrf_k)
            scores[chunk_id] += rrf_score
            doc.metadata["vector_rank"] = rank + 1
            docs_by_id[chunk_id] = doc

        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_score = 1.0 / (rank + self.rrf_k)
            scores[chunk_id] += rrf_score
            doc.metadata["bm25_rank"] = rank + 1

            # Keep the doc object (may have extra metadata)
            if chunk_id not in docs_by_id:
                docs_by_id[chunk_id] = doc

        # Sort by accumulated RRF score (highest first)
        ranked_ids = sorted(scores.keys(), key=lambda id_: scores[id_], reverse=True)

        # Build final result list with RRF scores attached
        final_results = []
        for chunk_id in ranked_ids:
            doc = docs_by_id[chunk_id]
            doc.metadata["rrf_score"] = round(scores[chunk_id], 6)
            final_results.append(doc)

        print(f"  ✅ RRF fusion: {len(final_results)} unique results")
        return final_results
