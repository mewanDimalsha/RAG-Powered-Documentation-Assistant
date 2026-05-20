from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List
import time


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    Why cross-encoder beats bi-encoder for reranking:
    - Bi-encoder: encodes query and chunk separately → fast but loses context
    - Cross-encoder: reads query + chunk together → slower but much more accurate

    Production pattern:
    - Retrieve 10-20 candidates fast (hybrid retrieval)
    - Rerank with cross-encoder (runs on small set, slowness acceptable)
    - Pass top 3-5 to LLM
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ):
        print(f"🔄 Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        print(f"✅ Reranker ready")

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Reranks documents by true relevance to the query.

        Steps:
        1. Build (query, chunk) pairs
        2. Score every pair with cross-encoder
        3. Sort by score descending
        4. Return top_k with scores attached
        """

        if not documents:
            return []

        start_time = time.time()

        # Step 1: Build pairs — model reads query + chunk together
        pairs = [[query, doc.page_content] for doc in documents]

        # Step 2: Score all pairs in one batch
        # Returns a list of floats — one score per pair
        scores = self.model.predict(pairs)

        # Step 3: Attach scores to documents and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)
            scored_docs.append(doc)

        # Sort by rerank score, highest first
        scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)

        latency = round((time.time() - start_time) * 1000, 1)

        # Step 4: Return top_k
        top_docs = scored_docs[: self.top_k]

        # Log what happened — important for production observability
        print(
            f"\n  ⚡ Reranker: {len(documents)} → {len(top_docs)} results ({latency}ms)"
        )
        for i, doc in enumerate(top_docs):
            print(
                f"  Rank {i+1}: score={doc.metadata['rerank_score']:.4f} "
                f"| {doc.page_content[:60]}..."
            )

        return top_docs
