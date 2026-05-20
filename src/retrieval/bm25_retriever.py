from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List
import pickle
import os


class BM25Retriever:
    """
    Keyword-based retrieval using BM25.
    Handles exact term matching — product codes, names,
    acronyms, error codes that vector search misses.
    """

    def __init__(self, persist_path: str = "./bm25_index.pkl"):
        self.persist_path = persist_path
        self.chunks: List[Document] = []
        self.bm25 = None

    def build_index(self, chunks: List[Document]):
        """
        Build BM25 index from chunks.
        Must be called after vector store is populated.
        """
        self.chunks = chunks

        # Tokenize: split each chunk into individual words
        # BM25 works on word lists, not raw strings
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]

        # Build the BM25 index
        self.bm25 = BM25Okapi(tokenized)

        # Save to disk so we don't rebuild every restart
        self._save()
        print(f"✅ BM25 index built: {len(chunks)} chunks indexed")

    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Search using BM25 keyword matching.
        Returns top-k chunks ranked by BM25 score.
        """
        if self.bm25 is None:
            self._load()

        # Tokenize the query the same way we tokenized chunks
        query_tokens = query.lower().split()

        # Get BM25 scores for every chunk
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k scores, sorted descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]

        # Return the actual Document objects
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only return chunks with actual matches
                doc = self.chunks[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)

        return results

    def _save(self):
        """Persist index to disk."""
        with open(self.persist_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

    def _load(self):
        """Load index from disk."""
        if not os.path.exists(self.persist_path):
            raise RuntimeError("BM25 index not found. Run build_index() first.")
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.bm25 = data["bm25"]
        print("✅ BM25 index loaded from disk")
