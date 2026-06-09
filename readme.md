# RAG-Powered Documentation Assistant

A production-grade Retrieval-Augmented Generation system 
with hybrid search, cross-encoder reranking, and citation 
verification.

## What It Does

Ask questions in plain English. Get answers grounded in 
your documentation — with source citations, confidence 
scores, and zero hallucination on out-of-context questions.

## Architecture
Query
→ Hybrid Retrieval (BM25 + ChromaDB vector search)
→ RRF Fusion (combines both ranked lists)
→ Cross-Encoder Reranker (picks top 3)
→ Claude API Generation (cited answer)
→ Citation Verification (validates [Chunk N] references)
→ Audit Log (every request logged)

## Key Engineering Decisions

**Why hybrid retrieval?**
Vector search finds semantically similar chunks but misses 
exact keyword matches (function names, error codes, specific 
terms). BM25 catches those. RRF fusion combines both without 
needing to tune weights.

**Why a cross-encoder reranker?**
The retriever fetches 10 candidates fast. The reranker 
scores each one precisely against the query. Two-stage 
approach: recall first, precision second.

**Why citation verification?**
The LLM is instructed to cite [Chunk N] references. The 
parser checks every cited number actually exists in the 
retrieved set. Catches hallucinated citations before 
they reach the user.

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Vector Store | ChromaDB | File-based, no server needed |
| Sparse Search | BM25 (rank_bm25) | Keyword matching |
| Fusion | Reciprocal Rank Fusion | No weight tuning needed |
| Reranker | Cross-Encoder | Precision second pass |
| Generation | Claude API (Anthropic) | Strong citation following |
| API | FastAPI | Async, production-grade |
| Logging | Custom audit logger | Full request traceability |

## Project Structure
src/
ingestion/     # Document loading and chunking
retrieval/     # Vector store, BM25, hybrid RRF
reranking/     # Cross-encoder reranker
generation/    # Prompt builder, Claude chain, audit log
api/           # FastAPI service
eval/            # Evaluation framework
docs/            # Sample documentation corpus

## Setup

```bash
git clone https://github.com/mewandimalsha/RAG-Powered-Documentation-Assistant
cd RAG-Powered-Documentation-Assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # for embeddings

Index your documents:
```bash
python -m src.ingestion.pipeline
```

Run the API:
```bash
uvicorn src.api.main:app --reload
```

Ask a question:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Example Response

```json
{
  "answer": "The refund policy allows returns within 30 
             days of purchase [Chunk 2]. Digital products 
             are non-refundable after download [Chunk 4].",
  "confidence": "HIGH",
  "citations_valid": true,
  "sources": "[Chunk 2] refund_policy.md, [Chunk 4] terms.md",
  "latency_ms": 847,
  "audit_id": "a3f2c1..."
}
```

## What's Different From a Basic RAG Demo

Most RAG demos: single PDF + vector search + GPT call.

This system:
- Hybrid retrieval (BM25 + vector, not just vector)
- RRF fusion (mathematically principled combination)
- Two-stage retrieval (fast recall + precise reranking)  
- Citation verification (no hallucinated sources)
- Audit logging (every request traceable)
- Production API pattern (pipeline initialized once at boot)
