"""Retrieval-augmented generation building blocks for the study chat.

    query.py      conversation-aware query understanding (no LLM)
    retrieve.py   hybrid multi-query retrieval, fusion, rerank, MMR, per-PDF balance, neighbours,
                  confidence, cross-subject search
    grounding.py  numbered context blocks, grounding prompts, citation verification

The orchestration (adaptive LLM step, tool-style re-search, SSE steps) lives in
``backend.agentic_rag``. Nothing here calls an LLM.
"""
