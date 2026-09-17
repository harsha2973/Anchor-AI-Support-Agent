"""
Retrieval module for grounded response generation.

Manages vector embeddings, lexical BM25 indexing, and semantic similarity search
over historical brand replies and canonical support policies.
"""

from retrieval.store import GroundingDocument, RetrievalResult, BaseGroundingStore

__all__ = ["GroundingDocument", "RetrievalResult", "BaseGroundingStore"]
