"""
Grounding store interfaces and vector storage abstractions.

Enables searching across curated historical brand replies and policy documents.
All agent drafts must cite or condition on documents retrieved through this store.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundingDocument:
    """
    A verified knowledge item or historical brand response used for grounding.

    Attributes:
        doc_id: Unique identifier for the document or response turn.
        title: Short descriptive headline or query context.
        content: The actual verified text or brand response template.
        category: Relevant intent category tag.
        source_url_or_id: Pointer back to source conversation or policy manual.
        metadata: Custom metadata (e.g. brand name, verification status, date).
    """

    doc_id: str
    content: str
    title: Optional[str] = None
    category: Optional[str] = None
    source_url_or_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """
    Result of a retrieval query against the grounding store.

    Attributes:
        document: The matched GroundingDocument.
        score: Relevance/similarity score (e.g., cosine similarity [0.0, 1.0] or BM25).
        rank: 1-indexed retrieval rank.
    """

    document: GroundingDocument
    score: float
    rank: int


class BaseGroundingStore(abc.ABC):
    """Abstract interface for grounding stores."""

    @abc.abstractmethod
    def add_documents(self, documents: List[GroundingDocument]) -> None:
        """
        Index a batch of documents into the store.

        Args:
            documents: List of GroundingDocument instances to persist and index.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_category: Optional[str] = None,
        min_score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant grounding documents for a query.

        Args:
            query: Sanitized query text.
            top_k: Maximum number of results to return.
            filter_category: Optional intent category to restrict search space.
            min_score_threshold: Minimum similarity score required.

        Returns:
            List of RetrievalResult objects sorted by score descending.
        """
        raise NotImplementedError


class InMemoryGroundingStore(BaseGroundingStore):
    """
    In-memory baseline grounding store using token overlap.

    Useful for unit testing, fast scaffolding, and low-dependency benchmarking.
    """

    def __init__(self) -> None:
        self._documents: Dict[str, GroundingDocument] = {}

    def add_documents(self, documents: List[GroundingDocument]) -> None:
        logger.info("Adding %d documents to InMemoryGroundingStore", len(documents))
        for doc in documents:
            self._documents[doc.doc_id] = doc

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_category: Optional[str] = None,
        min_score_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        logger.debug("InMemoryGroundingStore searching for query: '%s' (top_k=%d)", query, top_k)
        results: List[RetrievalResult] = []

        query_tokens = set(query.lower().split())

        for idx, (doc_id, doc) in enumerate(self._documents.items()):
            if filter_category and doc.category != filter_category:
                continue

            doc_tokens = set(doc.content.lower().split())
            intersection = query_tokens.intersection(doc_tokens)
            score = len(intersection) / max(len(query_tokens), 1)

            if score >= min_score_threshold:
                results.append(RetrievalResult(document=doc, score=score, rank=1))

        results.sort(key=lambda x: x.score, reverse=True)
        # Update ranks
        ranked = [
            RetrievalResult(document=r.document, score=r.score, rank=i + 1)
            for i, r in enumerate(results[:top_k])
        ]
        return ranked


class SemanticGroundingStore(BaseGroundingStore):
    """
    Vector-indexed semantic grounding store.

    Indexes verified brand replies and policy documents using text embeddings
    and cosine similarity search. Supports intent category filtering and
    similarity score thresholding.
    """

    def __init__(self) -> None:
        self.documents: List[GroundingDocument] = []
        self._vectorizer = None
        self._embedding_matrix = None

    def load_from_parquet(
        self,
        parquet_path: Any = Path("data/processed/grounding_store.parquet"),
    ) -> int:
        """Loads and vector-indexes pre-built grounding store corpus directly from parquet."""
        from pathlib import Path
        import pandas as pd

        path = Path(parquet_path)
        if not path.exists():
            logger.warning("Grounding store parquet not found at %s", path)
            return 0

        logger.info("Loading grounding store from %s ...", path)
        df = pd.read_parquet(path)
        docs = [
            GroundingDocument(
                doc_id=str(row.doc_id),
                title=str(row.title) if hasattr(row, "title") and pd.notna(row.title) else None,
                category=str(row.category) if hasattr(row, "category") and pd.notna(row.category) else None,
                content=str(row.content),
                source_url_or_id=str(row.source_url_or_id) if hasattr(row, "source_url_or_id") and pd.notna(row.source_url_or_id) else None,
                metadata={"is_policy": getattr(row, "is_policy", False)},
            )
            for row in df.itertuples(index=False)
        ]
        self.add_documents(docs)
        return len(docs)

    def add_documents(self, documents: List[GroundingDocument]) -> None:
        """Add and vector-index documents."""
        if not documents:
            return

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.documents.extend(documents)
        logger.info("SemanticGroundingStore indexing %d total documents", len(self.documents))

        # Index combined title (customer query context) and content (brand reply)
        corpus = [
            f"{doc.title or ''} {doc.content} category_{doc.category or 'general'}"
            for doc in self.documents
        ]

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=15000,
            sublinear_tf=True,
        )
        self._embedding_matrix = self._vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_category: Optional[str] = None,
        min_score_threshold: float = 0.05,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents matching query semantics, optionally filtered by intent.
        """
        if not self.documents or self._vectorizer is None or self._embedding_matrix is None:
            return []

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Transform query vector
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._embedding_matrix).flatten()

        candidate_results: List[Tuple[float, GroundingDocument]] = []

        for idx, score in enumerate(sims):
            doc = self.documents[idx]

            # Category filtering / boosting
            effective_score = float(score)
            if filter_category:
                if doc.category == filter_category:
                    # Priority boost for matching intent precedents
                    effective_score *= 1.35
                else:
                    # Penalize out-of-intent precedents if filter is active
                    effective_score *= 0.65

            if effective_score >= min_score_threshold:
                candidate_results.append((effective_score, doc))

        # Sort descending by score
        candidate_results.sort(key=lambda x: x[0], reverse=True)

        ranked = [
            RetrievalResult(document=doc, score=round(min(score, 1.0), 4), rank=i + 1)
            for i, (score, doc) in enumerate(candidate_results[:top_k])
        ]
        return ranked
