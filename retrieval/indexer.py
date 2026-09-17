"""
Indexing pipeline for historical brand replies and policy documents.

Transforms processed conversation threads and knowledge articles into
indexed vector/lexical embeddings ready for real-time inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from data.loader import SupportTurn
from retrieval.store import BaseGroundingStore, GroundingDocument

logger = logging.getLogger(__name__)


class ResponseIndexer:
    """
    Indexes verified brand responses and reference policies into a GroundingStore.

    Deduplicates repeated responses, associates intent tags, and prepares
    documents for semantic search.
    """

    def __init__(self, store: BaseGroundingStore) -> None:
        """
        Initialize indexer with target grounding store.

        Args:
            store: Concrete BaseGroundingStore implementation.
        """
        self.store = store

    def index_brand_turns(self, brand_turns: List[SupportTurn]) -> int:
        """
        Convert raw brand reply turns into GroundingDocuments and index them.

        Args:
            brand_turns: List of verified brand support turns.

        Returns:
            Number of documents successfully indexed.
        """
        logger.info("Indexing %d brand turns into grounding store", len(brand_turns))
        docs = []
        for turn in brand_turns:
            if not turn.is_brand_reply or len(turn.text.strip()) < 10:
                continue

            doc = GroundingDocument(
                doc_id=turn.turn_id,
                content=turn.text,
                category=turn.metadata.get("category"),
                source_url_or_id=turn.conversation_id,
                metadata=turn.metadata,
            )
            docs.append(doc)

        self.store.add_documents(docs)
        return len(docs)

    def index_taxonomy_policies(self) -> int:
        """
        Index official policy and resolution guidance from the empirical taxonomy.
        """
        from intents.taxonomy import SUPPORT_TAXONOMY

        docs = []
        for cat, defn in SUPPORT_TAXONOMY.items():
            content = f"POLICY GUIDANCE: {defn.resolution_guidance}\nSCOPE: {defn.description}"
            docs.append(
                GroundingDocument(
                    doc_id=f"POLICY_{cat.value.upper()}",
                    title=defn.name,
                    category=cat.value,
                    content=content,
                    source_url_or_id="intents/taxonomy.py",
                    metadata={"risk_level": defn.risk_level.value, "is_policy": True},
                )
            )

        self.store.add_documents(docs)
        logger.info("Indexed %d official taxonomy policy documents", len(docs))
        return len(docs)

    def index_historical_pairs(
        self,
        parquet_path: Path = Path("data/processed/amazon_support_pairs.parquet"),
        max_pairs_per_intent: int = 200,
    ) -> int:
        """
        Index real historical (customer_message, brand_reply) pairs from the processed dataset.

        Args:
            parquet_path: Path to amazon_support_pairs.parquet.
            max_pairs_per_intent: Number of diverse precedent pairs to index per intent.

        Returns:
            Number of precedent pairs indexed.
        """
        import pandas as pd
        from intents.classifier import CalibratedHeuristicClassifier

        if not parquet_path.exists():
            logger.warning("Parquet path %s does not exist; skipping historical pair indexing.", parquet_path)
            return 0

        logger.info("Loading historical pairs from %s ...", parquet_path)
        df = pd.read_parquet(parquet_path)

        # Filter clean English messages with substantial replies
        filtered = df[(df["customer_message"].str.len() > 15) & (df["brand_reply"].str.len() > 20)]

        classifier = CalibratedHeuristicClassifier()
        seen_replies = set()
        intent_counts: Dict[str, int] = {}
        docs: List[GroundingDocument] = []

        # Iterate rapidly using itertuples
        for row in filtered.itertuples(index=False):
            b_reply = str(row.brand_reply).strip()
            # Normalize reply signature for deduplication
            reply_sig = b_reply[:60].lower()
            if reply_sig in seen_replies:
                continue

            c_msg = str(row.customer_message).strip()
            pred = classifier.classify(c_msg)
            cat = pred.predicted_category.value

            if intent_counts.get(cat, 0) >= max_pairs_per_intent:
                continue

            seen_replies.add(reply_sig)
            intent_counts[cat] = intent_counts.get(cat, 0) + 1

            docs.append(
                GroundingDocument(
                    doc_id=f"PRECEDENT_{row.pair_id}",
                    title=c_msg,
                    category=cat,
                    content=b_reply,
                    source_url_or_id=f"conversation_{row.conversation_id}",
                    metadata={
                        "turn_index": int(row.turn_index),
                        "customer_query": c_msg,
                    },
                )
            )

            # Break early once all intent buckets are sufficiently populated
            if len(docs) >= max_pairs_per_intent * 12:
                break

        self.store.add_documents(docs)
        logger.info("Successfully indexed %d historical brand precedents across %d intents", len(docs), len(intent_counts))
        return len(docs)
