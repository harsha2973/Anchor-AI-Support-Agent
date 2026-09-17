"""
State contracts and data transfer objects for the support agent pipeline.

Ensures typed, reproducible, and verifiable state transitions across each step:
classify -> retrieve -> draft -> policy guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from intents.classifier import IntentPrediction
from retrieval.store import RetrievalResult


@dataclass
class AgentState:
    """
    Mutable context container flowing through the support agent pipeline.

    Attributes:
        query_id: Unique trace identifier for this customer query.
        raw_query: Initial uncleaned customer query string.
        cleaned_query: Sanitized and normalized query string.
        intent_prediction: Intent classification result (category, confidence, ambiguity).
        retrieved_documents: Grounding documents returned by vector/lexical retrieval.
        draft_response: Tentative generated response synthesized from retrieved context.
        should_escalate: Boolean flag determined by policy (True if requiring human intervention).
        escalation_reason: Explicit rationale explaining why escalation was triggered.
        citations: List of doc_ids or citations backing the response draft.
        metadata: Tracing metadata (timestamps, latency, model names, tokens).
    """

    query_id: str
    raw_query: str
    cleaned_query: str = ""
    intent_prediction: Optional[IntentPrediction] = None
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    draft_response: Optional[str] = None
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for logging, evaluation recording, and debugging."""
        return {
            "query_id": self.query_id,
            "raw_query": self.raw_query,
            "cleaned_query": self.cleaned_query,
            "intent": self.intent_prediction.predicted_category.value if self.intent_prediction else None,
            "intent_confidence": self.intent_prediction.confidence if self.intent_prediction else None,
            "retrieved_doc_count": len(self.retrieved_documents),
            "draft_response": self.draft_response,
            "should_escalate": self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "citations": self.citations,
            "metadata": self.metadata,
        }
