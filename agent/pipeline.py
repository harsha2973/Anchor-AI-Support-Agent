"""
End-to-end support agent pipeline orchestrator.

Coordinates each stage:
1. Input preprocessing & sanitization
2. Intent classification & confidence scoring
3. Grounding document retrieval
4. Policy check (escalation vs. autonomous response)
5. Response drafting & citation injection
"""

from __future__ import annotations

import functools
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from agent.drafter import BaseResponseDrafter, GroundedResponseDrafter
from agent.policy import EscalationPolicy, PolicyConfig
from agent.state import AgentState
from data.preprocess import TextPreprocessor
from intents.classifier import BaseIntentClassifier, LLMIntentClassifier
from retrieval.indexer import ResponseIndexer
from retrieval.store import BaseGroundingStore, SemanticGroundingStore

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_default_store() -> SemanticGroundingStore:
    """Initializes and caches the semantic grounding store with historical brand replies and policies."""
    from pathlib import Path
    store = SemanticGroundingStore()
    parquet_path = Path("data/processed/grounding_store.parquet")
    if parquet_path.exists():
        logger.info("Initializing SemanticGroundingStore from %s ...", parquet_path)
        store.load_from_parquet(parquet_path)
    else:
        logger.info("Building Grounding Store from taxonomy and historical pairs ...")
        indexer = ResponseIndexer(store)
        indexer.index_taxonomy_policies()
        indexer.index_historical_pairs(max_pairs_per_intent=200)
    logger.info("SemanticGroundingStore ready with %d verified documents", len(store.documents))
    return store


class SupportAgentPipeline:
    """
    Production pipeline orchestrator for customer support interactions.

    Encapsulates all components, manages dependency injection, and returns
    auditable AgentState objects for each processed query.
    """

    def __init__(
        self,
        classifier: Optional[BaseIntentClassifier] = None,
        store: Optional[BaseGroundingStore] = None,
        drafter: Optional[BaseResponseDrafter] = None,
        policy: Optional[EscalationPolicy] = None,
        preprocessor: Optional[TextPreprocessor] = None,
    ) -> None:
        self.preprocessor = preprocessor or TextPreprocessor()
        self.classifier = classifier or LLMIntentClassifier()
        self.store = store or get_default_store()
        self.drafter = drafter or GroundedResponseDrafter()
        self.policy = policy or EscalationPolicy(PolicyConfig(confidence_threshold=0.65, min_retrieval_score=0.15))

    def process_query(
        self,
        raw_query: str,
        query_id: Optional[str] = None,
        thread_context: Optional[Union[str, List[str]]] = None,
    ) -> AgentState:
        """
        Execute the full pipeline on an inbound customer query.

        Control flow guarantees:
        1. Classify intent and retrieve grounding context.
        2. ALWAYS attempt to draft a grounded response, regardless of escalation status.
        3. Independently evaluate escalation policy.
        4. If escalated, retain the draft as a suggested response for human review.
        5. Only fall back to generic specialist message if drafting itself failed.
        """
        trace_id = query_id or str(uuid.uuid4())
        state = AgentState(query_id=trace_id, raw_query=raw_query)

        # Stage 1: Preprocessing & PII Redaction
        state.cleaned_query = self.preprocessor.clean(raw_query)

        # Stage 2: Intent Classification
        state.intent_prediction = self.classifier.classify(state.cleaned_query)

        # Stage 3: Grounding Retrieval
        category_filter = state.intent_prediction.predicted_category.value if state.intent_prediction else None
        state.retrieved_documents = self.store.search(
            query=state.cleaned_query,
            top_k=3,
            filter_category=category_filter,
            min_score_threshold=0.05,
        )

        # Stage 4: Grounded Response Drafting (ALWAYS attempted first)
        try:
            draft, citations = self.drafter.draft(
                query=state.cleaned_query,
                retrieved_docs=state.retrieved_documents,
                intent_category=category_filter,
                thread_context=thread_context,
            )
            state.draft_response = draft
            state.citations = citations
        except Exception as e:
            logger.warning("Grounded response drafting failed (%s); fallback will be used", e)
            state.draft_response = None
            state.citations = []

        # Stage 5: Independent Policy & Escalation Evaluation
        should_escalate, reason = self.policy.evaluate(state)
        state.should_escalate = should_escalate
        state.escalation_reason = reason

        # Stage 6: Fallback ONLY in the rare edge case where drafting genuinely failed
        if not state.draft_response:
            state.draft_response = (
                "I am connecting you with a customer support specialist who can review your account "
                "and assist with this directly. Your inquiry has been prioritized."
            )

        return state


_default_pipeline: Optional[SupportAgentPipeline] = None


def run(
    message: str,
    thread_context: Optional[Union[str, List[str]]] = None,
    pipeline: Optional[SupportAgentPipeline] = None,
) -> Dict[str, Any]:
    """
    Top-level single function entrypoint for executing the AI customer support agent.

    Args:
        message: Inbound customer message text.
        thread_context: Optional previous conversation dialogue context.
        pipeline: Optional SupportAgentPipeline instance (uses default if None).

    Returns:
        Structured dictionary matching task specification:
        {
            "intent": str,
            "confidence": float,
            "escalate": bool,
            "escalate_reason": Optional[str],
            "draft_reply": str,
            "grounding_sources": List[Dict[str, Any]],
        }
    """
    global _default_pipeline
    active_pipeline = pipeline or _default_pipeline
    if active_pipeline is None:
        active_pipeline = SupportAgentPipeline()
        if pipeline is None:
            _default_pipeline = active_pipeline

    state = active_pipeline.process_query(raw_query=message, thread_context=thread_context)

    intent_str = state.intent_prediction.predicted_category.value if state.intent_prediction else "other_unclear"
    conf_val = round(state.intent_prediction.confidence, 4) if state.intent_prediction else 0.0

    sources = [
        {
            "doc_id": r.document.doc_id,
            "category": r.document.category,
            "score": round(float(r.score), 4),
            "content": r.document.content,
            "title": r.document.title or "",
            "customer_issue": (
                r.document.metadata.get("customer_query")
                or r.document.title
                or ("Policy Specification" if r.document.metadata.get("is_policy") else "Historical Customer Query")
            ),
        }
        for r in state.retrieved_documents
    ]

    return {
        "intent": intent_str,
        "confidence": conf_val,
        "escalate": state.should_escalate,
        "escalate_reason": state.escalation_reason,
        "draft_reply": state.draft_response,
        "grounding_sources": sources,
    }
