"""
Escalation policies, guardrails, and decision rules.

Determines whether an interaction can be handled autonomously or MUST be escalated
to a human support agent. Implements transparent heuristics based on:
1. Intent risk level (e.g. legal disputes, billing fraud)
2. Model confidence thresholds
3. Grounding document availability & similarity thresholds
4. Ambiguity and sentiment indicators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from agent.state import AgentState
from intents.taxonomy import IntentCategory, RiskLevel, SUPPORT_TAXONOMY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    """
    Thresholds and settings governing escalation policy decisions.

    Attributes:
        confidence_threshold: Minimum intent classifier confidence required for autonomy.
        min_retrieval_score: Minimum retrieval similarity required to consider a response grounded.
        min_retrieved_docs: Minimum number of relevant grounding documents required.
        require_grounding_for_medium_risk: If True, medium risk queries require verified docs.
    """

    confidence_threshold: float = 0.60
    min_retrieval_score: float = 0.20
    min_retrieved_docs: int = 1
    require_grounding_for_medium_risk: bool = True


class EscalationPolicy:
    """
    Evaluates pipeline state to determine whether to escalate to a human agent.

    Produces an explicit binary decision (escalate: True/False) and an auditable
    rationale string explaining why.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def evaluate(self, state: AgentState) -> Tuple[bool, str]:
        """
        Evaluate current state against safety, risk, and grounding policies.

        Args:
            state: Active AgentState after classification and retrieval.

        Returns:
            Tuple of (should_escalate: bool, reason: str).
        """
        import re

        # Rule 1: Missing classification
        if not state.intent_prediction:
            return True, "Escalated: No intent prediction available."

        category = state.intent_prediction.predicted_category
        confidence = state.intent_prediction.confidence
        taxonomy_def = SUPPORT_TAXONOMY.get(category)
        query_text = (state.cleaned_query or state.raw_query or "").lower()

        # Rule 2: Explicit critical safety and litigation triggers in message text
        safety_trigger_pat = re.compile(
            r"\b(lawyer|attorney|police|court|ftc|sue\s+(you|amazon)|fraudulent\s+charge|account\s+hacked)\b",
            re.I,
        )
        if safety_trigger_pat.search(query_text):
            return True, "Escalated: Customer inquiry contains explicit legal, regulatory, or crime allegation."

        # Rule 3: Catch-all 'other_unclear' intent
        if category == IntentCategory.OTHER_UNCLEAR:
            return True, "Escalated: Query intent is unclassifiable or unclear ('other_unclear'); human triage required."

        # Rule 4: Mandatory escalation by taxonomy definition (fraud/billing/security/conduct)
        if taxonomy_def and taxonomy_def.requires_human_escalation:
            return True, (
                f"Escalated: Intent '{category.value}' is categorized as {taxonomy_def.risk_level.value} "
                "risk and requires mandatory human handling per policy."
            )

        # Rule 5: Low classification confidence or high ambiguity
        if confidence < self.config.confidence_threshold or state.intent_prediction.is_ambiguous:
            return True, (
                f"Escalated: Intent confidence ({confidence:.2f}) is below operational "
                f"threshold ({self.config.confidence_threshold:.2f}) or prediction is ambiguous."
            )

        # Rule 6: Grounding check (no relevant precedent retrieved)
        qualified_docs = [
            doc for doc in state.retrieved_documents
            if doc.score >= self.config.min_retrieval_score
        ]
        if len(qualified_docs) < self.config.min_retrieved_docs:
            return True, (
                f"Escalated: No relevant precedent retrieved with similarity >= {self.config.min_retrieval_score:.2f} "
                f"(found {len(qualified_docs)} qualified precedents)."
            )

        # Autonomy approved with explicit rationale
        return False, (
            f"Autonomous handling approved: High confidence ({confidence:.2%}) in intent '{category.value}' "
            f"backed by {len(qualified_docs)} qualified grounding precedent(s)."
        )
