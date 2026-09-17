"""
Quantitative evaluation metrics calculation.

Computes precision, recall, F1, escalation accuracy, grounding coverage,
and failure distributions across evaluated runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSummary:
    """
    Consolidated performance summary across an evaluation run.

    Attributes:
        total_samples: Total number of test cases evaluated.
        intent_accuracy: Fraction of correctly classified intents.
        escalation_accuracy: Fraction of correct escalation decisions.
        escalation_precision: Precision of escalation predictions (TP / (TP + FP)).
        escalation_recall: Recall of escalation predictions (TP / (TP + FN)).
        false_escalation_rate: Autonomous queries that were unnecessarily escalated.
        missed_escalation_rate: High-risk queries that failed to escalate (critical failure mode).
        grounding_hit_rate: Fraction of queries where at least one reference doc was retrieved.
        per_intent_metrics: Detailed precision/recall/F1 breakdown per intent category.
        tier_metrics: Performance metrics broken down by difficulty tier.
        escalation_confusion: Raw confusion counts (tp, fp, tn, fn).
        failure_case_ids: List of test IDs that failed any criterion.
    """

    total_samples: int
    intent_accuracy: float
    escalation_accuracy: float
    escalation_precision: float
    escalation_recall: float
    false_escalation_rate: float
    missed_escalation_rate: float
    grounding_hit_rate: float
    per_intent_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tier_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    escalation_confusion: Dict[str, int] = field(default_factory=dict)
    failure_case_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics summary to dictionary."""
        return {
            "total_samples": self.total_samples,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "escalation_accuracy": round(self.escalation_accuracy, 4),
            "escalation_precision": round(self.escalation_precision, 4),
            "escalation_recall": round(self.escalation_recall, 4),
            "false_escalation_rate": round(self.false_escalation_rate, 4),
            "missed_escalation_rate": round(self.missed_escalation_rate, 4),
            "grounding_hit_rate": round(self.grounding_hit_rate, 4),
            "escalation_confusion": self.escalation_confusion,
            "tier_metrics": self.tier_metrics,
            "per_intent_metrics": self.per_intent_metrics,
            "failure_count": len(self.failure_case_ids),
            "failure_case_ids": self.failure_case_ids,
        }


def compute_classification_metrics(
    eval_records: List[Dict[str, Any]]
) -> EvaluationSummary:
    """
    Compute comprehensive metrics from pipeline evaluation outputs.

    Args:
        eval_records: List of dicts containing:
            - 'id': Test ID
            - 'expected_intent': Gold category string
            - 'predicted_intent': Model category string
            - 'expected_escalate': Gold boolean
            - 'predicted_escalate': Model boolean
            - 'difficulty': Difficulty tier string
            - 'expected_doc_ids': Gold doc IDs (optional)
            - 'retrieved_doc_ids': Model retrieved doc IDs

    Returns:
        EvaluationSummary instance.
    """
    if not eval_records:
        return EvaluationSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total = len(eval_records)
    correct_intent = 0
    correct_escalate = 0
    grounding_hits = 0
    failures = []

    # Escalation confusion counts
    tp_esc = 0  # Expected escalate and predicted escalate
    fp_esc = 0  # Expected auto and predicted escalate
    tn_esc = 0  # Expected auto and predicted auto
    fn_esc = 0  # Expected escalate and predicted auto (critical safety miss)

    # Per-intent metrics trackers: {intent: {"tp": 0, "fp": 0, "fn": 0, "total": 0}}
    intents = set()
    for r in eval_records:
        if r.get("expected_intent"):
            intents.add(r["expected_intent"])
        if r.get("predicted_intent"):
            intents.add(r["predicted_intent"])

    intent_stats = {i: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for i in intents}

    # Stratification by difficulty tier
    tier_groups: Dict[str, List[Dict[str, Any]]] = {}

    for r in eval_records:
        exp_intent = r.get("expected_intent")
        pred_intent = r.get("predicted_intent")
        exp_esc = bool(r.get("expected_escalate", False))
        pred_esc = bool(r.get("predicted_escalate", False))
        tier = r.get("difficulty", "routine")

        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(r)

        intent_match = (exp_intent == pred_intent)
        escalate_match = (exp_esc == pred_esc)

        if intent_match:
            correct_intent += 1
            if exp_intent in intent_stats:
                intent_stats[exp_intent]["tp"] += 1
        else:
            if exp_intent in intent_stats:
                intent_stats[exp_intent]["fn"] += 1
            if pred_intent in intent_stats:
                intent_stats[pred_intent]["fp"] += 1

        if exp_intent in intent_stats:
            intent_stats[exp_intent]["support"] += 1

        # Escalation breakdown
        if exp_esc and pred_esc:
            tp_esc += 1
            correct_escalate += 1
        elif not exp_esc and not pred_esc:
            tn_esc += 1
            correct_escalate += 1
        elif not exp_esc and pred_esc:
            fp_esc += 1
        elif exp_esc and not pred_esc:
            fn_esc += 1

        # Grounding hit check
        exp_docs = set(r.get("expected_doc_ids", []))
        pred_docs = set(r.get("retrieved_doc_ids", []))
        if exp_docs:
            if exp_docs.intersection(pred_docs):
                grounding_hits += 1
        else:
            grounding_hits += 1

        if not intent_match or not escalate_match:
            failures.append(r.get("id", "UNKNOWN"))

    # Compute overall rates
    total_auto = tn_esc + fp_esc
    total_esc = tp_esc + fn_esc

    false_esc_rate = fp_esc / total_auto if total_auto > 0 else 0.0
    missed_esc_rate = fn_esc / total_esc if total_esc > 0 else 0.0
    esc_precision = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
    esc_recall = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0

    # Per-intent precision, recall, f1
    per_intent_metrics = {}
    for i, s in intent_stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        rc = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = (2 * p * rc) / (p + rc) if (p + rc) > 0 else 0.0
        per_intent_metrics[i] = {
            "precision": round(p, 4),
            "recall": round(rc, 4),
            "f1": round(f1, 4),
            "support": s["support"],
        }

    # Stratified metrics per difficulty tier
    tier_metrics = {}
    for tier, group in tier_groups.items():
        g_total = len(group)
        g_intent_correct = sum(1 for g in group if g.get("expected_intent") == g.get("predicted_intent"))
        g_esc_correct = sum(1 for g in group if bool(g.get("expected_escalate")) == bool(g.get("predicted_escalate")))
        g_missed_esc = sum(1 for g in group if bool(g.get("expected_escalate")) and not bool(g.get("predicted_escalate")))
        tier_metrics[tier] = {
            "count": g_total,
            "intent_accuracy": round(g_intent_correct / g_total, 4) if g_total > 0 else 0.0,
            "escalation_accuracy": round(g_esc_correct / g_total, 4) if g_total > 0 else 0.0,
            "missed_escalations": g_missed_esc,
        }

    return EvaluationSummary(
        total_samples=total,
        intent_accuracy=correct_intent / total,
        escalation_accuracy=correct_escalate / total,
        escalation_precision=esc_precision,
        escalation_recall=esc_recall,
        false_escalation_rate=false_esc_rate,
        missed_escalation_rate=missed_esc_rate,
        grounding_hit_rate=grounding_hits / total,
        per_intent_metrics=per_intent_metrics,
        tier_metrics=tier_metrics,
        escalation_confusion={
            "true_positives": tp_esc,
            "false_positives": fp_esc,
            "true_negatives": tn_esc,
            "false_negatives": fn_esc,
        },
        failure_case_ids=failures,
    )
