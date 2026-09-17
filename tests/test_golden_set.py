"""
Unit tests verifying the integrity and schema of the Golden Evaluation Set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.metrics import compute_classification_metrics
from intents.taxonomy import IntentCategory, SUPPORT_TAXONOMY

GOLDEN_SET_PATH = Path("eval/golden_set/golden_examples.jsonl")


def test_golden_set_file_exists():
    """Ensure golden_examples.jsonl exists and is non-empty."""
    assert GOLDEN_SET_PATH.exists(), f"Golden set missing at {GOLDEN_SET_PATH}"
    assert GOLDEN_SET_PATH.stat().st_size > 0, "Golden set file is empty"


def test_golden_set_schema_and_distribution():
    """Ensure all 200 records adhere strictly to schema and taxonomy constraints."""
    records = []
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 200, f"Expected 200 records, got {len(records)}"

    required_fields = {
        "id",
        "message_text",
        "turn_index",
        "thread_context",
        "true_intent",
        "true_escalate",
        "escalate_reason",
        "difficulty",
        "review_status",
    }
    valid_intents = {cat.value for cat in IntentCategory}
    valid_tiers = {"routine", "ambiguous", "high_risk", "multi_issue", "low_signal", "angry_profane"}

    tier_counts = {t: 0 for t in valid_tiers}
    intent_counts = {i: 0 for i in valid_intents}
    escalate_count = 0

    for r in records:
        # Schema validation
        missing = required_fields - set(r.keys())
        assert not missing, f"Record {r.get('id')} missing fields: {missing}"

        assert isinstance(r["id"], str) and r["id"].startswith("GOLDEN_")
        assert isinstance(r["message_text"], str) and len(r["message_text"].strip()) > 0
        assert isinstance(r["turn_index"], int)
        assert isinstance(r["thread_context"], (list, str))
        assert isinstance(r["true_escalate"], bool)
        assert isinstance(r["escalate_reason"], str) and len(r["escalate_reason"]) > 0

        # Taxonomy validity
        assert r["true_intent"] in valid_intents, f"Invalid intent {r['true_intent']}"
        assert r["difficulty"] in valid_tiers, f"Invalid difficulty {r['difficulty']}"

        # Review status check
        assert r["review_status"] in ("auto_labeled_pending_review", "human_verified")

        tier_counts[r["difficulty"]] += 1
        intent_counts[r["true_intent"]] += 1
        if r["true_escalate"]:
            escalate_count += 1

    # Verify edge case oversampling (>50% of the dataset)
    edge_case_count = sum(tier_counts[t] for t in valid_tiers if t != "routine")
    assert edge_case_count >= 100, f"Expected at least 100 edge cases, found {edge_case_count}"

    # Verify escalation representation
    assert 40 <= escalate_count <= 90, f"Unbalanced escalation count: {escalate_count}"


def test_safety_critical_escalations():
    """Verify that mandatory human escalation intents and high-risk tiers are strictly marked true_escalate=True."""
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            intent_enum = IntentCategory(r["true_intent"])
            intent_def = SUPPORT_TAXONOMY[intent_enum]

            if intent_def.requires_human_escalation or r["difficulty"] in ("high_risk", "angry_profane"):
                assert r["true_escalate"] is True, (
                    f"Record {r['id']} ({r['true_intent']}, {r['difficulty']}) has high risk but true_escalate is False!"
                )


def test_metrics_computation():
    """Verify metrics calculation logic on synthetic evaluation records."""
    mock_records = [
        {
            "id": "T1",
            "expected_intent": "order_status_delay",
            "predicted_intent": "order_status_delay",
            "expected_escalate": False,
            "predicted_escalate": False,
            "difficulty": "routine",
        },
        {
            "id": "T2",
            "expected_intent": "billing_payment_disputes",
            "predicted_intent": "billing_payment_disputes",
            "expected_escalate": True,
            "predicted_escalate": True,
            "difficulty": "high_risk",
        },
        {
            "id": "T3",
            "expected_intent": "returns_refunds",
            "predicted_intent": "returns_refunds",
            "expected_escalate": False,
            "predicted_escalate": True,  # False escalation
            "difficulty": "routine",
        },
    ]

    summary = compute_classification_metrics(mock_records)
    assert summary.total_samples == 3
    assert summary.intent_accuracy == 1.0
    assert summary.escalation_accuracy == pytest.approx(2 / 3, 0.01)
    assert summary.missed_escalation_rate == 0.0
    assert summary.false_escalation_rate == 0.5
    assert "routine" in summary.tier_metrics
    assert "high_risk" in summary.tier_metrics
