"""
Unit tests verifying the Trivial and Simple baseline models and evaluation artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.harness.baselines import RESULTS_OUTPUT_PATH, SimpleBaseline, TrivialBaseline


def test_trivial_baseline_execution():
    """Verify that TrivialBaseline returns consistent majority class and fixed policy."""
    bot_always = TrivialBaseline(majority_intent="order_status_delay", escalation_mode="always_escalate")
    res1 = bot_always.process("Where is my package?")
    assert res1["predicted_intent"] == "order_status_delay"
    assert res1["predicted_escalate"] is True

    bot_never = TrivialBaseline(majority_intent="other_unclear", escalation_mode="never_escalate")
    res2 = bot_never.process("I want to speak with a lawyer immediately!")
    assert res2["predicted_intent"] == "other_unclear"
    assert res2["predicted_escalate"] is False


def test_simple_baseline_execution():
    """Verify SimpleBaseline keyword matching and escalation rules."""
    bot = SimpleBaseline()

    # Rule matching: Returns
    res_ret = bot.process("How do I return this item? Where is the refund?")
    assert res_ret["predicted_intent"] == "returns_refunds"
    assert res_ret["predicted_escalate"] is False
    assert "Online Returns Center" in res_ret["response"]

    # Rule matching: Billing dispute (mandatory escalation)
    res_bill = bot.process("You charged me twice for this order!")
    assert res_bill["predicted_intent"] == "billing_payment_disputes"
    assert res_bill["predicted_escalate"] is True

    # Rule matching: Keyword escalation trigger
    res_esc = bot.process("This service is unacceptable, let me talk to a supervisor or lawyer.")
    assert res_esc["predicted_escalate"] is True


def test_baseline_results_json_integrity():
    """Ensure baseline_results.json exists, is non-empty, and has valid metric keys."""
    assert RESULTS_OUTPUT_PATH.exists(), f"Baseline results missing at {RESULTS_OUTPUT_PATH}"
    with open(RESULTS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "baselines" in data
    baselines = data["baselines"]

    for name in ["trivial_baseline_always_escalate", "trivial_baseline_never_escalate", "simple_keyword_baseline"]:
        assert name in baselines, f"Missing baseline {name}"
        b = baselines[name]
        assert "intent_accuracy" in b
        assert "escalation_accuracy" in b
        assert "escalation_confusion" in b
        assert "per_intent_metrics" in b
        assert "intent_confusion_matrix" in b

        conf = b["escalation_confusion"]
        assert all(k in conf for k in ["true_positives", "false_positives", "true_negatives", "false_negatives"])
        assert sum(conf.values()) == 200
