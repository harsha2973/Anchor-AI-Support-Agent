"""
Unit tests for the evaluation harness: LLM-as-a-judge and full pipeline runner.
"""

from pathlib import Path
import pytest

from eval.harness.judge import LLMJudge, JudgeEvaluation
from eval.harness.runner import generate_markdown_report


def test_judge_evaluation_pass():
    judge = LLMJudge()
    verdict = judge.evaluate_response(
        test_id="TEST_001",
        customer_query="Where is my order #123?",
        retrieved_context="Track your order status directly via amazon.com/orders.",
        generated_response="You can track your order status directly via amazon.com/orders.",
        escalated=False,
        expected_escalate=False,
    )
    assert isinstance(verdict, JudgeEvaluation)
    assert verdict.passed is True
    assert verdict.grounding_score == 5
    assert verdict.overall_score >= 4.0


def test_judge_evaluation_untraceable_claim():
    judge = LLMJudge()
    verdict = judge.evaluate_response(
        test_id="TEST_002",
        customer_query="Can I have a refund?",
        retrieved_context="For refunds, please check your orders page.",
        generated_response="Sure! We will process your complete refund within 48 hours via https://fake-claim.com.",
        escalated=False,
        expected_escalate=False,
    )
    assert verdict.grounding_score <= 3
    assert verdict.passed is False


def test_generate_markdown_report():
    summary_mock = {
        "classification_metrics": {
            "intent_accuracy": 0.88,
            "per_intent_metrics": {
                "order_status_delay": {"precision": 0.92, "recall": 0.92, "f1": 0.92, "support": 25}
            },
        },
        "escalation_metrics": {
            "escalation_accuracy": 0.715,
            "escalation_precision": 0.566,
            "escalation_recall": 0.945,
            "false_auto_handle_rate": 0.055,
            "false_escalation_rate": 0.417,
            "safety_assessment": "FAIL: 4 Critical False Auto-Handles detected",
            "confusion_breakdown": {
                "true_escalations_tp": 69,
                "true_auto_handles_tn": 74,
                "false_escalations_fp": 53,
                "false_auto_handles_critical_fn": 4,
            },
        },
        "judge_qualitative_metrics": {
            "mean_grounding_score": 5.0,
            "mean_correctness_score": 4.89,
            "mean_tone_score": 5.0,
            "mean_completeness_score": 4.98,
            "mean_overall_score": 4.97,
            "pass_rate": 0.945,
        },
        "tier_stratification": {
            "routine": {
                "count": 72,
                "intent_accuracy": 0.931,
                "escalation_accuracy": 0.847,
                "false_auto_handles": 0,
            }
        },
    }
    baselines_mock = {
        "trivial_baseline_always_escalate": {"intent_accuracy": 0.135, "missed_escalation_rate": 0.0},
        "simple_keyword_baseline": {"intent_accuracy": 0.605, "missed_escalation_rate": 0.507},
    }
    report_md = generate_markdown_report(summary_mock, baselines_mock)
    assert "Production Agent Evaluation Benchmark" in report_md
    assert "88.0%" in report_md
    assert "5.5% (4 cases)" in report_md
