"""
Evaluation harness containing metrics calculations and LLM-as-judge scoring.
"""

from eval.harness.judge import LLMJudge, JudgeEvaluation
from eval.harness.metrics import EvaluationSummary, compute_classification_metrics

__all__ = ["LLMJudge", "JudgeEvaluation", "EvaluationSummary", "compute_classification_metrics"]
