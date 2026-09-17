"""
CLI entrypoint for running the AI Customer Support Agent evaluation suite.

Usage:
    python -m eval.run --golden-set eval/golden_set/golden_examples.jsonl
    python -m eval.run --golden-set eval/golden_set/golden_examples.jsonl --judge
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from eval.harness.runner import run_full_evaluation
from eval.harness.judge import LLMJudge
from eval.harness.metrics import EvaluationSummary, compute_classification_metrics

logger = logging.getLogger(__name__)


def load_golden_set(file_path: Path) -> List[Dict[str, Any]]:
    """Loads JSON Lines golden evaluation dataset."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    return records


def run_evaluation(
    golden_set_path: Path,
    enable_judge: bool = False,
    quiet: bool = False,
) -> EvaluationSummary:
    """
    Executes pipeline against golden evaluation dataset and computes metrics.

    Args:
        golden_set_path: Path to golden_examples.jsonl.
        enable_judge: Whether to run qualitative LLM judge scoring.
        quiet: If True, suppresses individual test case output.

    Returns:
        EvaluationSummary instance.
    """
    records = load_golden_set(golden_set_path)
    print(f"\nLoaded {len(records)} golden test examples from {golden_set_path}")

    # Set up pipeline
    store = setup_demo_store()
    pipeline = SupportAgentPipeline(store=store)
    judge = LLMJudge() if enable_judge else None

    eval_results = []

    if not quiet:
        print("\nEvaluating test cases...")
        print("-" * 75)

    for item in records:
        test_id = item["id"]
        query = item.get("message_text") or item.get("query", "")
        exp_intent = item.get("true_intent") or item.get("expected_intent")
        exp_esc = item.get("true_escalate") if "true_escalate" in item else item.get("expected_escalate", False)
        difficulty = item.get("difficulty", "routine")
        exp_docs = item.get("grounding_doc_ids", [])

        # Process through pipeline
        state = pipeline.process_query(query, query_id=test_id)

        pred_intent = state.intent_prediction.predicted_category.value if state.intent_prediction else None
        pred_esc = state.should_escalate
        retrieved_ids = [doc.document.doc_id for doc in state.retrieved_documents]

        record = {
            "id": test_id,
            "query": query,
            "expected_intent": exp_intent,
            "predicted_intent": pred_intent,
            "expected_escalate": exp_esc,
            "predicted_escalate": pred_esc,
            "difficulty": difficulty,
            "expected_doc_ids": exp_docs,
            "retrieved_doc_ids": retrieved_ids,
            "draft_response": state.draft_response,
            "escalation_reason": state.escalation_reason,
        }

        if judge:
            judge_res = judge.evaluate_response(
                test_id=test_id,
                customer_query=query,
                retrieved_context="\n".join(d.document.content for d in state.retrieved_documents),
                generated_response=state.draft_response or "",
                escalated=pred_esc,
                expected_escalate=exp_esc,
            )
            record["judge_score"] = judge_res.to_dict()

        eval_results.append(record)

        # Print live status
        if not quiet:
            intent_status = "PASS" if pred_intent == exp_intent else "FAIL"
            esc_status = "PASS" if pred_esc == exp_esc else "FAIL"
            clean_q = query.replace("\n", " ")[:35]
            print(f"[{test_id:<12}] [{difficulty:<13}] Intent: {intent_status:<4} | Esc: {esc_status:<4} | {clean_q}...")

    summary = compute_classification_metrics(eval_results)
    return summary


def main() -> None:
    """CLI parser for evaluation run."""
    parser = argparse.ArgumentParser(
        description="Run the evaluation harness on the customer support agent."
    )
    parser.add_argument(
        "--golden-set",
        "-g",
        type=Path,
        default=Path("eval/golden_set/golden_examples.jsonl"),
        help="Path to golden set JSONL file.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-a-judge qualitative scoring in addition to quantitative metrics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw summary JSON instead of console report.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress individual test item output.",
    )

    args = parser.parse_args()
    results = run_full_evaluation(golden_set_path=args.golden_set, quiet=args.quiet)
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
