"""
Executes agent.run end-to-end on a diverse sample of golden set examples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import agent

GOLDEN_SET_PATH = Path("eval/golden_set/golden_examples.jsonl")


def run_sample_eval(sample_ids: list[str] = None):
    # Selected sample representing routine, high-risk, ambiguous, angry, low-signal
    target_ids = sample_ids or [
        "GOLDEN_001",  # routine order tracking
        "GOLDEN_014",  # routine return label
        "GOLDEN_024",  # delivery marked delivered but missing
        "GOLDEN_030",  # damaged product
        "GOLDEN_033",  # unauthorized billing dispute (high risk)
        "GOLDEN_042",  # angry profane delivery complaint
        "GOLDEN_054",  # account locked / access security
        "GOLDEN_071",  # carrier physical incident
        "GOLDEN_088",  # multi-issue (charged twice + late)
        "GOLDEN_101",  # low-signal / unclear
    ]

    records = {}
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec

    results = []
    print("\n" + "=" * 80)
    print("RUNNING AGENT.RUN END-TO-END ON 10 DIVERSE GOLDEN BENCHMARK EXAMPLES")
    print("=" * 80)

    for test_id in target_ids:
        if test_id not in records:
            continue
        item = records[test_id]
        msg = item["message_text"]
        ctx = item.get("thread_context")

        print(f"\n--- [TEST ID: {test_id}] (Difficulty: {item.get('difficulty')}, Expected Intent: {item.get('true_intent')}) ---")
        print(f"Customer Message : {msg}")
        print(f"Thread Context   : {ctx}")

        output = agent.run(message=msg, thread_context=ctx)
        output["test_id"] = test_id
        output["expected_intent"] = item.get("true_intent")
        output["expected_escalate"] = item.get("true_escalate")
        output["difficulty"] = item.get("difficulty")
        results.append(output)

        print("Raw Agent Output:")
        print(json.dumps(output, indent=2))

    return results


if __name__ == "__main__":
    run_sample_eval()
