"""
CLI entrypoint for running the AI Customer Support Agent pipeline.

Usage:
    python -m agent.run --query "Where is my package #12345?"
    python -m agent.run --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.pipeline import SupportAgentPipeline
from intents.taxonomy import SUPPORT_TAXONOMY
from retrieval.store import GroundingDocument, InMemoryGroundingStore


def setup_demo_store():
    """Returns the full semantic grounding store populated with thousands of real brand precedents."""
    from agent.pipeline import get_default_store
    return get_default_store()


def format_state_output(state) -> str:
    """Format the AgentState output cleanly for console inspection."""
    category = state.intent_prediction.predicted_category.value if state.intent_prediction else "Unknown"
    confidence = state.intent_prediction.confidence if state.intent_prediction else 0.0

    lines = [
        "\n" + "=" * 60,
        "SUPPORT AGENT PIPELINE RESULT",
        "=" * 60,
        f"Query ID           : {state.query_id}",
        f"Inbound Query      : {state.raw_query}",
        f"Sanitized Query    : {state.cleaned_query}",
        "-" * 60,
        f"Predicted Intent   : {category} (Confidence: {confidence:.2%})",
        f"Grounding Docs     : {len(state.retrieved_documents)} retrieved",
        f"Escalate to Human  : {'[YES - ESCALATED]' if state.should_escalate else '[NO - AUTONOMOUS]'}",
    ]

    if state.retrieved_documents:
        lines.append("Top Grounding Precedent:")
        top_doc = state.retrieved_documents[0]
        lines.append(f"  - [{top_doc.document.doc_id}] ({top_doc.document.category}) Score: {top_doc.score:.2f}")

    if state.should_escalate:
        lines.append(f"Escalation Reason  : {state.escalation_reason}")
        lines.extend([
            "-" * 60,
            "SUGGESTED DRAFT (UNSENT, PENDING HUMAN REVIEW):",
            state.draft_response or "(No response generated)",
            "=" * 60 + "\n",
        ])
    else:
        lines.extend([
            "-" * 60,
            "FINAL AGENT RESPONSE (AUTO-HANDLED):",
            state.draft_response or "(No response generated)",
            "=" * 60 + "\n",
        ])
    return "\n".join(lines)


def run_repl(pipeline: SupportAgentPipeline) -> None:
    """Interactive CLI REPL for testing the support agent in real time."""
    print("\nStarting Support Agent Interactive REPL (type 'exit' or 'quit' to stop)...")
    print("-" * 60)
    while True:
        try:
            query = input("\nCustomer Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting REPL.")
                break

            state = pipeline.process_query(query)
            print(format_state_output(state))
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main() -> None:
    """CLI argument parser and dispatcher."""
    parser = argparse.ArgumentParser(
        description="Run the AI Customer Support Agent pipeline on customer queries."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single customer query string to process through the pipeline.",
    )
    group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch an interactive terminal REPL session.",
    )
    group.add_argument(
        "--batch",
        type=str,
        help="Path to JSON or JSONL file containing customer queries to process in batch.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Instantiate pipeline with demo grounding store
    store = setup_demo_store()
    pipeline = SupportAgentPipeline(store=store)

    if args.interactive:
        run_repl(pipeline)
    elif args.query:
        state = pipeline.process_query(args.query)
        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(format_state_output(state))
    elif args.batch:
        print(f"Batch processing from {args.batch} is not yet implemented.")
        sys.exit(1)


if __name__ == "__main__":
    main()
