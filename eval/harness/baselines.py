"""
Benchmark Baselines for AI Customer Support Agent Evaluation.

Implements two deterministic baseline agents evaluated against the golden set:
1. Trivial Baseline:
   - Majority-class intent prediction (or empirical training majority).
   - Single fixed canned template reply.
   - Fixed escalation policy (Always-Escalate vs. Never-Escalate with explicit justification).
2. Simple Baseline:
   - Deterministic keyword / regex intent classifier (no LLM, no embeddings).
   - Intent-specific canned template replies mapped to the 12-intent taxonomy.
   - Simple rule-based escalation policy gating on high-risk intents and escalation keywords.

Computes accuracy, per-intent F1, and complete confusion matrices saved to
eval/harness/baseline_results.json to establish the empirical floor.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from intents.taxonomy import IntentCategory, SUPPORT_TAXONOMY

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("eval/golden_set/golden_examples.jsonl")
RESULTS_OUTPUT_PATH = Path("eval/harness/baseline_results.json")


# ---------------------------------------------------------------------------
# 1. Trivial Baseline Implementation
# ---------------------------------------------------------------------------

class TrivialBaseline:
    """
    Trivial baseline agent representing the absolute simplest heuristic floor.

    Components:
    - Intent: Predicts the majority class across all queries.
    - Response: Emits a single fixed canned response template.
    - Escalation: Fixed static policy.
      Default: 'always_escalate' (Safety-First Floor).
      Justification: In enterprise customer support, if an automated system possesses
      zero classification or grounding capability, defaulting to 'never-escalate' creates
      catastrophic compliance, legal, and fraud liability (100% missed escalation of account
      takeovers and fraudulent charges). 'Always-escalate' represents the defensible
      fail-safe floor (0% missed escalation, at the cost of 0% automated deflection).
      For completeness, 'never_escalate' is also computed as the naive deflection floor.
    """

    CANNED_REPLY: str = (
        "Thank you for contacting customer support. We have received your inquiry. "
        "You can check your order status, track shipments, and find answers to common questions "
        "anytime in our Help Center at amazon.com/help."
    )

    def __init__(
        self,
        majority_intent: str = "order_status_delay",
        escalation_mode: str = "always_escalate",
    ) -> None:
        """
        Initialize TrivialBaseline.

        Args:
            majority_intent: Default intent prediction (domain majority: order_status_delay).
            escalation_mode: 'always_escalate' or 'never_escalate'.
        """
        self.majority_intent = majority_intent
        self.escalation_mode = escalation_mode

    def process(self, text: str) -> Dict[str, Any]:
        """Process a single message through the trivial baseline."""
        should_escalate = (self.escalation_mode == "always_escalate")
        reason = (
            "Trivial baseline fixed policy: Always escalate to human agent for safety."
            if should_escalate
            else "Trivial baseline fixed policy: Never escalate; attempt 100% autonomous deflection."
        )

        reply = (
            "I am transferring your request to a support specialist who can assist you directly."
            if should_escalate
            else self.CANNED_REPLY
        )

        return {
            "predicted_intent": self.majority_intent,
            "predicted_escalate": should_escalate,
            "escalation_reason": reason,
            "response": reply,
        }


# ---------------------------------------------------------------------------
# 2. Simple Baseline Implementation
# ---------------------------------------------------------------------------

class SimpleBaseline:
    """
    Simple rule-based baseline agent without machine learning or LLM calls.

    Components:
    - Intent: Deterministic keyword and regex matching against the 12 taxonomy categories.
    - Response: Intent-specific canned response templates derived from verified brand patterns.
    - Escalation: Rule-based policy escalating high-risk intents and explicit escalation triggers.
    """

    # Intent-specific resolution templates
    TEMPLATES: Dict[str, str] = {
        "order_status_delay": (
            "You can track the real-time status and estimated delivery date for your order "
            "under 'Your Orders' at amazon.com/orders. If your package is delayed past the delivery "
            "window, carrier tracking will update within 24-48 hours."
        ),
        "delivery_not_received": (
            "If tracking indicates your package was delivered but you haven't received it, "
            "please check safe spots around your home, porch, or with neighbors. Carriers occasionally "
            "mark parcels delivered up to 36 hours prior to actual arrival."
        ),
        "returns_refunds": (
            "Items in original condition can be returned within 30 days of receipt. "
            "To generate a prepaid return shipping label or check refund status, visit the "
            "Online Returns Center at amazon.com/returns."
        ),
        "damaged_defective_wrong_item": (
            "We apologize for the issue with your item. You can request an immediate free replacement "
            "or return the damaged item via our Online Returns Center at amazon.com/returns."
        ),
        "billing_payment_disputes": (
            "For your security, please never share credit card or banking details publicly. "
            "Billing discrepancies and payment disputes require private account verification by a specialist."
        ),
        "prime_subscription_benefits": (
            "Amazon Prime includes fast shipping, Prime Video, and exclusive member deals. "
            "You can manage your membership, payment settings, and plan benefits at amazon.com/mc."
        ),
        "account_access_security": (
            "If you are experiencing login issues, 2FA failures, or suspect unauthorized account access, "
            "please do not post credentials publicly. Visit amazon.com/help for secure identity recovery."
        ),
        "carrier_physical_delivery_issue": (
            "We take carrier safety, conduct complaints, and delivery incidents very seriously. "
            "Your report is being forwarded to logistics operations for immediate investigation."
        ),
        "order_cancellation_modification": (
            "Orders can be cancelled before dispatch in 'Your Orders' by selecting 'Cancel Items'. "
            "If the package has already shipped, you may refuse delivery or request a return label upon arrival."
        ),
        "customer_service_escalation": (
            "We sincerely apologize for your frustrating support experience. We are prioritizing "
            "your issue and connecting you with a senior supervisor to resolve this directly."
        ),
        "product_inquiry_availability": (
            "Product compatibility details and dimensions are listed under Product Details on the item page. "
            "For items temporarily out of stock, add the item to your Wishlist to receive an in-stock notification."
        ),
        "other_unclear": (
            "Thank you for contacting us. Could you please provide a few more details or clarify your request "
            "so that we can direct you to the right support resource?"
        ),
    }

    # Keyword and regex patterns for intent classification
    INTENT_RULES: List[Tuple[str, List[re.Pattern]]] = [
        (
            "billing_payment_disputes",
            [
                re.compile(r"\b(charged(\s+\w+)?\s+(twice|3\s+times|3x|again|extra)|unauthorized\s+charge|double\s+charge)\b", re.I),
                re.compile(r"\b(dispute\s+the\s+charge|unauthorized\s+transaction|taking\s+money\s+from\s+my\s+bank)\b", re.I),
                re.compile(r"\b(credit\s+card\s+charge|snatch\s+the\s+money|deducted\s+money)\b", re.I),
            ],
        ),
        (
            "account_access_security",
            [
                re.compile(r"\b(locked\s+out|lock\s+my\s+account|account\s+locked|cant\s+log\s*in|can't\s+log\s*in)\b", re.I),
                re.compile(r"\b(password\s+reset|otp\s+messages|two-step|2fa|tfa\s+code|account\s+hacked)\b", re.I),
                re.compile(r"\b(disable\s+tfa|recovery\s+phone|compromised\s+credentials)\b", re.I),
            ],
        ),
        (
            "carrier_physical_delivery_issue",
            [
                re.compile(r"\b(delivery\s+(guy|driver|agent)|driver)\b.*\b(puppy|dog|threw|throw|gate|rude|car|damage|police)\b", re.I),
                re.compile(r"\b(leaves?\s+the\s+gate\s+open|parcel\s+over\s+the\s+fence|reckless\s+driving)\b", re.I),
            ],
        ),
        (
            "customer_service_escalation",
            [
                re.compile(r"\b(worst|terrible|awful|disgusting|horrible)\s+customer\s+service\b", re.I),
                re.compile(r"\b(talk\s+to\s+a\s+(manager|supervisor)|speak\s+with\s+a\s+human|demand\s+attention)\b", re.I),
                re.compile(r"\b(filing\s+a\s+complaint|attorney|lawyer|sue\s+you|ftc\s+complaint)\b", re.I),
            ],
        ),
        (
            "delivery_not_received",
            [
                re.compile(r"\b(showing|says|marked|status)\b.*\bdelivered\b.*\b(never|not|haven't|did\s*not)\s+(received|get|got|arrive)\b", re.I),
                re.compile(r"\bmarked\s+delivered\b.*\b(porch|stolen|missing|not\s+here)\b", re.I),
                re.compile(r"\bdelivered\b.*\bwhere\s+is\s+it\b", re.I),
            ],
        ),
        (
            "order_cancellation_modification",
            [
                re.compile(r"\b(cancel\s+(this|my|the)\s+order|how\s+(do\s+i|i)\s+cancel\s+my\s+order)\b", re.I),
                re.compile(r"\b(change\s+(the\s+)?delivery\s+address|modify\s+order|wrong\s+address)\b", re.I),
            ],
        ),
        (
            "damaged_defective_wrong_item",
            [
                re.compile(r"\b(damaged|broken|shattered|cracked|defective|faulty|not\s+working)\b", re.I),
                re.compile(r"\b(wrong\s+(size|item|color|product)|received\s+(the\s+)?wrong)\b", re.I),
                re.compile(r"\b(zipper\s+broke|screen\s+cracked|missing\s+parts)\b", re.I),
            ],
        ),
        (
            "returns_refunds",
            [
                re.compile(r"\b(return|refund|reimburse|money\s+back|cashback|exchange)\b", re.I),
                re.compile(r"\b(how\s+do\s+i\s+return|where\s+is\s+my\s+refund|refund\s+not\s+received)\b", re.I),
                re.compile(r"\breturn\s+(label|item|package)\b", re.I),
            ],
        ),
        (
            "order_status_delay",
            [
                re.compile(r"\b(where('?s|\s+is)\s+my\s+(package|order|delivery|item)|tracking\s+number)\b", re.I),
                re.compile(r"\b(was\s+supposed\s+to\s+be\s+delivered|supposed\s+to\s+arrive|delayed|running\s+late)\b", re.I),
                re.compile(r"\b(when\s+will\s+it\s+arrive|hasn't\s+arrived|not\s+delivered\s+yet|track\s+order)\b", re.I),
                re.compile(r"\b(estimated\s+delivery|one\s+day\s+shipping.*delayed)\b", re.I),
            ],
        ),
        (
            "prime_subscription_benefits",
            [
                re.compile(r"\b(prime\s+video|prime\s+membership|prime\s+benefits|music\s+unlimited)\b", re.I),
                re.compile(r"\b(annual\s+subscription|cloud\s+storage.*prime|watch\s+movies.*prime)\b", re.I),
            ],
        ),
        (
            "product_inquiry_availability",
            [
                re.compile(r"\b(when\s+do\s+y'all\s+restock|restock|when\s+will.*be\s+available)\b", re.I),
                re.compile(r"\b(compatible\s+with|work\s+in\s+ireland|in\s+stock|out\s+of\s+stock)\b", re.I),
            ],
        ),
    ]

    # Mandatory escalation intents and escalation keywords
    ESCALATION_INTENTS = {
        "billing_payment_disputes",
        "account_access_security",
        "carrier_physical_delivery_issue",
        "customer_service_escalation",
    }

    ESCALATION_KEYWORDS = re.compile(
        r"\b(lawyer|attorney|police|fraud|stolen|court|supervisor|manager|fucking|shit|unacceptable|chargeback)\b",
        re.I,
    )

    def classify_intent(self, text: str) -> str:
        """Rule-based pattern matching intent classifier."""
        for intent_name, patterns in self.INTENT_RULES:
            for pat in patterns:
                if pat.search(text):
                    return intent_name
        return "other_unclear"

    def evaluate_escalation(self, text: str, predicted_intent: str) -> Tuple[bool, str]:
        """Simple rule-based escalation policy."""
        # Rule 1: High-risk taxonomy intent
        if predicted_intent in self.ESCALATION_INTENTS:
            return True, f"Rule escalation: Intent '{predicted_intent}' is a high-risk policy category."

        # Rule 2: Escalation keywords or profanity in message
        if self.ESCALATION_KEYWORDS.search(text):
            return True, "Rule escalation: Query contains high-friction sentiment or escalation keywords."

        # Autonomous handling approved
        return False, "Rule: Query matched standard autonomous intent without risk triggers."

    def process(self, text: str) -> Dict[str, Any]:
        """Process a single message through the simple baseline."""
        intent = self.classify_intent(text)
        should_escalate, reason = self.evaluate_escalation(text, intent)

        if should_escalate:
            reply = (
                "Your request has been escalated to a customer support specialist for prioritized review. "
                "A representative will contact you shortly."
            )
        else:
            reply = self.TEMPLATES.get(intent, self.TEMPLATES["other_unclear"])

        return {
            "predicted_intent": intent,
            "predicted_escalate": should_escalate,
            "escalation_reason": reason,
            "response": reply,
        }


# ---------------------------------------------------------------------------
# 3. Evaluation & Metrics Calculation
# ---------------------------------------------------------------------------

@dataclass
class BaselineEvaluationResult:
    """Consolidated metrics for a baseline run."""

    baseline_name: str
    total_samples: int
    intent_accuracy: float
    escalation_accuracy: float
    escalation_precision: float
    escalation_recall: float
    false_escalation_rate: float
    missed_escalation_rate: float
    escalation_confusion: Dict[str, int]
    per_intent_metrics: Dict[str, Dict[str, Any]]
    intent_confusion_matrix: Dict[str, Dict[str, int]]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_name": self.baseline_name,
            "description": self.description,
            "total_samples": self.total_samples,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "escalation_accuracy": round(self.escalation_accuracy, 4),
            "escalation_precision": round(self.escalation_precision, 4),
            "escalation_recall": round(self.escalation_recall, 4),
            "false_escalation_rate": round(self.false_escalation_rate, 4),
            "missed_escalation_rate": round(self.missed_escalation_rate, 4),
            "escalation_confusion": self.escalation_confusion,
            "per_intent_metrics": self.per_intent_metrics,
            "intent_confusion_matrix": self.intent_confusion_matrix,
        }


def load_golden_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from golden set JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_baseline(
    baseline_agent: Any,
    records: List[Dict[str, Any]],
    name: str,
    description: str,
) -> BaselineEvaluationResult:
    """Evaluates a baseline agent on the provided golden records."""
    all_categories = sorted([cat.value for cat in IntentCategory])

    # Confusion matrix structure: true_intent -> {pred_intent: count}
    intent_matrix: Dict[str, Dict[str, int]] = {
        true_c: {pred_c: 0 for pred_c in all_categories} for true_c in all_categories
    }

    intent_tp = Counter()
    intent_fp = Counter()
    intent_fn = Counter()
    intent_support = Counter()

    tp_esc = 0
    fp_esc = 0
    tn_esc = 0
    fn_esc = 0
    correct_intent = 0

    for r in records:
        text = r.get("message_text") or r.get("query", "")
        true_intent = r.get("true_intent") or r.get("expected_intent")
        true_esc = bool(r.get("true_escalate") if "true_escalate" in r else r.get("expected_escalate", False))

        pred = baseline_agent.process(text)
        pred_intent = pred["predicted_intent"]
        pred_esc = bool(pred["predicted_escalate"])

        # Update intent confusion matrix
        if true_intent in intent_matrix and pred_intent in intent_matrix[true_intent]:
            intent_matrix[true_intent][pred_intent] += 1

        intent_support[true_intent] += 1

        if true_intent == pred_intent:
            correct_intent += 1
            intent_tp[true_intent] += 1
        else:
            intent_fn[true_intent] += 1
            intent_fp[pred_intent] += 1

        # Update escalation confusion
        if true_esc and pred_esc:
            tp_esc += 1
        elif not true_esc and not pred_esc:
            tn_esc += 1
        elif not true_esc and pred_esc:
            fp_esc += 1
        elif true_esc and not pred_esc:
            fn_esc += 1

    total = len(records)
    total_auto = tn_esc + fp_esc
    total_esc = tp_esc + fn_esc

    intent_acc = correct_intent / total if total > 0 else 0.0
    esc_acc = (tp_esc + tn_esc) / total if total > 0 else 0.0
    esc_prec = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
    esc_rec = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
    false_esc_rate = fp_esc / total_auto if total_auto > 0 else 0.0
    missed_esc_rate = fn_esc / total_esc if total_esc > 0 else 0.0

    # Per-intent metrics
    per_intent = {}
    for cat in all_categories:
        tp = intent_tp[cat]
        fp = intent_fp[cat]
        fn = intent_fn[cat]
        sup = intent_support[cat]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * rc) / (p + rc) if (p + rc) > 0 else 0.0
        per_intent[cat] = {
            "precision": round(p, 4),
            "recall": round(rc, 4),
            "f1": round(f1, 4),
            "support": sup,
        }

    return BaselineEvaluationResult(
        baseline_name=name,
        total_samples=total,
        intent_accuracy=intent_acc,
        escalation_accuracy=esc_acc,
        escalation_precision=esc_prec,
        escalation_recall=esc_rec,
        false_escalation_rate=false_esc_rate,
        missed_escalation_rate=missed_esc_rate,
        escalation_confusion={
            "true_positives": tp_esc,
            "false_positives": fp_esc,
            "true_negatives": tn_esc,
            "false_negatives": fn_esc,
        },
        per_intent_metrics=per_intent,
        intent_confusion_matrix=intent_matrix,
        description=description,
    )


# ---------------------------------------------------------------------------
# 4. CLI Execution and Report Generator
# ---------------------------------------------------------------------------

def print_baseline_summary(res: BaselineEvaluationResult) -> None:
    """Clean terminal printout of evaluation results."""
    print("\n" + "=" * 75)
    print(f"BASELINE: {res.baseline_name.upper()}")
    print("=" * 75)
    print(f"Description            : {res.description}")
    print(f"Total Test Cases       : {res.total_samples}")
    print(f"Intent Accuracy        : {res.intent_accuracy:.2%}")
    print(f"Escalation Accuracy    : {res.escalation_accuracy:.2%}")
    print(f"Escalation Precision   : {res.escalation_precision:.2%}")
    print(f"Escalation Recall      : {res.escalation_recall:.2%}")
    print(f"False Escalation Rate  : {res.false_escalation_rate:.2%}  (Unnecessary escalation)")
    print(f"Missed Escalation Rate : {res.missed_escalation_rate:.2%}  (Critical safety risk)")

    conf = res.escalation_confusion
    print("\n--- ESCALATION CONFUSION MATRIX ---")
    print(f"True Escalations  (TP) : {conf['true_positives']:>4}")
    print(f"False Escalations (FP) : {conf['false_positives']:>4}")
    print(f"True Auto-Handles (TN) : {conf['true_negatives']:>4}")
    print(f"Missed Escalations(FN) : {conf['false_negatives']:>4}  (Safety risk)")

    print("\n--- PER-INTENT METRICS ---")
    print(f"{'Intent Category':<35} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Support':<7}")
    print("-" * 72)
    for intent, m in res.per_intent_metrics.items():
        print(
            f"{intent:<35} | "
            f"{m['precision']:>5.2f} | "
            f"{m['recall']:>5.2f} | "
            f"{m['f1']:>5.2f} | "
            f"{m['support']:>7}"
        )


def run_all_baselines(
    golden_set_path: Path = GOLDEN_SET_PATH,
    output_path: Path = RESULTS_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Runs all baselines, prints comparison, and saves baseline_results.json."""
    records = load_golden_records(golden_set_path)
    print(f"Loaded {len(records)} golden records from {golden_set_path}")

    # Determine empirical majority class from the golden set
    counts = Counter(r.get("true_intent") for r in records)
    majority_intent, maj_count = counts.most_common(1)[0]
    print(f"Top Intent in Golden Set: '{majority_intent}' ({maj_count} / {len(records)} = {maj_count/len(records):.1%})")

    # 1. Trivial Baseline (Always-Escalate: Defensible Safety-First Policy)
    trivial_always = TrivialBaseline(majority_intent=majority_intent, escalation_mode="always_escalate")
    res_trivial_always = evaluate_baseline(
        trivial_always,
        records,
        name="trivial_always_escalate",
        description=(
            "Majority-class intent prediction ('other_unclear') + single canned template + "
            "Always-Escalate policy. Defensible fail-safe floor (0.0% missed escalation safety floor, "
            "0% automated deflection)."
        ),
    )

    # 1b. Trivial Baseline (Never-Escalate: Naive Full Deflection Floor)
    trivial_never = TrivialBaseline(majority_intent=majority_intent, escalation_mode="never_escalate")
    res_trivial_never = evaluate_baseline(
        trivial_never,
        records,
        name="trivial_never_escalate",
        description=(
            "Majority-class intent prediction ('other_unclear') + single canned template + "
            "Never-Escalate policy. Naive full deflection bot floor (100.0% missed escalation failure)."
        ),
    )

    # 2. Simple Baseline (Keyword/Regex Classifier + Canned Templates + Rule Escalation)
    simple = SimpleBaseline()
    res_simple = evaluate_baseline(
        simple,
        records,
        name="simple_keyword_baseline",
        description=(
            "Deterministic keyword/regex intent classifier + intent-specific canned response templates + "
            "rule-based escalation on high-risk intents and keywords. Classical non-ML floor."
        ),
    )

    # Print results
    print_baseline_summary(res_trivial_always)
    print_baseline_summary(res_trivial_never)
    print_baseline_summary(res_simple)

    # Comparison summary table
    print("\n" + "=" * 85)
    print("BASELINE COMPARISON: THE PERFORMANCE FLOOR TO BEAT")
    print("=" * 85)
    header = f"{'Metric':<25} | {'Trivial (Always-Esc)':<20} | {'Trivial (Never-Esc)':<20} | {'Simple (Keyword)':<20}"
    print(header)
    print("-" * 85)
    print(f"{'Intent Accuracy':<25} | {res_trivial_always.intent_accuracy:>18.2%} | {res_trivial_never.intent_accuracy:>18.2%} | {res_simple.intent_accuracy:>18.2%}")
    print(f"{'Escalation Accuracy':<25} | {res_trivial_always.escalation_accuracy:>18.2%} | {res_trivial_never.escalation_accuracy:>18.2%} | {res_simple.escalation_accuracy:>18.2%}")
    print(f"{'Escalation Precision':<25} | {res_trivial_always.escalation_precision:>18.2%} | {res_trivial_never.escalation_precision:>18.2%} | {res_simple.escalation_precision:>18.2%}")
    print(f"{'Escalation Recall':<25} | {res_trivial_always.escalation_recall:>18.2%} | {res_trivial_never.escalation_recall:>18.2%} | {res_simple.escalation_recall:>18.2%}")
    print(f"{'Missed Escalation Rate':<25} | {res_trivial_always.missed_escalation_rate:>18.2%} | {res_trivial_never.missed_escalation_rate:>18.2%} | {res_simple.missed_escalation_rate:>18.2%}")
    print(f"{'False Escalation Rate':<25} | {res_trivial_always.false_escalation_rate:>18.2%} | {res_trivial_never.false_escalation_rate:>18.2%} | {res_simple.false_escalation_rate:>18.2%}")
    print("=" * 85)

    # Serialize to JSON
    output_data = {
        "metadata": {
            "golden_set_path": str(golden_set_path),
            "total_records": len(records),
            "top_majority_intent": majority_intent,
            "timestamp": "2026-09-17",
        },
        "baselines": {
            "trivial_baseline_always_escalate": res_trivial_always.to_dict(),
            "trivial_baseline_never_escalate": res_trivial_never.to_dict(),
            "simple_keyword_baseline": res_simple.to_dict(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved complete baseline evaluation results to {output_path}\n")
    return output_data


def main() -> None:
    """CLI parser for baselines execution."""
    parser = argparse.ArgumentParser(description="Run baseline evaluations against the golden set.")
    parser.add_argument(
        "--golden-set",
        "-g",
        type=Path,
        default=GOLDEN_SET_PATH,
        help="Path to golden set JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=RESULTS_OUTPUT_PATH,
        help="Path to output JSON file.",
    )
    args = parser.parse_args()
    run_all_baselines(args.golden_set, args.output)


if __name__ == "__main__":
    main()
