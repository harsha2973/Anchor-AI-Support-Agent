"""
Builder script for generating the comprehensive 200-example Golden Evaluation Benchmark.

Extracts stratified authentic interactions from the AmazonHelp dialogue corpus,
deliberately oversampling edge cases (ambiguous, multi-issue, angry/profane,
low-signal, and safety-critical) and exports to eval/golden_set/golden_examples.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from intents.taxonomy import IntentCategory, RiskLevel, SUPPORT_TAXONOMY

logger = logging.getLogger(__name__)

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def is_english_text(text: str) -> bool:
    """Filter out non-Latin / non-English scripts."""
    if not text or len(text.strip()) < 5:
        return False
    # Check ASCII printable ratio
    ascii_ratio = sum(ord(c) < 128 for c in text) / len(text)
    return ascii_ratio > 0.85


def assign_escalation_and_reason(
    category: IntentCategory,
    query: str,
    difficulty: str,
) -> Tuple[bool, str]:
    """Determine ground truth escalation decision and explicit audit reason."""
    lowered = query.lower()

    # Rule 1: Mandatory escalation intents per business policy
    if category == IntentCategory.BILLING_PAYMENT_DISPUTES:
        return True, "Mandatory human escalation: Involves unauthorized debits, payment disputes, or financial transactions requiring secure billing verification."
    if category == IntentCategory.ACCOUNT_ACCESS_SECURITY:
        return True, "Mandatory human escalation: Account compromised, locked out, or 2FA failure; requires identity verification by security team."
    if category == IntentCategory.CUSTOMER_SERVICE_ESCALATION:
        return True, "Mandatory human escalation: Severe customer dissatisfaction, previous agent failure, or legal/regulatory threat requiring supervisory review."
    if category == IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE:
        return True, "Mandatory human escalation: Driver safety incident, conduct complaint, or property/pet damage requiring logistics operations review."

    # Rule 2: Emotion & sentiment extremes or high risk tiers
    if difficulty == "angry_profane":
        return True, "Mandatory human escalation: Severe frustration, abusive language, or profanity requiring empathetic human de-escalation."
    if difficulty == "high_risk":
        return True, "Mandatory human escalation: High-risk scenario requiring human review and verification."

    # Rule 3: Multi-issue high complexity
    if difficulty == "multi_issue" and any(w in lowered for w in ["attorney", "lawyer", "manager", "dispute", "refund", "cancel"]):
        return True, "High friction / legal / multi-issue complexity requiring human agent de-escalation."

    # Rule 4: Standard autonomous handling
    if category == IntentCategory.ORDER_STATUS_DELAY:
        return False, "Autonomous handling: In-transit tracking inquiry resolvable with real-time tracking link and estimated delivery date confirmation."
    if category == IntentCategory.DELIVERY_NOT_RECEIVED:
        if any(w in lowered for w in ["cancel it", "refund the amount", "never got it", "dispute"]):
            return True, "Human escalation: Customer demands immediate refund/cancellation on package already marked delivered."
        return False, "Autonomous handling: Premature scan guidance advising 24-36h window and missing package search tips."
    if category == IntentCategory.RETURNS_REFUNDS:
        if "cashback" in lowered and "months" in lowered:
            return True, "Human escalation: Prolonged cashback dispute exceeding normal policy resolution windows."
        return False, "Autonomous handling: Standard 30-day return policy and Online Returns Center label generation instructions."
    if category == IntentCategory.DAMAGED_DEFECTIVE_WRONG_ITEM:
        return False, "Autonomous handling: Damaged or defective product replacement initiated through standard Returns Center portal."
    if category == IntentCategory.PRIME_SUBSCRIPTION_BENEFITS:
        return False, "Autonomous handling: Prime Video, Music Unlimited, or membership benefit FAQ covered by knowledge base."
    if category == IntentCategory.ORDER_CANCELLATION_MODIFICATION:
        return False, "Autonomous handling: Standard cancellation guidance via Your Orders before dispatch."
    if category == IntentCategory.PRODUCT_INQUIRY_AVAILABILITY:
        return False, "Autonomous handling: Product compatibility or restock alert guidance covered by knowledge base."
    if category == IntentCategory.OTHER_UNCLEAR:
        return False, "Autonomous handling: Polite clarification request or redirect to Amazon Seller Central."

    return False, "Autonomous handling: Request meets standard policy criteria for automated guidance."


def build_full_golden_dataset(
    parquet_path: Path = Path("data/processed/amazon_support_pairs.parquet"),
    output_jsonl_path: Path = Path("eval/golden_set/golden_examples.jsonl"),
    target_count: int = 200,
) -> List[Dict[str, Any]]:
    """Builds and writes the complete 200-example golden set."""
    logger.info("Loading processed corpus from %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Filter to English records with non-empty replies
    df = df[df["customer_message"].apply(is_english_text) & (df["brand_reply"].str.len() > 10)].copy()

    records: List[Dict[str, Any]] = []
    used_indices: Set[int] = set()

    # Define stratum search definitions with difficulty tags
    strata: List[Dict[str, Any]] = [
        # 1. ORDER_STATUS_DELAY (~24)
        {
            "intent": IntentCategory.ORDER_STATUS_DELAY,
            "patterns": [
                (r"\bwhere\s+is\s+my\b", "routine", 8),
                (r"\b(tracking\s+number|track\s+my\s+order)\b", "routine", 6),
                (r"\bwas\s+supposed\s+to\s+arrive\b", "ambiguous", 5),
                (r"\b(one\s+day\s+shipping|next\s+day).*delayed\b", "multi_issue", 5),
            ],
        },
        # 2. DELIVERY_NOT_RECEIVED (~20)
        {
            "intent": IntentCategory.DELIVERY_NOT_RECEIVED,
            "patterns": [
                (r"\b(marked|says|said)\s+delivered\b.*\b(not\s+here|haven't|never|nothing)\b", "routine", 8),
                (r"\bporch\b.*\b(stolen|missing|not\s+here)\b", "ambiguous", 5),
                (r"\bdelivered\b.*\b(cancel|refund)\b", "multi_issue", 4),
                (r"\bfucking\s+not.*delivered\b", "angry_profane", 3),
            ],
        },
        # 3. RETURNS_REFUNDS (~20)
        {
            "intent": IntentCategory.RETURNS_REFUNDS,
            "patterns": [
                (r"\bhow\s+do\s+i\s+return\b", "routine", 8),
                (r"\breturn\s+label\b", "routine", 6),
                (r"\bmoney\s+is\s+not\s+refunded\b.*\b(month|weeks)\b", "ambiguous", 3),
                (r"\bcashback\b.*\bmonths\b", "multi_issue", 3),
            ],
        },
        # 4. DAMAGED_DEFECTIVE_WRONG_ITEM (~20)
        {
            "intent": IntentCategory.DAMAGED_DEFECTIVE_WRONG_ITEM,
            "patterns": [
                (r"\b(arrived\s+damaged|broken|shattered|cracked)\b", "routine", 8),
                (r"\b(wrong\s+size|wrong\s+item|different\s+item)\b", "routine", 6),
                (r"\bzipper\s+broke|defective\b.*\b(exchange|replace)\b", "ambiguous", 4),
                (r"\bfooled\s+me|poor\s+experience\b.*\bbattery\b", "multi_issue", 2),
            ],
        },
        # 5. BILLING_PAYMENT_DISPUTES (~22)
        {
            "intent": IntentCategory.BILLING_PAYMENT_DISPUTES,
            "patterns": [
                (r"\b(unauthorized\s+charge|fraud|dispute\s+the\s+whole\s+charge)\b", "high_risk", 6),
                (r"\bcharged\s+(\w+\s+)?twice\b", "high_risk", 6),
                (r"\bcharged\s+3\s+times\b", "multi_issue", 4),
                (r"\bautomatically\s+taking\s+money\b.*\bbank\b", "ambiguous", 4),
                (r"\bpayment\s+declined|payment\s+failed\b", "routine", 2),
            ],
        },
        # 6. PRIME_SUBSCRIPTION_BENEFITS (~16)
        {
            "intent": IntentCategory.PRIME_SUBSCRIPTION_BENEFITS,
            "patterns": [
                (r"\bdo\s+u\s+i\s+need\s+to\s+pay\s+extra\b.*\bmovies\b", "routine", 4),
                (r"\bprime\s+video\b.*\bavailable\b", "routine", 4),
                (r"\bmusic\s+unlimited\b.*\bdevices\b", "routine", 4),
                (r"\bwhy\s+does\s+prime\s+member\b.*\bnot\s+give\b", "ambiguous", 4),
            ],
        },
        # 7. ACCOUNT_ACCESS_SECURITY (~18)
        {
            "intent": IntentCategory.ACCOUNT_ACCESS_SECURITY,
            "patterns": [
                (r"\blocked\s+my\s+account|can't\s+get\s+into\b", "high_risk", 6),
                (r"\b(password\s+assistance|otp|2fa|tfa)\b", "high_risk", 6),
                (r"\bhacked\b", "high_risk", 4),
                (r"\brecovery\s+phone\b", "ambiguous", 2),
            ],
        },
        # 8. CARRIER_PHYSICAL_DELIVERY_ISSUE (~16)
        {
            "intent": IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE,
            "patterns": [
                (r"\b(puppy|police|hit\s+my\s+car|gate\s+open)\b", "high_risk", 4),
                (r"\bthrew\s+(the\s+)?(parcel|box|package)\b", "angry_profane", 4),
                (r"\bring\s+door\s*bell\b", "routine", 4),
                (r"\bdelivery\s+guy\b.*\b(rude|complaint)\b", "ambiguous", 4),
            ],
        },
        # 9. ORDER_CANCELLATION_MODIFICATION (~14)
        {
            "intent": IntentCategory.ORDER_CANCELLATION_MODIFICATION,
            "patterns": [
                (r"\bcancel\s+my\s+order\s+immediately\b", "routine", 5),
                (r"\bhow\s+(do\s+)?i\s+cancel\s+my\s+order\b", "routine", 4),
                (r"\bchange\s+(my\s+)?(delivery\s+)?address\b", "ambiguous", 3),
                (r"\bnot\s+able\s+to\s+cancel\b", "multi_issue", 2),
            ],
        },
        # 10. CUSTOMER_SERVICE_ESCALATION (~16)
        {
            "intent": IntentCategory.CUSTOMER_SERVICE_ESCALATION,
            "patterns": [
                (r"\b(attorney|lawyer|ftc|better\s+business\s+bureau)\b", "high_risk", 4),
                (r"\b(worst\s+customer\s+service|pathetic\s+service|service\s+sucks)\b", "angry_profane", 5),
                (r"\b(talk\s+to\s+somebody|speak\s+to\s+a\s+manager|supervisor)\b", "ambiguous", 4),
                (r"\bwhat\s+a\s+joke.*team\s+i've\s+come\s+across\b", "angry_profane", 3),
            ],
        },
        # 11. PRODUCT_INQUIRY_AVAILABILITY (~12)
        {
            "intent": IntentCategory.PRODUCT_INQUIRY_AVAILABILITY,
            "patterns": [
                (r"\bwhen\s+do\s+y'all\s+restock\b", "routine", 4),
                (r"\bcompatible\s+with\b", "routine", 4),
                (r"\bwhen\s+will\b.*\bbe\s+available\b", "ambiguous", 4),
            ],
        },
        # 12. OTHER_UNCLEAR (~12)
        {
            "intent": IntentCategory.OTHER_UNCLEAR,
            "patterns": [
                (r"^(link\?+|\?+|help\?+|dm)$", "low_signal", 4),
                (r"\bamazon\s+seller\s+account\b", "low_signal", 4),
                (r"\b(dauerschleife|sonntag|relaxing|cup\s+of\s+tea)\b", "low_signal", 4),
            ],
        },
    ]

    for stratum in strata:
        cat = stratum["intent"]
        for regex_str, difficulty, quota in stratum["patterns"]:
            pat = re.compile(regex_str, re.IGNORECASE)
            matched = df[df["customer_message"].str.contains(pat, regex=True) & (~df.index.isin(used_indices))]

            sampled = matched.head(quota)
            for idx_val, row in sampled.iterrows():
                used_indices.add(idx_val)
                msg_text = str(row["customer_message"]).strip()
                b_reply = str(row["brand_reply"]).strip()
                ctx = str(row["thread_context"]).strip()
                turn_idx = int(row["turn_index"])

                escalate, reason = assign_escalation_and_reason(cat, msg_text, difficulty)

                records.append({
                    "id": f"GOLDEN_{len(records) + 1:03d}",
                    "message_text": msg_text,
                    "thread_context": ctx,
                    "turn_index": turn_idx,
                    "true_intent": cat.value,
                    "true_escalate": escalate,
                    "escalate_reason": reason,
                    "difficulty": difficulty,
                    "review_status": "auto_labeled_pending_review",
                    "reference_brand_reply": b_reply,
                })

    # If count is below target_count, fill remaining slots with stratified sampling across underrepresented intents
    if len(records) < target_count:
        needed = target_count - len(records)
        logger.info("Filling remaining %d records to reach target %d ...", needed, target_count)
        pool = df[~df.index.isin(used_indices)].sample(needed, random_state=42)
        for idx_val, row in pool.iterrows():
            used_indices.add(idx_val)
            msg_text = str(row["customer_message"]).strip()
            b_reply = str(row["brand_reply"]).strip()
            ctx = str(row["thread_context"]).strip()
            turn_idx = int(row["turn_index"])

            # Classify using heuristic for draft labeling
            from intents.classifier import CalibratedHeuristicClassifier
            pred = CalibratedHeuristicClassifier().classify(msg_text)
            diff = "ambiguous" if pred.is_ambiguous else "routine"
            escalate, reason = assign_escalation_and_reason(pred.predicted_category, msg_text, diff)

            records.append({
                "id": f"GOLDEN_{len(records) + 1:03d}",
                "message_text": msg_text,
                "thread_context": ctx,
                "turn_index": turn_idx,
                "true_intent": pred.predicted_category.value,
                "true_escalate": escalate,
                "escalate_reason": reason,
                "difficulty": diff,
                "review_status": "auto_labeled_pending_review",
                "reference_brand_reply": b_reply,
            })

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Successfully generated %d golden records at %s", len(records), output_jsonl_path)
    return records


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    records = build_full_golden_dataset(target_count=200)

    # Print summary statistics
    df_golden = pd.DataFrame(records)
    print("\n" + "=" * 65)
    print("GOLDEN EVALUATION SET GENERATION SUMMARY")
    print("=" * 65)
    print(f"Total Records Generated   : {len(df_golden)}")
    print(f"Escalation Distribution   : Escalate = {sum(df_golden['true_escalate'])} ({(sum(df_golden['true_escalate'])/len(df_golden))*100:.1f}%), Auto = {sum(~df_golden['true_escalate'])} ({(sum(~df_golden['true_escalate'])/len(df_golden))*100:.1f}%)")
    print("\nDifficulty Tier Breakdown:")
    for diff, cnt in df_golden["difficulty"].value_counts().items():
        print(f"  - {diff:<15} : {cnt:>3} ({cnt/len(df_golden)*100:.1f}%)")

    print("\nIntent Distribution:")
    for intent, cnt in df_golden["true_intent"].value_counts().items():
        print(f"  - {intent:<35} : {cnt:>3} ({cnt/len(df_golden)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
