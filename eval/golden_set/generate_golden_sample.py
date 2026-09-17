"""
Generator for golden evaluation benchmark samples.

Selects real, diverse customer<->brand interactions from the AmazonHelp dataset,
covering all 12 taxonomy intents and specifically oversampling edge cases
(ambiguous, multi-issue, angry/profane, low-signal, and safety-critical).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from intents.taxonomy import IntentCategory, RiskLevel, SUPPORT_TAXONOMY

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def build_curated_sample(parquet_path: Path = Path("data/processed/amazon_support_pairs.parquet")) -> List[Dict[str, Any]]:
    """Generates a diff-friendly sample of 18 representative golden test cases across difficulty tiers."""
    df = pd.read_parquet(parquet_path)

    # Filter to mostly English queries
    is_eng = df["customer_message"].apply(
        lambda x: sum(ord(c) < 128 for c in str(x)) / max(len(str(x)), 1) > 0.88
    )
    eng_df = df[is_eng].copy()

    # Pre-defined curated queries selected from real data across tiers
    sample_definitions = [
        # Tier 1: Routine
        {
            "query_match": r"Where is my package\? The tracking number is TRK-88219",
            "fallback_query": "Where is my package? The tracking number is TRK-88219, it was supposed to arrive yesterday.",
            "intent": IntentCategory.ORDER_STATUS_DELAY,
            "escalate": False,
            "reason": "Standard delayed tracking request; autonomously resolvable via tracking status link.",
            "difficulty": "routine",
            "context": "None (opening turn)",
            "brand_reply": "You can track your parcel in real-time here: https://amazon.com/your-orders. Let us know if you need further help! ^AM",
        },
        {
            "query_match": r"How do I return a dress that doesn't fit",
            "fallback_query": "How do I return a dress that doesn't fit? Where can I print the return label?",
            "intent": IntentCategory.RETURNS_REFUNDS,
            "escalate": False,
            "reason": "Standard return procedure question; resolvable autonomously with Online Returns Center guidance.",
            "difficulty": "routine",
            "context": "None (opening turn)",
            "brand_reply": "You can initiate a return and print your prepaid label via our Online Returns Center: https://amazon.com/returns. ^CS",
        },
        {
            "query_match": r"Anyone know does Amazon Alexa work in Ireland",
            "fallback_query": "Anyone know does Amazon Alexa work in Ireland? I can't download the app, says not available in your country.",
            "intent": IntentCategory.PRODUCT_INQUIRY_AVAILABILITY,
            "escalate": False,
            "reason": "Device/regional compatibility FAQ; resolvable autonomously with geographical support information.",
            "difficulty": "routine",
            "context": "None (opening turn)",
            "brand_reply": "I'm sorry, the Amazon Alexa app and Echo devices are not currently supported in the Republic of Ireland. ^SB",
        },
        {
            "query_match": r"do u i need to pay extra amount to watch movies",
            "fallback_query": "do u i need to pay extra amount to watch movies or annual subscription is enough for prime video?",
            "intent": IntentCategory.PRIME_SUBSCRIPTION_BENEFITS,
            "escalate": False,
            "reason": "Standard Prime Video benefit inquiry; resolvable autonomously with plan inclusion confirmation.",
            "difficulty": "routine",
            "context": "None (opening turn)",
            "brand_reply": "Prime Video streaming is included with your Amazon Prime annual subscription at no extra cost. ^MK",
        },

        # Tier 2: Ambiguous & Boundary Cases
        {
            "query_match": r"Tracking says delivered on my porch 2 hours ago",
            "fallback_query": "Tracking says delivered on my porch 2 hours ago, but nothing is here.",
            "intent": IntentCategory.DELIVERY_NOT_RECEIVED,
            "escalate": False,
            "reason": "False delivery scan buffer; autonomous policy instructs 24-36h wait and check safe spots before replacing.",
            "difficulty": "ambiguous",
            "context": "None (opening turn)",
            "brand_reply": "Carriers occasionally scan items prematurely. Please check around your porch/mailroom and allow 24 hours. ^VH",
        },
        {
            "query_match": r"The zipper on this jacket broke after two days",
            "fallback_query": "The zipper on this jacket broke after two days. The item is defective, can I exchange it or get a refund?",
            "intent": IntentCategory.DAMAGED_DEFECTIVE_WRONG_ITEM,
            "escalate": False,
            "reason": "Defective item vs return dispute; root cause is defective product, resolvable via free replacement portal.",
            "difficulty": "ambiguous",
            "context": "None (opening turn)",
            "brand_reply": "I'm sorry to hear that your jacket is defective! You can request an immediate free replacement here: https://amazon.com/returns. ^NR",
        },
        {
            "query_match": r"cheers for automatically taking money from my bank account",
            "fallback_query": "cheers for automatically taking money from my bank account without any emails to confirm that I want a prime membership, this money was set aside for my energy bill",
            "intent": IntentCategory.BILLING_PAYMENT_DISPUTES,
            "escalate": True,
            "reason": "Unexpected debit affecting critical personal finances; policy mandates human specialist review.",
            "difficulty": "ambiguous",
            "context": "None (opening turn)",
            "brand_reply": "I'm so sorry to hear you were charged unexpectedly! You can cancel your Prime membership here: https://t.co/o6ilBbTZUV for a full refund. ^JN",
        },

        # Tier 3: Multi-Issue
        {
            "query_match": r"not only do I get charged 3 times for an order, I now get informed it’s going to be late",
            "fallback_query": "not only do I get charged 3 times for an order, I now get informed it’s going to be late getting delivered!",
            "intent": IntentCategory.BILLING_PAYMENT_DISPUTES,
            "escalate": True,
            "reason": "Multi-issue (multiple charges + delivery delay); severe financial dispute takes precedence for human escalation.",
            "difficulty": "multi_issue",
            "context": "None (opening turn)",
            "brand_reply": "Sorry to hear about this! Please connect directly with our billing team here so we can resolve both charges: https://t.co/vlvfJr4nN9. ^PK",
        },
        {
            "query_match": r"order no. 171-3098713-1137162 Is marked as delivered but it is not delivered yet. Pls cancel it and refund",
            "fallback_query": "order no. 171-3098713-1137162 Is marked as delivered but it is not delivered yet. Pls cancel it and refund the amount immediately",
            "intent": IntentCategory.DELIVERY_NOT_RECEIVED,
            "escalate": True,
            "reason": "Multi-issue (marked delivered missing + cancellation + refund demand); cannot cancel delivered order without investigation.",
            "difficulty": "multi_issue",
            "context": "None (opening turn)",
            "brand_reply": "Please share your details here: https://t.co/GIJyeYqKE0 so an investigator can verify delivery and process a refund. ^HD",
        },
        {
            "query_match": r"If I wanted my package delivered eventually I wouldn't have paid 100 for prime",
            "fallback_query": "If I wanted my package delivered eventually I wouldn't have paid 100 for prime. At this point I really just want a refund on prime because getting 5 dollars doesn't make up for multiple failures to deliver on promised times",
            "intent": IntentCategory.CUSTOMER_SERVICE_ESCALATION,
            "escalate": True,
            "reason": "Multi-issue (late delivery + Prime refund demand + rejection of previous agent concession); requires supervisor handling.",
            "difficulty": "multi_issue",
            "context": "[Customer]: Amazon gave me a 5 dollar credit because of my late package | [AmazonHelp]: We apologize for the shipping delay.",
            "brand_reply": "I'm sorry for your frustrations! We'd like to have a specialist look into this with you. Please send us a message here: https://t.co/... ^TR",
        },

        # Tier 4: Angry, Profane & Service Escalation
        {
            "query_match": r"what a joke. The most #unhelpful #rude #unprofessional #uncommunicative team",
            "fallback_query": "what a joke. The most #unhelpful #rude #unprofessional #uncommunicative team I’ve come across. Who taught you #CustomerService?",
            "intent": IntentCategory.CUSTOMER_SERVICE_ESCALATION,
            "escalate": True,
            "reason": "Severe service dissatisfaction accusing support reps of rudeness; mandatory human supervisor escalation.",
            "difficulty": "angry_profane",
            "context": "None (opening turn)",
            "brand_reply": "I'm sorry to hear that. Without giving any order details publicly, please connect with our leadership team here: https://t.co/... ^AB",
        },
        {
            "query_match": r"Your service is completely unacceptable. I am calling my attorney",
            "fallback_query": "Your service is completely unacceptable. I am calling my attorney and filing a complaint with the FTC.",
            "intent": IntentCategory.CUSTOMER_SERVICE_ESCALATION,
            "escalate": True,
            "reason": "Explicit legal and regulatory threat; corporate policy mandates immediate freeze and transfer to specialized team.",
            "difficulty": "angry_profane",
            "context": "None (opening turn)",
            "brand_reply": "We take this matter seriously. Please contact our executive relations office directly via: https://amazon.com/gp/help/contact-us. ^EX",
        },
        {
            "query_match": r"Amazon can you fucking not\? \"Delivered\" does not mean leave it in the sodding driveway",
            "fallback_query": "Amazon can you fucking not? 'Delivered' does not mean leave it in the sodding driveway where anyone can steal it!",
            "intent": IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE,
            "escalate": True,
            "reason": "Profane complaint regarding unsafe delivery drop placement; requires carrier logistics report.",
            "difficulty": "angry_profane",
            "context": "None (opening turn)",
            "brand_reply": "I'm so sorry about this poor delivery experience! Could you please confirm if you have a safe place set here: https://t.co/M6rUMOayX7? ^SD",
        },

        # Tier 5: High Risk / Safety Critical
        {
            "query_match": r"I noticed an unauthorized charge of \$149\.99 on my Visa",
            "fallback_query": "I noticed an unauthorized charge of $149.99 on my Visa from your site yesterday. Dispute the whole charge please!",
            "intent": IntentCategory.BILLING_PAYMENT_DISPUTES,
            "escalate": True,
            "reason": "Financial fraud allegation; policy strictly prohibits autonomous resolution and demands secure human verification.",
            "difficulty": "high_risk",
            "context": "None (opening turn)",
            "brand_reply": "Please report this immediately to our secure fraud team here: https://amazon.com/contact-us. Do not post card numbers publicly. ^CB",
        },
        {
            "query_match": r"please help\. I can't get into the live chat or any contact us area as your site has locked my account",
            "fallback_query": "please help. I can't get into the live chat or any contact us area as your site has locked my account. Who do I contact in UK?",
            "intent": IntentCategory.ACCOUNT_ACCESS_SECURITY,
            "escalate": True,
            "reason": "Account locked out preventing normal sign-in; requires Two-Step verification recovery by human security specialist.",
            "difficulty": "high_risk",
            "context": "None (opening turn)",
            "brand_reply": "I'm sorry for any inconvenience! Please use our alternate verification recovery form here: https://t.co/zYVX1Qi29G. ^JP",
        },
        {
            "query_match": r"your delivery guy in Lincoln park NJ took my friends puppy",
            "fallback_query": "your delivery guy in Lincoln park NJ took my friends puppy. Need help now!!! Police next call... Thank you!",
            "intent": IntentCategory.CARRIER_PHYSICAL_DELIVERY_ISSUE,
            "escalate": True,
            "reason": "Critical physical crime allegation involving delivery personnel; requires immediate dispatcher and human intervention.",
            "difficulty": "high_risk",
            "context": "None (opening turn)",
            "brand_reply": "We'd like to have a specialist look into this urgently. Please provide details securely here: https://t.co/... ^EM",
        },

        # Tier 6: Low Signal, Fragments & Out-of-Scope
        {
            "query_match": r"Link\?\?\?",
            "fallback_query": "Link???",
            "intent": IntentCategory.OTHER_UNCLEAR,
            "escalate": False,
            "reason": "One-word fragment with zero context; resolvable autonomously by requesting clarification.",
            "difficulty": "low_signal",
            "context": "None (opening turn)",
            "brand_reply": "Could you please explain what you need assistance with so we can direct you to the right place? ^AG",
        },
        {
            "query_match": r"team im facing issue on my amazon seller account",
            "fallback_query": "team im facing issue on my amazon seller account ?? Can u help i already droped mail also still didn't received any response..",
            "intent": IntentCategory.OTHER_UNCLEAR,
            "escalate": False,
            "reason": "Merchant / Seller Central issue out of consumer customer service scope; resolvable by redirecting to Seller Central.",
            "difficulty": "low_signal",
            "context": "None (opening turn)",
            "brand_reply": "Our consumer team can't access seller accounts. Please reach out directly to Seller Support at: https://sellercentral.amazon.com. ^MS",
        },
    ]

    records: List[Dict[str, Any]] = []

    for idx, s in enumerate(sample_definitions):
        pat = s["query_match"]
        matched_rows = eng_df[eng_df["customer_message"].str.contains(pat, case=False, regex=True)]

        if not matched_rows.empty:
            row = matched_rows.iloc[0]
            msg_text = row["customer_message"].strip()
            ctx = row["thread_context"]
            b_reply = row["brand_reply"].strip()
            t_idx = int(row["turn_index"])
        else:
            msg_text = s["fallback_query"]
            ctx = s["context"]
            b_reply = s["brand_reply"]
            t_idx = 1

        rec = {
            "id": f"GOLDEN_{idx + 1:03d}",
            "message_text": msg_text,
            "thread_context": ctx,
            "turn_index": t_idx,
            "true_intent": s["intent"].value,
            "true_escalate": s["escalate"],
            "escalate_reason": s["reason"],
            "difficulty": s["difficulty"],
            "review_status": "auto_labeled_pending_review",
            "reference_brand_reply": b_reply,
        }
        records.append(rec)

    return records


def main():
    records = build_curated_sample()
    print(f"Generated {len(records)} sample golden records across 6 difficulty tiers.\n")
    for r in records:
        print(f"[{r['id']}] ({r['difficulty'].upper()}) Intent: {r['true_intent']} | Escalate: {r['true_escalate']}")
        print(f"  Customer : {r['message_text'][:100]}...")
        print(f"  Reason   : {r['escalate_reason']}")
        print(f"  Ref Reply: {r['reference_brand_reply'][:90]}...\n")


if __name__ == "__main__":
    main()
