"""
Script to extract authentic candidate examples and historical replies for taxonomy definition.
"""

import sys
import re
from pathlib import Path
import pandas as pd

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    df = pd.read_parquet("data/processed/amazon_support_pairs.parquet")
    
    # Filter to mostly English queries
    is_eng = df["customer_message"].apply(
        lambda x: sum(ord(c) < 128 for c in str(x)) / max(len(str(x)), 1) > 0.90
        and len(str(x)) > 25
    )
    eng_df = df[is_eng].copy()

    categories = {
        "ORDER_TRACKING_DELAY": r"\b(where is my|tracking|track|delay|delayed|late|hasn't arrived|not arrived|when will.*arrive|eta)\b",
        "DELIVERED_NOT_RECEIVED": r"\b(says delivered|marked delivered|said delivered|showing delivered|delivered.*(not|never|didn|haven)|haven't received.*delivered)\b",
        "RETURNS_REFUNDS": r"\b(refund|return|return label|money back|cashback|reimburse|exchange|send.*back)\b",
        "DAMAGED_DEFECTIVE_WRONG": r"\b(damaged|broken|crushed|shattered|wrong item|defective|not working|different item|wrong size|faulty)\b",
        "BILLING_PAYMENT_DISPUTES": r"\b(charged twice|double charge|unauthorized charge|overcharged|payment failed|deducted.*twice|bank|dispute)\b",
        "PRIME_SUBSCRIPTION_BENEFITS": r"\b(prime video|prime membership|annual subscription|prime member|music unlimited|prime delivery|cancel prime)\b",
        "ACCOUNT_ACCESS_SECURITY": r"\b(locked out|lock.*account|password|hacked|can't log in|login|otp|2fa|verification code)\b",
        "PHYSICAL_DELIVERY_CARRIER_INCIDENT": r"\b(delivery guy|driver|usps|courier|carrier|delivery boy|threw|doorstep|porch|stole)\b",
        "CANCELLATION_MODIFICATION": r"\b(cancel my order|cancel order|cancel it|change address|change shipping address|cancel this)\b",
        "CUSTOMER_SERVICE_ESCALATION": r"\b(worst customer service|pathetic service|speak to supervisor|talk to manager|terrible service|complaint|unhelpful|call me)\b",
        "PRODUCT_INQUIRY_AVAILABILITY": r"\b(in stock|out of stock|compatible with|restock|available in|when will.*be available|dimensions)\b",
    }

    for cat_name, pattern in categories.items():
        matched = eng_df[
            eng_df["customer_message"].str.contains(pattern, case=False, regex=True)
            & (eng_df["turn_index"] == 1)
        ]
        sample = matched.sample(min(4, len(matched)), random_state=42)
        print(f"\n### {cat_name} (Total matches: {len(matched):,})")
        for i, r in enumerate(sample.itertuples()):
            c_msg = r.customer_message.replace("\n", " ").strip()
            b_msg = r.brand_reply.replace("\n", " ").strip()
            print(f"  Example {i+1}:")
            print(f"    Customer : \"{c_msg}\"")
            print(f"    AmazonHelp: \"{b_msg}\"")

if __name__ == "__main__":
    main()
