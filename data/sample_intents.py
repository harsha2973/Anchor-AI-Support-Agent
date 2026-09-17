"""
Samples real customer messages from the processed Amazon dataset for intent discovery.
"""

import re
import sys
from pathlib import Path
import pandas as pd

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    parquet_path = Path("data/processed/amazon_support_pairs.parquet")
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)

    # Filter to English conversations for clean intent eyeballing
    is_eng = df["customer_message"].apply(
        lambda x: bool(re.search(r"\b(my|the|is|to|order|delivery|package|refund|item|account|delayed|prime|return|charge|cancel|received|card|tracking)\b", str(x), re.I))
        and sum(ord(c) < 128 for c in str(x)) / max(len(str(x)), 1) > 0.85
    )
    eng_df = df[is_eng].copy()

    # Stratified sample: 30 opening queries (Turn 1) + 10 follow-ups (Turn > 1)
    openings = eng_df[eng_df["turn_index"] == 1].sample(30, random_state=42)
    followups = eng_df[eng_df["turn_index"] > 1].sample(10, random_state=42)
    sample = pd.concat([openings, followups]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Total English dialogue pairs: {len(eng_df):,}")
    print("=" * 80)
    print("SAMPLE OF 40 REAL CUSTOMER INQUIRIES (FOR INTENT PATTERN EYEBALLING)")
    print("=" * 80)

    for i, row in sample.iterrows():
        turn_info = f"Turn {row['turn_index']}"
        ctx = f" | Prior Turn: \"{row['thread_context'][:60]}...\"" if row['turn_index'] > 1 else ""
        print(f"[{i + 1:02d}] ({turn_info}{ctx})")
        print(f"     CUSTOMER: {row['customer_message']}")
        print(f"     AMAZON  : {row['brand_reply'][:100]}...\n")

if __name__ == "__main__":
    main()
