"""
Builds and persists the comprehensive Grounding Store corpus (thousands of documents)
from data/processed/amazon_support_pairs.parquet and official taxonomy guidance.
"""

from pathlib import Path
import pandas as pd
from intents.classifier import CalibratedHeuristicClassifier
from intents.taxonomy import SUPPORT_TAXONOMY

PARQUET_IN = Path("data/processed/amazon_support_pairs.parquet")
PARQUET_OUT = Path("data/processed/grounding_store.parquet")

def build_grounding_corpus(target_count: int = 2500) -> None:
    print(f"Loading raw pairs from {PARQUET_IN}...")
    df = pd.read_parquet(PARQUET_IN)
    
    # Filter high quality English pairs
    sub = df[(df["customer_message"].str.len() > 15) & (df["brand_reply"].str.len() > 20)].copy()
    print(f"Filtered pool size: {len(sub)} candidate pairs.")
    
    clf = CalibratedHeuristicClassifier()
    records = []
    
    # 1. Add official taxonomy policies (12 core documents)
    for cat, defn in SUPPORT_TAXONOMY.items():
        records.append({
            "doc_id": f"POLICY_{cat.value.upper()}",
            "title": defn.name,
            "category": cat.value,
            "content": f"POLICY GUIDANCE: {defn.resolution_guidance}\nSCOPE: {defn.description}",
            "source_url_or_id": "intents/taxonomy.py",
            "is_policy": True,
        })
        
    # 2. Add real historical brand precedent pairs across intents
    seen_replies = set()
    intent_counts = {}
    
    # Priority scan for rare intents first to ensure rich grounding
    rare_categories = {"billing_payment_disputes", "account_access_security", "order_cancellation_modification", "delivery_not_received", "product_inquiry_availability"}
    
    for row in sub.itertuples(index=False):
        b_reply = str(row.brand_reply).strip()
        sig = b_reply[:60].lower()
        if sig in seen_replies:
            continue
            
        c_msg = str(row.customer_message).strip()
        pred = clf.classify(c_msg)
        cat = pred.predicted_category.value
        
        limit = 250 if cat in rare_categories else 200
        if intent_counts.get(cat, 0) >= limit:
            continue
            
        seen_replies.add(sig)
        intent_counts[cat] = intent_counts.get(cat, 0) + 1
        
        records.append({
            "doc_id": f"PRECEDENT_{row.pair_id}",
            "title": c_msg,
            "category": cat,
            "content": b_reply,
            "source_url_or_id": f"conversation_{row.conversation_id}",
            "is_policy": False,
        })
        
        if len(records) >= target_count:
            break
            
    out_df = pd.DataFrame(records)
    out_df.to_parquet(PARQUET_OUT, index=False)
    print(f"Successfully saved {len(out_df)} grounding documents to {PARQUET_OUT}")
    print("Intent distribution in grounding corpus:")
    print(out_df["category"].value_counts())

if __name__ == "__main__":
    build_grounding_corpus(target_count=2400)
