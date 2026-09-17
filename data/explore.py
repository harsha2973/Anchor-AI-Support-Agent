"""
Exploratory data analysis (EDA) for Amazon Customer Support dialogue dataset.

Computes:
1. Message length distribution (customer inquiry vs. brand reply)
2. Volume over time (monthly, daily trends, date range)
3. Most common opening phrases & keywords in customer messages (n-grams)
4. Duplicate, spam, and canned-reply boilerplate detection
5. Formatted sample of 30-50 real customer messages for intent pattern discovery.
"""

from __future__ import annotations

import argparse
import collections
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import numpy as np
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Basic English stopwords for cleaner n-gram analysis
COMMON_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "amazon",
    "help", "hi", "hello", "please", "thanks", "thank"
}


def compute_length_stats(series: pd.Series) -> Dict[str, float]:
    """Compute comprehensive descriptive distribution stats for text lengths."""
    lengths = series.dropna().astype(str).str.len()
    words = series.dropna().astype(str).str.split().str.len()
    return {
        "char_mean": float(lengths.mean()),
        "char_median": float(lengths.median()),
        "char_std": float(lengths.std()),
        "char_min": int(lengths.min()),
        "char_p25": float(lengths.quantile(0.25)),
        "char_p75": float(lengths.quantile(0.75)),
        "char_p95": float(lengths.quantile(0.95)),
        "char_max": int(lengths.max()),
        "word_mean": float(words.mean()),
        "word_median": float(words.median()),
        "word_std": float(words.std()),
        "word_p25": float(words.quantile(0.25)),
        "word_p75": float(words.quantile(0.75)),
        "word_p95": float(words.quantile(0.95)),
        "word_max": int(words.max()),
    }


def analyze_ngrams(
    texts: pd.Series,
    n: int = 2,
    top_k: int = 20,
    opening_only: bool = False,
    remove_stopwords: bool = False,
) -> List[Tuple[str, int]]:
    """
    Extract top n-grams from text corpus.

    Args:
        texts: Series of string texts.
        n: n-gram size (1 for unigram, 2 for bigram, etc.).
        top_k: Number of most frequent n-grams to return.
        opening_only: If True, only considers the first 6 tokens of each message.
        remove_stopwords: If True, filters out tokens in COMMON_STOPWORDS.
    """
    ngram_counts: collections.Counter = collections.Counter()

    for text in texts.dropna():
        # Lowercase and extract alphanumeric words
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", str(text).lower())
        if opening_only:
            tokens = tokens[:6]

        if remove_stopwords:
            tokens = [t for t in tokens if t not in COMMON_STOPWORDS]

        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i:i + n])
                ngram_counts[gram] += 1

    return ngram_counts.most_common(top_k)


def analyze_boilerplates(df: pd.DataFrame, top_k: int = 15) -> Dict[str, Any]:
    """Detect repetitive, canned responses and duplicate queries."""
    brand_counts = df["brand_reply"].value_counts().head(top_k)
    customer_counts = df["customer_message"].value_counts().head(top_k)

    total_brand = len(df["brand_reply"])
    unique_brand = df["brand_reply"].nunique()
    total_cust = len(df["customer_message"])
    unique_cust = df["customer_message"].nunique()

    top_10_brand_share = (df["brand_reply"].value_counts().head(10).sum() / total_brand) * 100
    top_10_cust_share = (df["customer_message"].value_counts().head(10).sum() / total_cust) * 100

    return {
        "total_pairs": len(df),
        "unique_brand_replies": unique_brand,
        "brand_uniqueness_ratio": unique_brand / total_brand,
        "top_10_brand_volume_share_pct": top_10_brand_share,
        "top_brand_canned_replies": brand_counts.to_dict(),
        "unique_customer_messages": unique_cust,
        "customer_uniqueness_ratio": unique_cust / total_cust,
        "top_10_cust_volume_share_pct": top_10_cust_share,
        "top_duplicate_customer_messages": customer_counts.to_dict(),
    }


def sample_customer_messages(
    df: pd.DataFrame,
    num_samples: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw a diverse, stratified sample of customer queries across dialogue turns and lengths.
    """
    # Sample from opening queries (turn 1) and follow-ups
    openings = df[df["turn_index"] == 1]
    follow_ups = df[df["turn_index"] > 1]

    n_open = int(num_samples * 0.75)
    n_follow = num_samples - n_open

    sampled_open = openings.sample(min(n_open, len(openings)), random_state=seed)
    sampled_follow = follow_ups.sample(min(n_follow, len(follow_ups)), random_state=seed)

    sample = pd.concat([sampled_open, sampled_follow]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sample


def run_exploration(
    processed_parquet_path: Path = Path("data/processed/amazon_support_pairs.parquet"),
    report_output_path: Path = Path("reports/data_exploration.md"),
    num_samples_to_print: int = 40,
) -> None:
    """Run full exploratory analysis and format output."""
    if not processed_parquet_path.exists():
        raise FileNotFoundError(f"Processed file not found at {processed_parquet_path}")

    logger.info("Loading processed dataset from %s ...", processed_parquet_path)
    df = pd.read_parquet(processed_parquet_path)
    total_records = len(df)
    unique_conversations = df["conversation_id"].nunique()

    # --- 1. Message Length Stats ---
    cust_len_stats = compute_length_stats(df["customer_message"])
    brand_len_stats = compute_length_stats(df["brand_reply"])

    # --- 2. Temporal Analysis ---
    df["created_dt"] = pd.to_datetime(
        df["customer_created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )
    valid_dt = df["created_dt"].dropna()
    min_date = valid_dt.min().strftime("%Y-%m-%d") if not valid_dt.empty else "N/A"
    max_date = valid_dt.max().strftime("%Y-%m-%d") if not valid_dt.empty else "N/A"
    monthly_vol = valid_dt.dt.to_period("M").value_counts().sort_index()

    # --- 3. Turn Distribution ---
    turn_dist = df["turn_index"].value_counts().sort_index().head(5)

    # --- 4. Phrases & Keywords ---
    opening_bigrams = analyze_ngrams(df[df["turn_index"] == 1]["customer_message"], n=2, top_k=15, opening_only=True)
    opening_trigrams = analyze_ngrams(df[df["turn_index"] == 1]["customer_message"], n=3, top_k=15, opening_only=True)
    content_keywords = analyze_ngrams(df["customer_message"], n=1, top_k=20, remove_stopwords=True)
    content_bigrams = analyze_ngrams(df["customer_message"], n=2, top_k=15, remove_stopwords=True)

    # --- 5. Boilerplate & Duplicates ---
    boilerplates = analyze_boilerplates(df, top_k=5)

    # --- 6. Diverse Sample ---
    samples = sample_customer_messages(df, num_samples=num_samples_to_print, seed=123)

    # Format Console Output & Markdown Report
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS: AMAZONHELP CUSTOMER SUPPORT DATASET")
    print("=" * 80)
    print(f"Total Turn Pairs           : {total_records:,}")
    print(f"Unique Root Conversations  : {unique_conversations:,}")
    print(f"Dataset Date Range         : {min_date} to {max_date}")
    print(f"Opening Queries (Turn 1)   : {sum(df['turn_index'] == 1):,} ({(sum(df['turn_index'] == 1)/total_records)*100:.1f}%)")
    print(f"Follow-up Turns (>1)       : {sum(df['turn_index'] > 1):,} ({(sum(df['turn_index'] > 1)/total_records)*100:.1f}%)")

    print("\n" + "-" * 80)
    print("1. MESSAGE LENGTH DISTRIBUTION (CHARACTERS & WORDS)")
    print("-" * 80)
    len_table = [
        ["Metric", "Customer Inquiry", "Brand Reply (AmazonHelp)"],
        ["Char Count: Mean +/- Std", f"{cust_len_stats['char_mean']:.1f} +/- {cust_len_stats['char_std']:.1f}", f"{brand_len_stats['char_mean']:.1f} +/- {brand_len_stats['char_std']:.1f}"],
        ["Char Count: Median [p25, p75]", f"{cust_len_stats['char_median']:.0f} [{cust_len_stats['char_p25']:.0f}, {cust_len_stats['char_p75']:.0f}]", f"{brand_len_stats['char_median']:.0f} [{brand_len_stats['char_p25']:.0f}, {brand_len_stats['char_p75']:.0f}]"],
        ["Char Count: 95th Percentile", f"{cust_len_stats['char_p95']:.0f}", f"{brand_len_stats['char_p95']:.0f}"],
        ["Word Count: Mean +/- Std", f"{cust_len_stats['word_mean']:.1f} +/- {cust_len_stats['word_std']:.1f}", f"{brand_len_stats['word_mean']:.1f} +/- {brand_len_stats['word_std']:.1f}"],
        ["Word Count: Median [p25, p75]", f"{cust_len_stats['word_median']:.0f} [{cust_len_stats['word_p25']:.0f}, {cust_len_stats['word_p75']:.0f}]", f"{brand_len_stats['word_median']:.0f} [{brand_len_stats['word_p25']:.0f}, {brand_len_stats['word_p75']:.0f}]"],
    ]
    print(tabulate(len_table, headers="firstrow", tablefmt="github"))

    print("\n" + "-" * 80)
    print("2. VOLUME OVER TIME (MONTHLY RECORD COUNTS)")
    print("-" * 80)
    for period, count in monthly_vol.items():
        bar = "#" * int(count / 3000)
        print(f"  {str(period):<10} : {count:>7,}  {bar}")

    print("\n" + "-" * 80)
    print("3. MOST FREQUENT OPENING PHRASES IN CUSTOMER QUERIES (Turn 1)")
    print("-" * 80)
    print("Top Opening Trigrams:")
    for gram, cnt in opening_trigrams[:10]:
        print(f"  - \"{gram}\" ({cnt:,} times)")

    print("\nTop Content Keywords (Stopwords Redacted):")
    keywords_str = ", ".join([f"{k} ({c:,})" for k, c in content_keywords[:15]])
    print(f"  {keywords_str}")

    print("\nTop Content Bigrams (Stopwords Redacted):")
    for gram, cnt in content_bigrams[:10]:
        print(f"  - \"{gram}\" ({cnt:,} times)")

    print("\n" + "-" * 80)
    print("4. BOILERPLATE & DUPLICATION ANALYSIS")
    print("-" * 80)
    print(f"Unique Brand Replies       : {boilerplates['unique_brand_replies']:,} / {boilerplates['total_pairs']:,} ({boilerplates['brand_uniqueness_ratio']*100:.2f}% unique)")
    print(f"Top 10 Canned Brand Replies: Account for {boilerplates['top_10_brand_volume_share_pct']:.2f}% of ALL Amazon replies!")
    print(f"Customer Query Uniqueness  : {boilerplates['unique_customer_messages']:,} / {boilerplates['total_pairs']:,} ({boilerplates['customer_uniqueness_ratio']*100:.2f}% unique)")
    print("\nMost Common AmazonHelp Boilerplate Template:")
    top_canned_text, top_canned_cnt = list(boilerplates["top_brand_canned_replies"].items())[0]
    print(f"  [{top_canned_cnt:,} times ({top_canned_cnt/total_records*100:.1f}%)] \"{top_canned_text[:120]}...\"")

    print("\n" + "=" * 80)
    print(f"5. SUMMARY OF {num_samples_to_print} REAL SAMPLED CUSTOMER MESSAGES (FOR INTENT EYEBALLING)")
    print("=" * 80)

    for idx, row in samples.iterrows():
        turn_label = f"Turn {row['turn_index']}"
        context_str = f" | Context: {row['thread_context'][:60]}..." if row['turn_index'] > 1 else ""
        print(f"\n[{idx + 1:02d}] ({turn_label}{context_str})")
        print(f"     CUSTOMER : {row['customer_message']}")
        print(f"     AMAZON   : {row['brand_reply'][:110]}...")

    # Write full markdown report
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis: AmazonHelp Customer Support Corpus\n\n")
        f.write(f"- **Total Interaction Pairs**: {total_records:,}\n")
        f.write(f"- **Unique Conversation Threads**: {unique_conversations:,}\n")
        f.write(f"- **Date Range**: {min_date} to {max_date}\n\n")
        f.write("## 1. Message Length Distributions\n\n")
        f.write(tabulate(len_table, headers="firstrow", tablefmt="github") + "\n\n")
        f.write("## 2. Monthly Volume Distribution\n\n")
        f.write("| Month | Interaction Count |\n| :--- | :--- |\n")
        for period, count in monthly_vol.items():
            f.write(f"| {str(period)} | {count:,} |\n")
        f.write("\n## 3. Frequent Opening Phrases & Top Keywords\n\n")
        f.write("### Top Opening Trigrams\n")
        for gram, cnt in opening_trigrams:
            f.write(f"- `{gram}` ({cnt:,})\n")
        f.write("\n### Top Content Bigrams (Domain Specific)\n")
        for gram, cnt in content_bigrams:
            f.write(f"- `{gram}` ({cnt:,})\n")
        f.write("\n## 4. Boilerplate & Automation Insights\n\n")
        f.write(f"- **Brand Uniqueness Ratio**: {boilerplates['brand_uniqueness_ratio']*100:.2f}%\n")
        f.write(f"- **Top 10 Brand Macros Share**: {boilerplates['top_10_brand_volume_share_pct']:.2f}% of total volume.\n")
        f.write(f"- **Customer Message Uniqueness**: {boilerplates['customer_uniqueness_ratio']*100:.2f}%\n\n")
        f.write("## 5. Sampled Real Customer Inquiries\n\n")
        for idx, row in samples.iterrows():
            f.write(f"**Sample {idx + 1:02d}** (Turn {row['turn_index']})\n")
            f.write(f"- **Customer**: {row['customer_message']}\n")
            f.write(f"- **AmazonHelp Reply**: {row['brand_reply']}\n")
            if row['turn_index'] > 1:
                f.write(f"- **Prior Context**: `{row['thread_context']}`\n")
            f.write("\n")

    logger.info("Exploratory analysis report saved to %s", report_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory analysis on processed Amazon dialogue dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/amazon_support_pairs.parquet"),
        help="Path to processed parquet dataset.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/data_exploration.md"),
        help="Path to output markdown report.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Number of real customer messages to print.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_exploration(
        processed_parquet_path=args.data_path,
        report_output_path=args.report_path,
        num_samples_to_print=args.samples,
    )


if __name__ == "__main__":
    main()
