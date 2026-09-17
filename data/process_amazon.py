"""
Pipeline for filtering AmazonHelp tweets, reconstructing conversation threads,
and generating paired customer<->brand dialogue records with conversational context.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from data.loader import DataLoader

logger = logging.getLogger(__name__)


def clean_tweet_text(text: Optional[str]) -> str:
    """
    Remove Twitter handle mentions (@AmazonHelp, @123456) and normalize whitespace.

    Preserves URLs, punctuation, and actual query semantics while removing
    superficial Twitter routing artifacts.
    """
    if not text or pd.isna(text):
        return ""
    # Strip leading/trailing user handles
    cleaned = re.sub(r"@[\w_]+", "", str(text))
    # Replace multiple whitespace/newlines with single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def filter_amazon_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters raw Twitter Customer Support DataFrame to interactions involving AmazonHelp.

    Identifies:
    1. Direct tweets by AmazonHelp (author_id == 'AmazonHelp')
    2. Customer tweets that AmazonHelp responded to (in_response_to_tweet_id from Amazon tweets)
    3. Customer follow-up tweets replying to AmazonHelp tweets

    Args:
        df: Raw twcs DataFrame.

    Returns:
        Filtered DataFrame containing only Amazon-related conversational turns.
    """
    logger.info("Filtering dataset for AmazonHelp interactions...")

    # Step 1: All tweets by AmazonHelp
    amazon_mask = df["author_id"] == "AmazonHelp"
    amazon_df = df[amazon_mask]
    amazon_tweet_ids: Set[int] = set(amazon_df["tweet_id"].dropna().astype(int))
    logger.info("Found %d tweets authored by AmazonHelp", len(amazon_tweet_ids))

    # Step 2: Inbound customer tweets that AmazonHelp replied to
    inbound_parent_ids: Set[int] = set(
        amazon_df["in_response_to_tweet_id"].dropna().astype(int)
    )
    logger.info("Identified %d unique parent tweet IDs replied to by AmazonHelp", len(inbound_parent_ids))

    # Step 3: Tweets that are in response to an AmazonHelp tweet (follow-ups)
    # in_response_to_tweet_id is float in pandas due to NaNs
    resp_to_amazon_mask = df["in_response_to_tweet_id"].isin(amazon_tweet_ids)

    # Step 4: Combined relevant tweet ID set
    relevant_tweet_ids = amazon_tweet_ids.union(inbound_parent_ids)
    final_mask = df["tweet_id"].isin(relevant_tweet_ids) | resp_to_amazon_mask

    filtered_df = df[final_mask].copy()
    logger.info("Retained %d total Amazon-related tweets across all turns", len(filtered_df))
    return filtered_df


def reconstruct_dialogue_pairs(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct customer<->brand turn pairs with full preceding conversational context.

    For every AmazonHelp reply:
    - Finds the antecedent customer message.
    - Traces backward along parent pointers to assemble prior dialogue history (`thread_context`).
    - Assigns turn order index within the conversation.

    Args:
        filtered_df: Subset of tweets involving AmazonHelp.

    Returns:
        DataFrame of paired interactions:
        (pair_id, conversation_id, turn_index, customer_tweet_id, brand_tweet_id,
         customer_message, brand_reply, thread_context, customer_created_at, brand_created_at)
    """
    logger.info("Reconstructing dialogue pairs and conversational threads...")

    # Build fast dictionary lookup keyed by tweet_id
    tweet_dict: Dict[int, Dict[str, Any]] = {}
    for row in filtered_df.itertuples(index=False):
        tweet_dict[int(row.tweet_id)] = {
            "tweet_id": int(row.tweet_id),
            "author_id": str(row.author_id),
            "inbound": bool(row.inbound),
            "created_at": str(row.created_at),
            "text": str(row.text),
            "clean_text": clean_tweet_text(row.text),
            "in_response_to": int(row.in_response_to_tweet_id) if pd.notna(row.in_response_to_tweet_id) else None,
        }

    pairs: List[Dict[str, Any]] = []

    # Iterate through all AmazonHelp tweets
    for t_id, tweet in tweet_dict.items():
        if tweet["author_id"] != "AmazonHelp" or tweet["in_response_to"] is None:
            continue

        parent_id = tweet["in_response_to"]
        if parent_id not in tweet_dict:
            # Parent tweet was not captured in the crawl
            continue

        parent_tweet = tweet_dict[parent_id]
        if not parent_tweet["inbound"]:
            # Parent was not a customer message (e.g. brand internal routing or self-reply)
            continue

        # Trace back chain to find thread history and canonical root conversation_id
        history_turns: List[Tuple[str, str]] = []  # List of (speaker, text)
        curr_id = parent_id
        visited: Set[int] = set()

        while curr_id and curr_id in tweet_dict and curr_id not in visited:
            visited.add(curr_id)
            node = tweet_dict[curr_id]
            speaker = "Customer" if node["inbound"] else "AmazonHelp"
            history_turns.append((speaker, node["clean_text"]))
            curr_id = node["in_response_to"]

        # Root tweet is the oldest ancestor in the chain
        root_tweet_id = list(visited)[-1]
        # Reverse history so it is chronological (from root forward to parent)
        history_turns.reverse()

        # Build thread_context from all turns BEFORE the current customer inquiry
        prior_turns = history_turns[:-1]
        if not prior_turns:
            thread_context = "None (opening turn)"
            turn_index = 1
        else:
            context_parts = []
            for i, (speaker, text) in enumerate(prior_turns):
                context_parts.append(f"[{speaker}]: {text}")
            thread_context = " | ".join(context_parts)
            # Count customer turns to establish turn index
            customer_turns_count = sum(1 for spk, _ in prior_turns if spk == "Customer") + 1
            turn_index = customer_turns_count

        pairs.append({
            "pair_id": f"amz_{parent_id}_{t_id}",
            "conversation_id": str(root_tweet_id),
            "turn_index": turn_index,
            "customer_tweet_id": str(parent_id),
            "brand_tweet_id": str(t_id),
            "customer_message": parent_tweet["clean_text"],
            "raw_customer_message": parent_tweet["text"],
            "brand_reply": tweet["clean_text"],
            "raw_brand_reply": tweet["text"],
            "thread_context": thread_context,
            "customer_created_at": parent_tweet["created_at"],
            "brand_created_at": tweet["created_at"],
        })

    logger.info("Successfully reconstructed %d customer<->brand dialogue pairs", len(pairs))
    pairs_df = pd.DataFrame(pairs)

    # Sort chronologically by customer_created_at if available
    try:
        pairs_df["customer_datetime"] = pd.to_datetime(
            pairs_df["customer_created_at"],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce",
        )
        pairs_df = pairs_df.sort_values(by=["conversation_id", "turn_index"]).reset_index(drop=True)
        pairs_df = pairs_df.drop(columns=["customer_datetime"])
    except Exception as e:
        logger.warning("Could not sort by datetime: %s", e)

    return pairs_df


def process_and_save(
    raw_csv_path: Optional[Path] = None,
    output_dir: Path = Path("data/processed"),
    max_pairs: Optional[int] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Execute end-to-end filtering and pair reconstruction, exporting Parquet and CSV files.

    Args:
        raw_csv_path: Optional path to twcs.csv. If None, auto-detected.
        output_dir: Destination folder for processed datasets.
        max_pairs: Optional limit on output pairs for testing.

    Returns:
        Tuple of (pairs_df, parquet_path, csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader()
    raw_df = loader.load_raw_dataframe(file_path=raw_csv_path)

    filtered_df = filter_amazon_tweets(raw_df)
    pairs_df = reconstruct_dialogue_pairs(filtered_df)

    if max_pairs:
        pairs_df = pairs_df.head(max_pairs)

    # Filter out empty or whitespace-only messages
    pairs_df = pairs_df[
        (pairs_df["customer_message"].str.len() > 3) &
        (pairs_df["brand_reply"].str.len() > 3)
    ].reset_index(drop=True)

    parquet_path = output_dir / "amazon_support_pairs.parquet"
    csv_path = output_dir / "amazon_support_pairs.csv"
    sample_csv_path = output_dir / "amazon_support_sample_50.csv"

    logger.info("Saving processed dataset to Parquet: %s ...", parquet_path)
    pairs_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    logger.info("Saving sample CSV (50 rows) for easy inspection: %s ...", sample_csv_path)
    pairs_df.head(50).to_csv(sample_csv_path, index=False, encoding="utf-8")

    logger.info("Saving complete CSV: %s ...", csv_path)
    pairs_df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(
        "Processing complete! Processed %d pairs (saved to %s and %s)",
        len(pairs_df),
        parquet_path,
        csv_path,
    )
    return pairs_df, parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and reconstruct AmazonHelp dialogue pairs.")
    parser.add_argument("--raw-file", type=Path, default=None, help="Path to raw twcs.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    process_and_save(raw_csv_path=args.raw_file, output_dir=args.output_dir, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
