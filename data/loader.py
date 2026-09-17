"""
Dataset ingestion and schema definitions for Customer Support datasets.

Handles downloading from Kaggle API (reading credentials from .env), locating
raw files on disk, parsing CSV records into typed SupportTurn and ConversationThread
structures, and exporting processed splits.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupportTurn:
    """
    Represents a single conversational turn between a customer and a brand support agent.

    Attributes:
        turn_id: Unique identifier for the interaction turn (tweet_id).
        conversation_id: Thread or ticket ID grouping sequential turns.
        author_id: Anonymized user ID or brand identifier (e.g. 'AmazonHelp', '115712').
        is_brand_reply: True if this turn is an authorized company response, False if customer query.
        text: Raw or normalized textual content of the message.
        created_at: String or ISO timestamp.
        in_reply_to_turn_id: Pointer to the immediate predecessor turn ID in the thread.
        metadata: Extensible key-value metadata (brand name, channel, verified status).
    """

    turn_id: str
    conversation_id: str
    author_id: str
    is_brand_reply: bool
    text: str
    created_at: Optional[str] = None
    in_reply_to_turn_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialoguePair:
    """
    A reconstructed dialogue exchange between a customer inquiry and a brand reply.

    Attributes:
        pair_id: Unique identifier for this pair (e.g. "{customer_id}_{brand_id}").
        conversation_id: Root thread identifier.
        turn_index: Turn order within the conversation thread (1, 2, ...).
        customer_tweet_id: Tweet ID of the customer message.
        brand_tweet_id: Tweet ID of the brand reply.
        customer_message: Raw or cleaned customer inquiry text.
        brand_reply: Brand support reply text.
        thread_context: Prior conversational history formatted as context string.
        customer_created_at: Timestamp of customer message.
        brand_created_at: Timestamp of brand reply.
    """

    pair_id: str
    conversation_id: str
    turn_index: int
    customer_tweet_id: str
    brand_tweet_id: str
    customer_message: str
    brand_reply: str
    thread_context: str
    customer_created_at: Optional[str] = None
    brand_created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert DialoguePair to flat dictionary representation."""
        return {
            "pair_id": self.pair_id,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "customer_tweet_id": self.customer_tweet_id,
            "brand_tweet_id": self.brand_tweet_id,
            "customer_message": self.customer_message,
            "brand_reply": self.brand_reply,
            "thread_context": self.thread_context,
            "customer_created_at": self.customer_created_at,
            "brand_created_at": self.brand_created_at,
        }


@dataclass(frozen=True)
class ConversationThread:
    """
    An ordered sequence of turns comprising a complete support interaction.

    Attributes:
        conversation_id: Root thread identifier.
        turns: Chronologically ordered list of SupportTurn instances.
    """

    conversation_id: str
    turns: List[SupportTurn]

    @property
    def initial_customer_query(self) -> Optional[SupportTurn]:
        """Returns the opening customer message that initiated the support request."""
        for turn in self.turns:
            if not turn.is_brand_reply:
                return turn
        return None

    @property
    def brand_replies(self) -> List[SupportTurn]:
        """Returns all verified brand responses within this thread."""
        return [t for t in self.turns if t.is_brand_reply]


class DataLoader:
    """
    Ingests, validates, and manages customer support datasets.
    """

    DEFAULT_DATASET = "thoughtvector/customer-support-on-twitter"

    def __init__(
        self,
        raw_data_dir: Path = Path("data/raw"),
        processed_data_dir: Path = Path("data/processed"),
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def download_from_kaggle(
        self,
        dataset_slug: str = DEFAULT_DATASET,
        force_download: bool = False,
    ) -> Path:
        """
        Download dataset via the Kaggle API reading credentials from environment / .env.

        Args:
            dataset_slug: Kaggle dataset identifier (owner/dataset-name).
            force_download: Whether to re-download if file already exists locally.

        Returns:
            Path to downloaded raw dataset CSV file.
        """
        existing_file = self.find_raw_csv()
        if existing_file and not force_download:
            logger.info("Found existing raw dataset file at: %s", existing_file)
            return existing_file

        # Load environment variables from .env
        load_dotenv()
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")

        if username and username != "your-kaggle-username":
            os.environ["KAGGLE_USERNAME"] = username
        if key and key != "your-kaggle-key":
            os.environ["KAGGLE_KEY"] = key

        logger.info("Connecting to Kaggle API to download %s ...", dataset_slug)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(dataset_slug, path=str(self.raw_data_dir), unzip=True)
            logger.info("Successfully downloaded and extracted dataset to %s", self.raw_data_dir)
        except Exception as e:
            logger.error("Kaggle download failed: %s", e)
            raise

        found = self.find_raw_csv()
        if not found:
            raise FileNotFoundError(f"Download succeeded but no CSV found in {self.raw_data_dir}")
        return found

    def find_raw_csv(self) -> Optional[Path]:
        """Locate twcs.csv or primary raw CSV file in raw_data_dir."""
        for path in self.raw_data_dir.glob("**/*.csv"):
            if "twcs" in path.name.lower() or path.stat().st_size > 10_000_000:
                return path
        # Fallback to any CSV in raw_data_dir
        csvs = [p for p in self.raw_data_dir.glob("*.csv") if not p.name.startswith(".")]
        return csvs[0] if csvs else None

    def load_raw_dataframe(
        self,
        file_path: Optional[Path] = None,
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read the raw CSV into a pandas DataFrame with optimized dtypes.

        Args:
            file_path: Path to CSV file. If None, auto-locates in raw_data_dir.
            nrows: Optional row limit for fast debugging or chunking.
            usecols: Optional subset of columns to load.

        Returns:
            pd.DataFrame with customer support tweets.
        """
        path = file_path or self.find_raw_csv()
        if not path or not path.exists():
            raise FileNotFoundError(
                f"Raw dataset CSV not found. Please run with --download first. (Looked in {self.raw_data_dir})"
            )

        logger.info("Loading raw dataset from %s (nrows=%s) ...", path, nrows)
        dtype_spec = {
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
            "text": "string",
            "response_tweet_id": "string",
        }

        df = pd.read_csv(
            path,
            nrows=nrows,
            usecols=usecols,
            dtype=dtype_spec,
            low_memory=False,
        )
        logger.info("Loaded DataFrame: %d rows, %d columns", len(df), len(df.columns))
        return df


def main() -> None:
    """CLI utility for downloading and inspecting the raw dataset."""
    parser = argparse.ArgumentParser(description="Dataset loader & ingestion utility.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw Twitter Customer Support dataset from Kaggle.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect loaded raw dataset columns and author distribution.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows to load for quick inspection.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    loader = DataLoader()

    if args.download:
        csv_path = loader.download_from_kaggle()
        print(f"\n[OK] Raw dataset ready at: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    elif args.inspect:
        df = loader.load_raw_dataframe(nrows=args.nrows)
        print("\n" + "=" * 60)
        print("DATASET INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Total Rows Loaded : {len(df):,}")
        print(f"Columns           : {df.columns.tolist()}")
        print("\nTop 10 Authors / Handles by Tweet Count:")
        print(df["author_id"].value_counts().head(10))
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
