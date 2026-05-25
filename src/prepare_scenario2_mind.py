"""
Prepare MIND data for scenario-2 news recommendation baselines.

Expected raw layout:
  data/raw/mind/MINDsmall_train/news.tsv
  data/raw/mind/MINDsmall_train/behaviors.tsv
  data/raw/mind/MINDsmall_dev/news.tsv
  data/raw/mind/MINDsmall_dev/behaviors.tsv

Recommended smoke command:
  python src/prepare_scenario2_mind.py \
    --raw_dir data/raw/mind --size small --data_mode smoke \
    --max_train_impressions 5000 --max_eval_impressions 500
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIND files for scenario-2 news baselines.")
    parser.add_argument("--raw_dir", default="data/raw/mind", help="Directory containing MIND raw split folders.")
    parser.add_argument("--output_dir", default="data/processed/mind", help="Directory for processed parquet files.")
    parser.add_argument("--size", choices=["small", "large"], default="small", help="MIND size suffix.")
    parser.add_argument(
        "--data_mode",
        choices=["smoke", "large_sample", "full"],
        default="smoke",
        help="Processed output suffix.",
    )
    parser.add_argument(
        "--max_train_impressions",
        type=int,
        default=5000,
        help="Maximum train behavior rows to read. Use 0 for all rows.",
    )
    parser.add_argument(
        "--max_eval_impressions",
        type=int,
        default=500,
        help="Maximum dev behavior rows to read. Use 0 for all rows.",
    )
    parser.add_argument(
        "--behavior_chunk_size",
        type=int,
        default=50_000,
        help="Behavior rows per expansion chunk when writing pair parquet files.",
    )
    return parser.parse_args()


def _mind_prefix(size: str) -> str:
    return "MINDsmall" if size == "small" else "MINDlarge"


def discover_split_dir(raw_dir: str, size: str, split: str) -> Path:
    base = Path(raw_dir)
    prefix = _mind_prefix(size)
    candidates = [
        base / f"{prefix}_{split}",
        base / f"{prefix.lower()}_{split}",
        base / split,
        base,
    ]
    for candidate in candidates:
        if (candidate / "news.tsv").exists() and (candidate / "behaviors.tsv").exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find MIND {split} files under {base}. Expected e.g. {base / f'{prefix}_{split}' / 'news.tsv'}"
    )


def _nrows(limit: int) -> Optional[int]:
    return None if limit is None or int(limit) <= 0 else int(limit)


def read_news(split_dir: Path) -> pd.DataFrame:
    news = pd.read_csv(
        split_dir / "news.tsv",
        sep="\t",
        names=NEWS_COLUMNS,
        header=None,
        dtype=str,
        keep_default_na=False,
        quoting=3,
    )
    news["news_id"] = news["news_id"].astype(str)
    return news


def read_behaviors(split_dir: Path, limit: int) -> pd.DataFrame:
    behaviors = pd.read_csv(
        split_dir / "behaviors.tsv",
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        header=None,
        dtype=str,
        keep_default_na=False,
        nrows=_nrows(limit),
        quoting=3,
    )
    behaviors["impression_id"] = behaviors["impression_id"].astype(str)
    behaviors["user_id"] = behaviors["user_id"].astype(str)
    behaviors["request_time"] = pd.to_datetime(behaviors["time"], errors="coerce")
    return behaviors


def iter_behavior_chunks(split_dir: Path, limit: int, chunk_size: int):
    remaining = _nrows(limit)
    reader = pd.read_csv(
        split_dir / "behaviors.tsv",
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        header=None,
        dtype=str,
        keep_default_na=False,
        chunksize=max(1, int(chunk_size)),
        quoting=3,
    )
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        chunk["impression_id"] = chunk["impression_id"].astype(str)
        chunk["user_id"] = chunk["user_id"].astype(str)
        chunk["request_time"] = pd.to_datetime(chunk["time"], errors="coerce")
        yield chunk


def _split_news_ids(text: Any) -> List[str]:
    return [token for token in str(text or "").split() if token]


def _parse_impression_token(token: str) -> Tuple[str, Optional[int]]:
    if "-" not in token:
        return token, None
    news_id, label = token.rsplit("-", 1)
    try:
        return news_id, int(label)
    except ValueError:
        return news_id, None


def build_news_table(train_news: pd.DataFrame, dev_news: pd.DataFrame) -> pd.DataFrame:
    news = pd.concat([train_news, dev_news], ignore_index=True).drop_duplicates("news_id", keep="first").copy()
    for col in ["category", "subcategory", "title", "abstract"]:
        news[col] = news[col].fillna("").astype(str)
    news["category"] = news["category"].str.lower().replace("", "unknown")
    news["subcategory"] = news["subcategory"].str.lower().replace("", "unknown")
    news["text"] = (news["title"].fillna("") + " " + news["abstract"].fillna("")).str.strip()
    return news.reset_index(drop=True)


def behaviors_to_pairs(behaviors: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    iterator = tqdm(behaviors.itertuples(index=False), total=len(behaviors), desc=f"Expand {split} impressions")
    for behavior in iterator:
        impression_id = str(getattr(behavior, "impression_id"))
        user_id = str(getattr(behavior, "user_id"))
        request_time = getattr(behavior, "request_time")
        history = str(getattr(behavior, "history", ""))
        for position, token in enumerate(_split_news_ids(getattr(behavior, "impressions", "")), start=1):
            news_id, label = _parse_impression_token(token)
            rows.append(
                {
                    "split": split,
                    "impression_id": impression_id,
                    "user_id": user_id,
                    "request_time": request_time,
                    "history": history,
                    "news_id": str(news_id),
                    "label": -1 if label is None else int(label),
                    "candidate_position": int(position),
                }
            )
    return pd.DataFrame(rows)


def write_pairs_parquet(split_dir: Path, split: str, path: Path, limit: int, chunk_size: int) -> int:
    writer = None
    total_rows = 0
    try:
        for behaviors in iter_behavior_chunks(split_dir, limit, chunk_size):
            pairs = behaviors_to_pairs(behaviors, split)
            if pairs.empty:
                continue
            table = pa.Table.from_pandas(pairs, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += int(len(pairs))
            LOGGER.info("Wrote %s %s pairs so far to %s", total_rows, split, path)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pd.DataFrame(
            columns=["split", "impression_id", "user_id", "request_time", "history", "news_id", "label", "candidate_position"]
        ).to_parquet(path, index=False)
    return total_rows


def save_outputs(
    output_dir: str,
    data_mode: str,
    size: str,
    news: pd.DataFrame,
    train_pairs_count: int,
    dev_pairs_count: int,
    args: argparse.Namespace,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    news_path = out / f"mind_news_{data_mode}.parquet"
    manifest_path = out / f"mind_prepare_manifest_{data_mode}.json"
    news.to_parquet(news_path, index=False)
    manifest = {
        "scenario": "scenario2_news",
        "size": size,
        "data_mode": data_mode,
        "news_rows": int(len(news)),
        "train_pairs": int(train_pairs_count),
        "dev_pairs": int(dev_pairs_count),
        "config": vars(args),
        "files": {
            "news": str(news_path),
            "train_pairs": str(out / f"mind_train_pairs_{data_mode}.parquet"),
            "dev_pairs": str(out / f"mind_dev_pairs_{data_mode}.parquet"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved scenario-2 news processed files under %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    train_dir = discover_split_dir(args.raw_dir, args.size, "train")
    dev_dir = discover_split_dir(args.raw_dir, args.size, "dev")
    LOGGER.info("Using MIND train=%s dev=%s", train_dir, dev_dir)

    train_news = read_news(train_dir)
    dev_news = read_news(dev_dir)
    news = build_news_table(train_news, dev_news)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_pairs_count = write_pairs_parquet(
        train_dir,
        "train",
        out / f"mind_train_pairs_{args.data_mode}.parquet",
        args.max_train_impressions,
        args.behavior_chunk_size,
    )
    dev_pairs_count = write_pairs_parquet(
        dev_dir,
        "dev",
        out / f"mind_dev_pairs_{args.data_mode}.parquet",
        args.max_eval_impressions,
        args.behavior_chunk_size,
    )
    save_outputs(args.output_dir, args.data_mode, args.size, news, train_pairs_count, dev_pairs_count, args)


if __name__ == "__main__":
    main()
