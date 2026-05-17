"""
Prepare MIND data for scenario-3 news recommendation baselines.

Expected raw layout:
  data/raw/mind/MINDsmall_train/news.tsv
  data/raw/mind/MINDsmall_train/behaviors.tsv
  data/raw/mind/MINDsmall_dev/news.tsv
  data/raw/mind/MINDsmall_dev/behaviors.tsv

Recommended smoke command:
  python src/prepare_scenario3_mind.py \
    --raw_dir data/raw/mind --size small --data_mode smoke \
    --max_train_impressions 5000 --max_eval_impressions 500
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
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
    parser = argparse.ArgumentParser(description="Prepare MIND files for scenario-3 baselines.")
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
    path = split_dir / "news.tsv"
    news = pd.read_csv(
        path,
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
    path = split_dir / "behaviors.tsv"
    behaviors = pd.read_csv(
        path,
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


def _word_count(*parts: Any) -> int:
    text = " ".join(str(part or "") for part in parts)
    return int(len(re.findall(r"\b\w+\b", text)))


def collect_publish_time_proxy(behaviors: Iterable[pd.DataFrame]) -> Dict[str, pd.Timestamp]:
    """Use earliest observed behavior time as a reproducible publish-time proxy."""
    first_seen: Dict[str, pd.Timestamp] = {}
    for behavior_df in behaviors:
        iterator = tqdm(behavior_df.itertuples(index=False), total=len(behavior_df), desc="Collect publish proxy")
        for row in iterator:
            request_time = getattr(row, "request_time")
            if pd.isna(request_time):
                continue
            ids = _split_news_ids(getattr(row, "history", ""))
            ids.extend(_parse_impression_token(token)[0] for token in _split_news_ids(getattr(row, "impressions", "")))
            for news_id in ids:
                previous = first_seen.get(news_id)
                if previous is None or request_time < previous:
                    first_seen[news_id] = request_time
    return first_seen


def build_news_table(train_news: pd.DataFrame, dev_news: pd.DataFrame, publish_proxy: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    news = pd.concat([train_news, dev_news], ignore_index=True).drop_duplicates("news_id", keep="first").copy()
    for col in ["category", "subcategory", "title", "abstract"]:
        news[col] = news[col].fillna("").astype(str)
    news["category"] = news["category"].str.lower().replace("", "unknown")
    news["subcategory"] = news["subcategory"].str.lower().replace("", "unknown")
    news["text"] = (news["title"].fillna("") + " " + news["abstract"].fillna("")).str.strip()
    news["word_count"] = news.apply(lambda row: _word_count(row["title"], row["abstract"]), axis=1).astype(int)
    fallback_time = min(publish_proxy.values()) if publish_proxy else pd.Timestamp("2019-01-01")
    news["publish_time_proxy"] = news["news_id"].map(publish_proxy).fillna(fallback_time)
    return news.reset_index(drop=True)


def behaviors_to_pairs(behaviors: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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


def save_outputs(
    output_dir: str,
    data_mode: str,
    size: str,
    news: pd.DataFrame,
    train_pairs: pd.DataFrame,
    dev_pairs: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    news_path = out / f"mind_news_{data_mode}.parquet"
    train_path = out / f"mind_train_pairs_{data_mode}.parquet"
    dev_path = out / f"mind_dev_pairs_{data_mode}.parquet"
    manifest_path = out / f"mind_prepare_manifest_{data_mode}.json"
    news.to_parquet(news_path, index=False)
    train_pairs.to_parquet(train_path, index=False)
    dev_pairs.to_parquet(dev_path, index=False)
    manifest = {
        "size": size,
        "data_mode": data_mode,
        "news_rows": int(len(news)),
        "train_pairs": int(len(train_pairs)),
        "dev_pairs": int(len(dev_pairs)),
        "config": vars(args),
        "files": {
            "news": str(news_path),
            "train_pairs": str(train_path),
            "dev_pairs": str(dev_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved scenario-3 processed files under %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    train_dir = discover_split_dir(args.raw_dir, args.size, "train")
    dev_dir = discover_split_dir(args.raw_dir, args.size, "dev")
    LOGGER.info("Using MIND train=%s dev=%s", train_dir, dev_dir)

    train_news = read_news(train_dir)
    dev_news = read_news(dev_dir)
    train_behaviors = read_behaviors(train_dir, args.max_train_impressions)
    dev_behaviors = read_behaviors(dev_dir, args.max_eval_impressions)
    publish_proxy = collect_publish_time_proxy([train_behaviors, dev_behaviors])
    news = build_news_table(train_news, dev_news, publish_proxy)
    train_pairs = behaviors_to_pairs(train_behaviors, "train")
    dev_pairs = behaviors_to_pairs(dev_behaviors, "dev")
    save_outputs(args.output_dir, args.data_mode, args.size, news, train_pairs, dev_pairs, args)


if __name__ == "__main__":
    main()
