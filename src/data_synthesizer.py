"""
场景一（电商环境）半合成数据生成脚本。

该脚本只读取 Amazon Reviews 2023 的原始 review/meta 文件，不修改 raw data。
输出目标是构造可支撑“动态库存机会约束、新品扶持、供应商覆盖、预算软约束”
的研究级特征层，同时保留一个兼容旧需求的合并 parquet 文件。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORY = "All_Beauty"
DEFAULT_OUTPUT_PREFIX = "beauty_scenario1"
RANDOM_SEED = 42


# -----------------------------
# 基础 I/O 与字段清洗
# -----------------------------


def _stable_hash(text: Any, modulo: Optional[int] = None) -> int:
    """使用 SHA1 生成跨进程、跨机器稳定的整数 hash，避免 Python 内置 hash 随机化。"""
    digest = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value % modulo if modulo else value


def _normalize_text(value: Any) -> Optional[str]:
    """将空字符串、None、NaN 统一视为缺失；其它值转为去空白字符串。"""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def _first_non_empty(*values: Any) -> Optional[str]:
    """返回第一个非空文本值。"""
    for value in values:
        text = _normalize_text(value)
        if text:
            return text
    return None


def _parse_price(value: Any) -> float:
    """解析 Amazon meta 中的价格字段，非法或非正价格返回 NaN。"""
    if value is None:
        return np.nan
    if isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        price = float(value)
    elif isinstance(value, str):
        # 兼容 "$12.99"、"12.99" 等文本价格；范围价格不做复杂推断，取首个数值。
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        if not match:
            return np.nan
        price = float(match.group(0))
    else:
        return np.nan

    if not np.isfinite(price) or price <= 0:
        return np.nan
    return price


def _extract_brand(details: Any) -> Optional[str]:
    """从 details 字典里抽取品牌字段，兼容大小写差异。"""
    if not isinstance(details, dict):
        return None
    for key in ("Brand", "brand", "Manufacturer", "manufacturer"):
        value = _normalize_text(details.get(key))
        if value:
            return value
    return None


def _candidate_file(data_dir: Path, category: str, kind: str) -> Path:
    """定位原始 gzip 文件，兼容仓库中实际文件名与 Amazon 原始文件名。"""
    if kind == "reviews":
        candidates = [
            data_dir / f"{category}_reviews.jsonl.gz",
            data_dir / f"{category}.jsonl.gz",
        ]
    elif kind == "meta":
        candidates = [data_dir / f"meta_{category}.jsonl.gz"]
    else:
        raise ValueError(f"Unknown file kind: {kind}")

    for path in candidates:
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {kind} file for category={category}. Tried: {formatted}")


def _iter_jsonl_gz(path: Path, max_rows: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """逐行读取 jsonl.gz，坏行跳过但不中断整个合成流程。"""
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        for row_idx, line in enumerate(tqdm(file_obj, desc=f"Loading {path.name}")):
            if max_rows is not None and row_idx >= max_rows:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skip malformed JSON line %s in %s", row_idx, path)


def load_reviews(data_dir: Path, category: str, max_reviews: Optional[int]) -> pd.DataFrame:
    """加载交互数据，只保留本场景合成需要的稳定字段。"""
    path = _candidate_file(data_dir, category, "reviews")
    rows: List[Dict[str, Any]] = []

    for record in _iter_jsonl_gz(path, max_reviews):
        item_id = _first_non_empty(record.get("parent_asin"), record.get("asin"))
        user_id = _normalize_text(record.get("user_id"))
        if not item_id or not user_id:
            continue
        rows.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "rating": record.get("rating"),
                "timestamp": record.get("timestamp"),
            }
        )

    reviews = pd.DataFrame(rows)
    if reviews.empty:
        raise ValueError("No valid review interactions loaded.")

    reviews["timestamp"] = pd.to_numeric(reviews["timestamp"], errors="coerce")
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews = reviews.dropna(subset=["timestamp"]).copy()
    reviews["timestamp"] = reviews["timestamp"].astype("int64")
    LOGGER.info(
        "Loaded %s interactions, %s users, %s items",
        len(reviews),
        reviews["user_id"].nunique(),
        reviews["item_id"].nunique(),
    )
    return reviews


def load_metadata(data_dir: Path, category: str) -> pd.DataFrame:
    """加载商品元数据，并标准化价格、品牌、卖家等字段。"""
    path = _candidate_file(data_dir, category, "meta")
    rows: List[Dict[str, Any]] = []

    for record in _iter_jsonl_gz(path):
        item_id = _first_non_empty(record.get("parent_asin"), record.get("asin"))
        if not item_id:
            continue

        details = record.get("details") or {}
        store = _normalize_text(record.get("store"))
        brand = _extract_brand(details)
        title = _normalize_text(record.get("title")) or "Unknown"
        main_category = _normalize_text(record.get("main_category")) or "Unknown"

        rows.append(
            {
                "item_id": item_id,
                "title": title,
                "main_category": main_category,
                "store": store,
                "raw_brand": brand,
                "price_raw": _parse_price(record.get("price")),
            }
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError("No valid metadata loaded.")

    # parent_asin 可能在 meta 中重复，保留第一条稳定记录即可。
    metadata = metadata.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    LOGGER.info("Loaded %s metadata items", len(metadata))
    return metadata


# -----------------------------
# 特征合成逻辑
# -----------------------------


def fill_prices(items: pd.DataFrame) -> pd.DataFrame:
    """按 store -> brand -> main_category -> global median 的顺序填充价格。"""
    items = items.copy()
    items["price_missing"] = items["price_raw"].isna()
    items["price_filled"] = items["price_raw"]
    items["price_fill_source"] = np.where(items["price_raw"].notna(), "raw", None)

    global_median = float(items["price_raw"].median()) if items["price_raw"].notna().any() else 1.0
    if not np.isfinite(global_median) or global_median <= 0:
        global_median = 1.0

    for group_col, source_name in (
        ("store", "store_median"),
        ("raw_brand", "brand_median"),
        ("main_category", "category_median"),
    ):
        medians = items.groupby(group_col, dropna=True)["price_raw"].transform("median")
        mask = items["price_filled"].isna() & medians.notna()
        items.loc[mask, "price_filled"] = medians[mask]
        items.loc[mask, "price_fill_source"] = source_name

    mask = items["price_filled"].isna()
    items.loc[mask, "price_filled"] = global_median
    items.loc[mask, "price_fill_source"] = "global_median"
    items["price_filled"] = items["price_filled"].astype(float)
    items["price_missing"] = items["price_missing"].astype(bool)
    return items


def add_identity_fields(items: pd.DataFrame) -> pd.DataFrame:
    """构造供应商和品牌代理字段，并记录是否发生缺失回退。"""
    items = items.copy()

    item_hashes = items["item_id"].map(lambda item_id: _stable_hash(item_id, modulo=1_000_000))
    fallback_sellers = "seller_fallback_" + item_hashes.astype(str)
    fallback_brands = "brand_fallback_" + item_hashes.astype(str)

    items["seller_missing"] = items["store"].isna()
    items["brand_missing"] = items["raw_brand"].isna() & items["store"].isna()

    # seller_id 优先使用 store；brand_id 优先使用 details.Brand，其次 store。
    items["seller_id"] = items["store"].where(items["store"].notna(), fallback_sellers)
    items["brand_id"] = items["raw_brand"].where(items["raw_brand"].notna(), items["store"])
    items["brand_id"] = items["brand_id"].where(items["brand_id"].notna(), fallback_brands)
    return items


def add_interaction_features(items: pd.DataFrame, reviews: pd.DataFrame, new_item_window_ratio: float) -> Tuple[pd.DataFrame, int]:
    """基于交互频次和首次出现时间生成 popularity、first_seen_ts、is_new。"""
    if not 0 < new_item_window_ratio < 1:
        raise ValueError("new_item_window_ratio must be in (0, 1).")

    item_stats = reviews.groupby("item_id").agg(
        interaction_count=("item_id", "size"),
        first_seen_ts=("timestamp", "min"),
    )
    items = items.merge(item_stats, on="item_id", how="left")
    items["interaction_count"] = items["interaction_count"].fillna(0).astype("int64")

    min_ts = int(reviews["timestamp"].min())
    max_ts = int(reviews["timestamp"].max())
    cutoff_ts = int(max_ts - (max_ts - min_ts) * new_item_window_ratio)

    # 没有任何交互的 meta 商品不能从行为数据证明为新品，因此默认 False。
    items["is_new"] = items["first_seen_ts"].notna() & (items["first_seen_ts"] >= cutoff_ts)
    items["first_seen_ts"] = items["first_seen_ts"].astype("Int64")

    max_count = max(int(items["interaction_count"].max()), 1)
    items["popularity"] = np.log1p(items["interaction_count"]) / math.log1p(max_count)
    items["popularity"] = items["popularity"].clip(0.0, 1.0).astype(float)
    return items, cutoff_ts


def _normal_survival(z: np.ndarray) -> np.ndarray:
    """标准正态 survival function：P(Z > z)。避免引入 scipy 作为额外依赖。"""
    erfc_vec = np.vectorize(math.erfc, otypes=[float])
    return 0.5 * erfc_vec(z / math.sqrt(2.0))


def add_inventory_features(items: pd.DataFrame, seed: int, min_inventory: int) -> pd.DataFrame:
    """合成动态库存及未来可用概率，供机会约束直接消费。"""
    if min_inventory < 1:
        raise ValueError("min_inventory must be >= 1.")

    items = items.copy()
    rng = np.random.default_rng(seed)
    popularity = items["popularity"].to_numpy(dtype=float)

    # 热门商品拥有更高备货，同时保留噪声模拟供应链与盘点扰动。
    base_inventory = min_inventory + 95.0 * popularity
    noise = rng.normal(loc=0.0, scale=4.0 + 10.0 * popularity, size=len(items))
    inventory_initial = np.rint(base_inventory + noise).astype(int)
    inventory_initial = np.maximum(inventory_initial, min_inventory)

    # 未来需求均值随流行度上升；波动也随流行度上升，用于估计点击/下单时缺货风险。
    inventory_mu = 1.5 + 60.0 * popularity
    inventory_sigma = 1.0 + 12.0 * popularity
    z = (inventory_initial.astype(float) - inventory_mu) / inventory_sigma
    stockout_risk = _normal_survival(z)
    stockout_risk = np.clip(stockout_risk, 0.0, 1.0)

    items["inventory_initial"] = inventory_initial.astype("int64")
    items["inventory_mu"] = inventory_mu.astype(float)
    items["inventory_sigma"] = inventory_sigma.astype(float)
    items["stockout_risk"] = stockout_risk.astype(float)
    items["inventory_available_prob"] = (1.0 - stockout_risk).astype(float)
    return items


def build_item_features(
    metadata: pd.DataFrame,
    reviews: pd.DataFrame,
    new_item_window_ratio: float,
    seed: int,
    min_inventory: int,
) -> Tuple[pd.DataFrame, int]:
    """生成完整商品侧特征表。"""
    items = metadata.copy()
    items = fill_prices(items)
    items = add_identity_fields(items)
    items, cutoff_ts = add_interaction_features(items, reviews, new_item_window_ratio)
    items = add_inventory_features(items, seed, min_inventory)

    columns = [
        "item_id",
        "title",
        "main_category",
        "brand_id",
        "seller_id",
        "seller_missing",
        "brand_missing",
        "price_filled",
        "price_missing",
        "price_fill_source",
        "popularity",
        "interaction_count",
        "first_seen_ts",
        "is_new",
        "inventory_initial",
        "inventory_mu",
        "inventory_sigma",
        "stockout_risk",
        "inventory_available_prob",
    ]
    return items[columns].copy(), cutoff_ts


def build_user_features(
    reviews: pd.DataFrame,
    item_features: pd.DataFrame,
    budget_tolerance_ratio: float,
) -> pd.DataFrame:
    """基于用户历史交互商品价格中位数合成目标预算和容忍区间。"""
    if budget_tolerance_ratio < 0:
        raise ValueError("budget_tolerance_ratio must be >= 0.")

    global_median = float(item_features["price_filled"].median())
    price_lookup = item_features[["item_id", "price_filled"]]
    user_prices = reviews[["user_id", "item_id"]].merge(price_lookup, on="item_id", how="left")

    user_stats = user_prices.groupby("user_id").agg(
        history_count=("item_id", "size"),
        target_budget=("price_filled", "median"),
    )
    user_stats = user_stats.reset_index()
    user_stats["budget_source"] = np.where(
        user_stats["target_budget"].notna(),
        "user_history_median",
        "global_median_fallback",
    )
    user_stats["target_budget"] = user_stats["target_budget"].fillna(global_median).astype(float)
    user_stats["budget_tolerance"] = user_stats["target_budget"] * float(budget_tolerance_ratio)
    user_stats["budget_low"] = (user_stats["target_budget"] - user_stats["budget_tolerance"]).clip(lower=0.0)
    user_stats["budget_high"] = user_stats["target_budget"] + user_stats["budget_tolerance"]

    columns = [
        "user_id",
        "history_count",
        "target_budget",
        "budget_tolerance",
        "budget_low",
        "budget_high",
        "budget_source",
    ]
    return user_stats[columns].copy()


def build_interaction_features(reviews: pd.DataFrame, item_features: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    """生成干净交互表，仅附加必要的价格和预算字段，避免信息过度重复。"""
    interactions = reviews.copy()
    interactions = interactions.merge(
        item_features[["item_id", "price_filled", "is_new", "inventory_initial"]],
        on="item_id",
        how="left",
    )
    interactions = interactions.merge(
        user_features[["user_id", "target_budget", "budget_tolerance"]],
        on="user_id",
        how="left",
    )
    return interactions


def build_compat_table(reviews: pd.DataFrame, item_features: pd.DataFrame, user_features: pd.DataFrame) -> pd.DataFrame:
    """按旧需求生成一个交互主表 + 商品特征 + 用户预算的合并 parquet。"""
    compat = reviews.copy()
    compat = compat.merge(item_features, on="item_id", how="left")
    compat = compat.merge(user_features, on="user_id", how="left")
    return compat


# -----------------------------
# 质量校验与保存
# -----------------------------


def validate_outputs(
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    compat: pd.DataFrame,
    budget_tolerance_ratio: float,
    min_inventory: int,
) -> None:
    """执行数据质量断言，失败时直接阻止保存错误数据。"""
    if item_features["price_filled"].isna().any():
        raise AssertionError("price_filled contains null values.")
    if user_features["target_budget"].isna().any():
        raise AssertionError("target_budget contains null values.")
    if item_features["inventory_initial"].isna().any():
        raise AssertionError("inventory_initial contains null values.")
    if (item_features["inventory_initial"] < min_inventory).any():
        raise AssertionError(f"inventory_initial contains values below {min_inventory}.")
    if not item_features["stockout_risk"].between(0.0, 1.0).all():
        raise AssertionError("stockout_risk must be in [0, 1].")
    if not item_features["inventory_available_prob"].between(0.0, 1.0).all():
        raise AssertionError("inventory_available_prob must be in [0, 1].")

    expected_tolerance = user_features["target_budget"] * float(budget_tolerance_ratio)
    if not np.allclose(user_features["budget_tolerance"], expected_tolerance):
        raise AssertionError("budget_tolerance does not match configured ratio.")

    required_compat_cols = {"inventory_initial", "is_new", "target_budget", "budget_tolerance"}
    missing = required_compat_cols - set(compat.columns)
    if missing:
        raise AssertionError(f"Compatibility table missing columns: {sorted(missing)}")


def save_outputs(
    output_dir: Path,
    prefix: str,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    interactions: pd.DataFrame,
    compat: pd.DataFrame,
) -> Dict[str, Path]:
    """写出三张干净表和一个兼容合并表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "items": output_dir / f"{prefix}_items.parquet",
        "users": output_dir / f"{prefix}_users.parquet",
        "interactions": output_dir / f"{prefix}_interactions.parquet",
        "synthetic": output_dir / f"{prefix}_synthetic.parquet",
    }
    item_features.to_parquet(paths["items"], index=False)
    user_features.to_parquet(paths["users"], index=False)
    interactions.to_parquet(paths["interactions"], index=False)
    compat.to_parquet(paths["synthetic"], index=False)
    return paths


def synthesize(args: argparse.Namespace) -> Dict[str, Path]:
    """完整执行场景一数据合成。"""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    prefix = args.output_prefix or DEFAULT_OUTPUT_PREFIX

    reviews = load_reviews(data_dir, args.category, args.max_reviews)
    metadata = load_metadata(data_dir, args.category)

    # 仅保留与当前交互集合相关的商品，可避免输出大量没有行为支撑的 meta-only 商品。
    metadata = metadata[metadata["item_id"].isin(reviews["item_id"].unique())].copy()
    if metadata.empty:
        raise ValueError("No metadata rows match loaded review item_ids.")

    item_features, cutoff_ts = build_item_features(
        metadata=metadata,
        reviews=reviews,
        new_item_window_ratio=args.new_item_window_ratio,
        seed=args.seed,
        min_inventory=args.min_inventory,
    )
    user_features = build_user_features(
        reviews=reviews,
        item_features=item_features,
        budget_tolerance_ratio=args.budget_tolerance_ratio,
    )
    interactions = build_interaction_features(reviews, item_features, user_features)
    compat = build_compat_table(reviews, item_features, user_features)

    validate_outputs(
        item_features=item_features,
        user_features=user_features,
        compat=compat,
        budget_tolerance_ratio=args.budget_tolerance_ratio,
        min_inventory=args.min_inventory,
    )
    paths = save_outputs(output_dir, prefix, item_features, user_features, interactions, compat)

    LOGGER.info("New item cutoff timestamp: %s", cutoff_ts)
    LOGGER.info(
        "Summary: interactions=%s, users=%s, items=%s, new_items=%s, price_missing_rate=%.4f",
        len(reviews),
        len(user_features),
        len(item_features),
        int(item_features["is_new"].sum()),
        float(item_features["price_missing"].mean()),
    )
    for name, path in paths.items():
        LOGGER.info("Wrote %s: %s", name, path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scenario-1 semi-synthetic e-commerce features.")
    parser.add_argument("--data_dir", default="data", help="Directory containing Amazon review/meta jsonl.gz files.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Amazon category name, e.g. All_Beauty.")
    parser.add_argument("--output_dir", default="data/processed", help="Directory for generated parquet files.")
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX, help="Output file prefix.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible inventory noise.")
    parser.add_argument("--max_reviews", type=int, default=None, help="Optional cap on loaded review rows for smoke tests.")
    parser.add_argument("--new_item_window_ratio", type=float, default=0.1, help="Last time-window ratio used to mark new items.")
    parser.add_argument("--budget_tolerance_ratio", type=float, default=0.2, help="Budget tolerance as target_budget ratio.")
    parser.add_argument("--min_inventory", type=int, default=5, help="Minimum synthesized initial inventory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    synthesize(args)


if __name__ == "__main__":
    main()
