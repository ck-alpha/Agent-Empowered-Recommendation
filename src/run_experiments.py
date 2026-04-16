"""
Experiment Runner for DualAgent-Rec.
Runs experiments with different configurations and baselines.

Key experiments:
1. Main comparison: DualAgent-Rec vs ablation variants
2. Ablation study: Effect of each component
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from dualagent_rec import DualAgentRec, DualAgentConfig
from agents import (
    GreedyRerankConfig,
    GreedyRerankingBaseline,
    InProcessingConfig,
    WeightedSumInProcessingBaseline,
)
from constraints import ConstraintConfig, ConstraintHandler
from data_utils import AmazonDataLoader, UserBehaviorProcessor
from evaluation import RecommendationMetrics, MultiObjectiveMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# [Task 1: 调高约束]
# 使用当前实验口径：统一严评阈值（w/o_Constraints 仅训练无约束，评估按此口径重算合法率）。
MAIN_FAIRNESS_THRESHOLD = 0.30
MAIN_SELLER_COVERAGE_THRESHOLD = 0.40
MAIN_NEW_ITEM_THRESHOLD = 0.25


def _strict_constraint_handler() -> ConstraintHandler:
    """Create a strict (epsilon=0) constraint handler for unified feasibility evaluation."""
    handler = ConstraintHandler(
        ConstraintConfig(
            fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
            seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
            new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
            epsilon_initial=0.0,
            epsilon_decay=1.0,
        )
    )
    handler.epsilon = 0.0
    return handler


def evaluate_strict_feasibility(
    recommended_items: List[str],
    item_features: Dict[str, Dict[str, Any]],
) -> Tuple[float, List[float]]:
    """Evaluate feasibility with unified strict thresholds."""
    handler = _strict_constraint_handler()
    violations = handler.calculate_violations(recommended_items or [], item_features)
    is_feasible = float(all(v <= 1e-12 for v in violations))
    return is_feasible, [float(v) for v in violations]


def load_amazon_data(data_dir: str, category: str = 'All_Beauty', max_reviews: int = 10000, n_users: int = 20) -> Tuple:
    """
    Load real Amazon data.

    Returns:
        Tuple of (candidate_items, user_train_histories, user_test_ground_truths, item_features, user_profiles)
    """
    logger.info(f"Loading Amazon {category} dataset...")

    loader = AmazonDataLoader(data_dir, category=category)

    # Download if needed
    loader.download_dataset()

    # Load and preprocess - scale max_reviews with n_users
    adjusted_max_reviews = max(max_reviews, n_users * 500)  # Ensure enough data
    df, metadata = loader.preprocess(
        min_user_interactions=5,
        min_item_interactions=3,
        max_reviews=adjusted_max_reviews
    )

    # Add compatibility fields to metadata
    for item_id, item_info in metadata.items():
        # Ensure both 'category' and 'main_category' exist
        if 'main_category' in item_info and 'category' not in item_info:
            item_info['category'] = item_info['main_category']
        elif 'category' in item_info and 'main_category' not in item_info:
            item_info['main_category'] = item_info['category']

        # Add seller_id if not present (use asin prefix as proxy)
        if 'seller_id' not in item_info:
            item_info['seller_id'] = f"seller_{hash(item_id) % 100}"

        # Add is_new flag (randomly for demo)
        if 'is_new' not in item_info:
            item_info['is_new'] = np.random.random() < 0.15

        # Add popularity based on interaction count
        if 'popularity' not in item_info:
            item_info['popularity'] = 0.5

    # Build user processor
    processor = UserBehaviorProcessor(df, metadata)

    # Get candidate items
    candidate_items = list(metadata.keys())

    # Calculate item popularity from interactions
    item_counts = df['item_id'].value_counts()
    max_count = item_counts.max() if len(item_counts) > 0 else 1
    for item_id in metadata:
        count = item_counts.get(item_id, 0)
        metadata[item_id]['popularity'] = count / max_count

    # Sample users for experiments
    user_ids = processor.sample_users(n_users=n_users, min_interactions=10)

    user_train_histories = {}
    user_test_ground_truths = {}
    user_profiles = {}
    for user_id in user_ids:
        train_history, test_ground_truth = processor.get_train_test_history(
            user_id=user_id,
            max_items=50,
            train_ratio=0.8
        )
        # 修改点：实验仅保留 train/test 都有效的用户，避免评测空集。
        if not train_history or not test_ground_truth:
            continue

        user_train_histories[user_id] = train_history
        user_test_ground_truths[user_id] = test_ground_truth
        user_profiles[user_id] = processor.get_user_profile(user_id)

    logger.info(
        f"Loaded {len(candidate_items)} items, {len(user_train_histories)} users "
        f"(with temporal train/test split)"
    )

    return candidate_items, user_train_histories, user_test_ground_truths, metadata, user_profiles


def run_single_experiment(
    config: DualAgentConfig,
    candidate_items: List[str],
    train_history: List[Dict],
    test_ground_truth: List[Dict],
    item_features: Dict[str, Dict],
    user_profile: Dict[str, Any],
    experiment_name: str,
    include_pareto_solutions: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment with given configuration."""
    logger.info(f"Running experiment: {experiment_name}")

    framework = DualAgentRec(config)

    start_time = datetime.now()
    best_solutions, metrics = framework.optimize(
        candidate_items=candidate_items,
        user_history=train_history,
        item_features=item_features,
        user_profile=user_profile
    )
    end_time = datetime.now()

    # 修改点：真实离线指标仅使用 test_ground_truth 评估，不参与优化过程。
    balanced_solution = framework.get_recommendation('balanced')
    recommended_items = balanced_solution.item_ids if balanced_solution else []
    offline_eval = RecommendationMetrics.evaluate_with_ground_truth(
        recommended=recommended_items,
        test_ground_truth=test_ground_truth,
        k=10
    )

    metrics['real_ndcg@10'] = offline_eval.get('ndcg@k', 0.0)
    metrics['real_hr@10'] = offline_eval.get('hr@k', 0.0)
    metrics['recommended_items'] = recommended_items

    # [Task 1: 调高约束]
    # 统一严评：所有方法输出合法率都按同一阈值和 epsilon=0 重算。
    strict_feasibility, strict_violations = evaluate_strict_feasibility(
        recommended_items=recommended_items,
        item_features=item_features,
    )
    metrics['internal_feasibility_rate'] = float(metrics.get('feasibility_rate', 0.0))
    metrics['feasibility_rate'] = strict_feasibility
    metrics['strict_constraint_violations'] = strict_violations
    metrics['strict_constraint_thresholds'] = {
        'fairness_threshold': MAIN_FAIRNESS_THRESHOLD,
        'seller_coverage_threshold': MAIN_SELLER_COVERAGE_THRESHOLD,
        'new_item_threshold': MAIN_NEW_ITEM_THRESHOLD,
    }

    metrics['experiment_name'] = experiment_name
    metrics['runtime_seconds'] = (end_time - start_time).total_seconds()
    if framework.history:
        metrics['history'] = framework.history

    if include_pareto_solutions:
        pareto_solutions = []
        for ind in best_solutions:
            offline_point = RecommendationMetrics.evaluate_with_ground_truth(
                recommended=ind.item_ids,
                test_ground_truth=test_ground_truth,
                k=10
            )
            pareto_solutions.append({
                'items': ind.item_ids,
                'avg_accuracy': float(ind.scores[0]) if len(ind.scores) > 0 else 0.0,
                'avg_diversity': float(ind.scores[1]) if len(ind.scores) > 1 else 0.0,
                'avg_novelty': float(ind.scores[2]) if len(ind.scores) > 2 else 0.0,
                'real_ndcg@10': offline_point.get('ndcg@k', 0.0),
                'real_hr@10': offline_point.get('hr@k', 0.0),
                'constraint_violations': ind.constraint_violations.tolist(),
                'is_feasible': bool(ind.is_feasible),
            })
        metrics['pareto_solutions'] = pareto_solutions

    return metrics


def run_inprocessing_baseline_experiment(
    candidate_items: List[str],
    train_history: List[Dict[str, Any]],
    test_ground_truth: List[Dict[str, Any]],
    item_features: Dict[str, Dict[str, Any]],
    weights: Tuple[float, float, float],
    penalty_lambda: float,
    experiment_name: str,
    population_size: int = 100,
    max_generations: int = 50,
    recommendation_size: int = 10,
    fairness_threshold: float = MAIN_FAIRNESS_THRESHOLD,
    seller_coverage_threshold: float = MAIN_SELLER_COVERAGE_THRESHOLD,
    new_item_threshold: float = MAIN_NEW_ITEM_THRESHOLD,
) -> Dict[str, Any]:
    """Run weighted-sum in-processing baseline for one user."""
    baseline = WeightedSumInProcessingBaseline(
        InProcessingConfig(
            population_size=population_size,
            max_generations=max_generations,
            recommendation_size=recommendation_size,
            fairness_threshold=fairness_threshold,
            seller_coverage_threshold=seller_coverage_threshold,
            new_item_threshold=new_item_threshold,
        )
    )
    start_time = datetime.now()
    metrics = baseline.optimize(
        candidate_items=candidate_items,
        user_history=train_history,
        test_ground_truth=test_ground_truth,
        item_features=item_features,
        weights=weights,
        penalty_lambda=penalty_lambda,
    )
    end_time = datetime.now()
    if not metrics:
        return {}

    recommended_items = metrics.get('recommended_items', [])
    strict_feasibility, strict_violations = evaluate_strict_feasibility(
        recommended_items=recommended_items,
        item_features=item_features,
    )
    metrics['internal_feasibility_rate'] = float(metrics.get('feasibility_rate', 0.0))
    metrics['feasibility_rate'] = strict_feasibility
    metrics['strict_constraint_violations'] = strict_violations
    metrics['strict_constraint_thresholds'] = {
        'fairness_threshold': fairness_threshold,
        'seller_coverage_threshold': seller_coverage_threshold,
        'new_item_threshold': new_item_threshold,
    }

    metrics['experiment_name'] = experiment_name
    metrics['runtime_seconds'] = (end_time - start_time).total_seconds()
    return metrics


def run_greedy_reranking_baseline_experiment(
    candidate_items: List[str],
    train_history: List[Dict[str, Any]],
    test_ground_truth: List[Dict[str, Any]],
    item_features: Dict[str, Dict[str, Any]],
    experiment_name: str,
    candidate_pool_size: int = 50,
    recommendation_size: int = 10,
    fairness_threshold: float = MAIN_FAIRNESS_THRESHOLD,
    seller_coverage_threshold: float = MAIN_SELLER_COVERAGE_THRESHOLD,
    new_item_threshold: float = MAIN_NEW_ITEM_THRESHOLD,
) -> Dict[str, Any]:
    """Run post-processing greedy reranking baseline for one user."""
    baseline = GreedyRerankingBaseline(
        GreedyRerankConfig(
            candidate_pool_size=candidate_pool_size,
            recommendation_size=recommendation_size,
            fairness_threshold=fairness_threshold,
            seller_coverage_threshold=seller_coverage_threshold,
            new_item_threshold=new_item_threshold,
        )
    )
    start_time = datetime.now()
    metrics = baseline.optimize(
        candidate_items=candidate_items,
        user_history=train_history,
        test_ground_truth=test_ground_truth,
        item_features=item_features,
    )
    end_time = datetime.now()
    if not metrics:
        return {}

    recommended_items = metrics.get('recommended_items', [])
    strict_feasibility, strict_violations = evaluate_strict_feasibility(
        recommended_items=recommended_items,
        item_features=item_features,
    )
    metrics['internal_feasibility_rate'] = float(metrics.get('feasibility_rate', 0.0))
    metrics['feasibility_rate'] = strict_feasibility
    metrics['strict_constraint_violations'] = strict_violations
    metrics['strict_constraint_thresholds'] = {
        'fairness_threshold': fairness_threshold,
        'seller_coverage_threshold': seller_coverage_threshold,
        'new_item_threshold': new_item_threshold,
    }

    metrics['experiment_name'] = experiment_name
    metrics['runtime_seconds'] = (end_time - start_time).total_seconds()
    return metrics


def aggregate_user_metrics(
    user_results: List[Dict[str, Any]],
    method_name: str,
    run_index: int,
) -> Dict[str, Any]:
    """Aggregate user-level metrics into one run-level metric dict."""
    aggregated = {
        'hypervolume': np.mean([r.get('hypervolume', 0) for r in user_results]),
        'spacing': np.mean([r.get('spacing', 0) for r in user_results]),
        'pareto_size': int(np.mean([r.get('pareto_size', 0) for r in user_results])),
        'avg_accuracy': np.mean([r.get('avg_accuracy', 0) for r in user_results]),
        'avg_diversity': np.mean([r.get('avg_diversity', 0) for r in user_results]),
        'avg_novelty': np.mean([r.get('avg_novelty', 0) for r in user_results]),
        'real_ndcg@10': np.mean([r.get('real_ndcg@10', 0) for r in user_results]),
        'real_hr@10': np.mean([r.get('real_hr@10', 0) for r in user_results]),
        'feasibility_rate': np.mean([r.get('feasibility_rate', 0) for r in user_results]),
        'runtime_seconds': np.sum([r.get('runtime_seconds', 0) for r in user_results]),
        'experiment_name': f"{method_name}_run{run_index + 1}",
        'num_users': len(user_results),
    }

    # Preserve DualAgent per-run Pareto set for real trade-off visualization.
    if any('pareto_solutions' in r for r in user_results):
        pareto_solutions = []
        for r in user_results:
            user_id = r.get('_user_id')
            for point in r.get('pareto_solutions', []):
                enriched = dict(point)
                if user_id is not None:
                    enriched['user_id'] = user_id
                pareto_solutions.append(enriched)
        aggregated['pareto_solutions'] = pareto_solutions
        aggregated['pareto_point_count'] = len(pareto_solutions)

    # [Task 2: 每代记录]
    # Aggregate per-user generation history to run-level history.
    if any(isinstance(r.get('history'), list) for r in user_results):
        gen_feas_map: Dict[int, List[float]] = {}
        gen_hv_map: Dict[int, List[float]] = {}
        for r in user_results:
            history = r.get('history') or []
            if not isinstance(history, list):
                continue
            for row in history:
                if not isinstance(row, dict):
                    continue
                gen = row.get('generation')
                if not isinstance(gen, int):
                    continue
                feas = row.get('feasibility_rate')
                if feas is None and isinstance(row.get('constraint_metrics'), dict):
                    feas = row['constraint_metrics'].get('overall_feasibility')
                hv = row.get('hypervolume')
                if isinstance(feas, (int, float)):
                    gen_feas_map.setdefault(gen, []).append(float(feas))
                if isinstance(hv, (int, float)):
                    gen_hv_map.setdefault(gen, []).append(float(hv))

        if gen_feas_map or gen_hv_map:
            history_rows = []
            all_generations = sorted(set(gen_feas_map.keys()) | set(gen_hv_map.keys()))
            for gen in all_generations:
                row: Dict[str, Any] = {'generation': gen}
                if gen in gen_feas_map:
                    row['feasibility_rate'] = float(np.mean(gen_feas_map[gen]))
                if gen in gen_hv_map:
                    row['hypervolume'] = float(np.mean(gen_hv_map[gen]))
                history_rows.append(row)
            aggregated['history'] = history_rows
            aggregated['history_point_count'] = len(history_rows)

    optional_numeric_keys = [
        'internal_feasibility_rate',
        'initial_real_ndcg@10',
        'initial_real_hr@10',
        'initial_total_violation',
        'final_total_violation',
        'num_rerank_iterations',
        'candidate_pool_size',
        'recommendation_size',
        'penalty_lambda',
    ]
    for key in optional_numeric_keys:
        if any(key in r for r in user_results):
            aggregated[key] = float(np.mean([r.get(key, 0.0) for r in user_results]))

    if any('strict_constraint_thresholds' in r for r in user_results):
        aggregated['strict_constraint_thresholds'] = {
            'fairness_threshold': MAIN_FAIRNESS_THRESHOLD,
            'seller_coverage_threshold': MAIN_SELLER_COVERAGE_THRESHOLD,
            'new_item_threshold': MAIN_NEW_ITEM_THRESHOLD,
        }

    # Preserve metadata for plotting and analysis.
    for key in ['weights', 'weight_combo', 'penalty_level', 'baseline_type']:
        if key in user_results[0]:
            aggregated[key] = user_results[0][key]

    return aggregated


def run_main_comparison(
    candidate_items: List[str],
    user_train_histories: Dict[str, List[Dict]],
    user_test_ground_truths: Dict[str, List[Dict]],
    item_features: Dict[str, Dict],
    user_profiles: Dict[str, Dict],
    output_dir: str,
    use_llm: bool = False,
    num_runs: int = 3
) -> Dict[str, List[Dict]]:
    """
    Run main comparison experiments.

    Compares:
    1. DualAgent-Rec (full model)
    2. w/o LLM Coordinator (rule-based allocation)
    3. w/o Dual-Agent (single population)
    4. w/o Hard Constraints (soft penalty only)
    5. In-processing baselines (weighted sum + soft penalty)
    6. Post-processing baseline (greedy reranking)
    """
    results = {}
    # [Task 1: 调高约束]
    # 统一主实验阈值（当前口径：fairness=0.30, seller=0.40, new_item=0.25）。
    dualagent_configs = {
        'DualAgent-Rec': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=use_llm,
            llm_model='qwen2.5:14b',
            fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
            seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
            new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
        ),
        'w/o_LLM': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=False,  # Disable LLM
            fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
            seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
            new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
        ),
        'w/o_Constraints': DualAgentConfig(
            population_size=100,
            max_generations=50,
            recommendation_size=10,
            use_llm=False,
            # 仅训练阶段关闭约束；最终合法率会用统一严评口径重算。
            fairness_threshold=0.0,
            seller_coverage_threshold=0.0,
            new_item_threshold=0.0,
        ),
        'Single_Population': DualAgentConfig(
            population_size=200,  # Combined population
            max_generations=50,
            recommendation_size=10,
            is_single_population=True,  # 修改点：启用真实单种群基线
            use_llm=False,
            fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
            seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
            new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
        ),
    }

    inprocessing_configs = {
        # A: accuracy-heavy
        'In-processing (w_A, weak)': {
            'weights': (0.8, 0.1, 0.1),
            'penalty_lambda': 0.1,
            'weight_combo': 'A',
            'penalty_level': 'weak',
        },
        'In-processing (w_A, strong)': {
            'weights': (0.8, 0.1, 0.1),
            'penalty_lambda': 100.0,
            'weight_combo': 'A',
            'penalty_level': 'strong',
        },
        # B: balanced
        'In-processing (w_B, weak)': {
            'weights': (0.4, 0.3, 0.3),
            'penalty_lambda': 0.1,
            'weight_combo': 'B',
            'penalty_level': 'weak',
        },
        'In-processing (w_B, strong)': {
            'weights': (0.4, 0.3, 0.3),
            'penalty_lambda': 100.0,
            'weight_combo': 'B',
            'penalty_level': 'strong',
        },
        # C: diversity-heavy
        'In-processing (w_C, weak)': {
            'weights': (0.2, 0.6, 0.2),
            'penalty_lambda': 0.1,
            'weight_combo': 'C',
            'penalty_level': 'weak',
        },
        'In-processing (w_C, strong)': {
            'weights': (0.2, 0.6, 0.2),
            'penalty_lambda': 100.0,
            'weight_combo': 'C',
            'penalty_level': 'strong',
        },
    }

    postprocessing_configs = {
        'Greedy_Reranking': {
            'candidate_pool_size': 50,
            'recommendation_size': 10,
        }
    }

    # Use all users for main comparison
    user_ids = sorted(set(user_train_histories.keys()) & set(user_test_ground_truths.keys()))
    logger.info(f"Running experiments on {len(user_ids)} users")

    # 1) DualAgent family
    for method_name, config in dualagent_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*50}")

        method_results = []
        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            np.random.seed(RANDOM_SEED + run)

            # Run on each user and aggregate
            user_results = []
            for user_id in user_ids:
                try:
                    metrics = run_single_experiment(
                        config=config,
                        candidate_items=candidate_items,
                        train_history=user_train_histories[user_id],
                        test_ground_truth=user_test_ground_truths[user_id],
                        item_features=item_features,
                        user_profile=user_profiles[user_id],
                        experiment_name=f"{method_name}_run{run+1}_{user_id}",
                        include_pareto_solutions=(method_name == 'DualAgent-Rec'),
                    )
                    metrics['_user_id'] = user_id
                    user_results.append(metrics)
                except Exception as e:
                    logger.error(f"  User {user_id} failed: {e}")
                    continue

            if user_results:
                method_results.append(aggregate_user_metrics(user_results, method_name, run))

        results[method_name] = method_results

        # Log summary
        if method_results:
            avg_hv = np.mean([r.get('hypervolume', 0) for r in method_results])
            avg_acc = np.mean([r.get('avg_accuracy', 0) for r in method_results])
            avg_div = np.mean([r.get('avg_diversity', 0) for r in method_results])
            avg_real_ndcg = np.mean([r.get('real_ndcg@10', 0) for r in method_results])
            avg_real_hr = np.mean([r.get('real_hr@10', 0) for r in method_results])
            avg_feasibility = np.mean([r.get('feasibility_rate', 0) for r in method_results])
            logger.info(
                f"  Avg HV: {avg_hv:.4f}, ProxyAcc: {avg_acc:.4f}, Div: {avg_div:.4f}, "
                f"RealNDCG@10: {avg_real_ndcg:.4f}, RealHR@10: {avg_real_hr:.4f}, "
                f"Feas: {avg_feasibility:.2%}"
            )

    # 2) In-processing weighted-sum baselines
    for method_name, baseline_cfg in inprocessing_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*50}")

        method_results = []
        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            np.random.seed(RANDOM_SEED + run)

            user_results = []
            for user_id in user_ids:
                try:
                    metrics = run_inprocessing_baseline_experiment(
                        candidate_items=candidate_items,
                        train_history=user_train_histories[user_id],
                        test_ground_truth=user_test_ground_truths[user_id],
                        item_features=item_features,
                        weights=baseline_cfg['weights'],
                        penalty_lambda=baseline_cfg['penalty_lambda'],
                        experiment_name=f"{method_name}_run{run+1}_{user_id}",
                        population_size=100,
                        max_generations=50,
                        recommendation_size=10,
                        fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
                        seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
                        new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
                    )
                    # [修复 In-processing 集成]
                    if not metrics:
                        logger.error(f"  User {user_id} returned empty metrics for {method_name}")
                        continue
                    metrics['_user_id'] = user_id
                    metrics['weights'] = {
                        'w1': baseline_cfg['weights'][0],
                        'w2': baseline_cfg['weights'][1],
                        'w3': baseline_cfg['weights'][2],
                    }
                    metrics['penalty_lambda'] = baseline_cfg['penalty_lambda']
                    metrics['weight_combo'] = baseline_cfg['weight_combo']
                    metrics['penalty_level'] = baseline_cfg['penalty_level']
                    metrics['baseline_type'] = 'inprocessing_weighted_sum'
                    user_results.append(metrics)
                except Exception as e:
                    logger.error(f"  User {user_id} failed: {e}")
                    continue

            if user_results:
                method_results.append(aggregate_user_metrics(user_results, method_name, run))

        results[method_name] = method_results

        if method_results:
            avg_hv = np.mean([r.get('hypervolume', 0) for r in method_results])
            avg_acc = np.mean([r.get('avg_accuracy', 0) for r in method_results])
            avg_div = np.mean([r.get('avg_diversity', 0) for r in method_results])
            avg_real_ndcg = np.mean([r.get('real_ndcg@10', 0) for r in method_results])
            avg_real_hr = np.mean([r.get('real_hr@10', 0) for r in method_results])
            avg_feasibility = np.mean([r.get('feasibility_rate', 0) for r in method_results])
            logger.info(
                f"  Avg HV: {avg_hv:.4f}, ProxyAcc: {avg_acc:.4f}, Div: {avg_div:.4f}, "
                f"RealNDCG@10: {avg_real_ndcg:.4f}, RealHR@10: {avg_real_hr:.4f}, "
                f"Feas: {avg_feasibility:.2%}"
            )

    # 3) Post-processing greedy reranking baseline
    for method_name, baseline_cfg in postprocessing_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*50}")

        method_results = []
        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            np.random.seed(RANDOM_SEED + run)

            user_results = []
            for user_id in user_ids:
                try:
                    metrics = run_greedy_reranking_baseline_experiment(
                        candidate_items=candidate_items,
                        train_history=user_train_histories[user_id],
                        test_ground_truth=user_test_ground_truths[user_id],
                        item_features=item_features,
                        experiment_name=f"{method_name}_run{run+1}_{user_id}",
                        candidate_pool_size=baseline_cfg['candidate_pool_size'],
                        recommendation_size=baseline_cfg['recommendation_size'],
                        fairness_threshold=MAIN_FAIRNESS_THRESHOLD,
                        seller_coverage_threshold=MAIN_SELLER_COVERAGE_THRESHOLD,
                        new_item_threshold=MAIN_NEW_ITEM_THRESHOLD,
                    )
                    metrics['_user_id'] = user_id
                    metrics['baseline_type'] = 'postprocessing_greedy_reranking'
                    user_results.append(metrics)
                except Exception as e:
                    logger.error(f"  User {user_id} failed: {e}")
                    continue

            if user_results:
                method_results.append(aggregate_user_metrics(user_results, method_name, run))

        results[method_name] = method_results

        if method_results:
            avg_hv = np.mean([r.get('hypervolume', 0) for r in method_results])
            avg_acc = np.mean([r.get('avg_accuracy', 0) for r in method_results])
            avg_div = np.mean([r.get('avg_diversity', 0) for r in method_results])
            avg_real_ndcg = np.mean([r.get('real_ndcg@10', 0) for r in method_results])
            avg_real_hr = np.mean([r.get('real_hr@10', 0) for r in method_results])
            avg_feasibility = np.mean([r.get('feasibility_rate', 0) for r in method_results])
            logger.info(
                f"  Avg HV: {avg_hv:.4f}, ProxyAcc: {avg_acc:.4f}, Div: {avg_div:.4f}, "
                f"RealNDCG@10: {avg_real_ndcg:.4f}, RealHR@10: {avg_real_hr:.4f}, "
                f"Feas: {avg_feasibility:.2%}"
            )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'main_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


def run_ablation_study(
    candidate_items: List[str],
    user_train_histories: Dict[str, List[Dict]],
    user_test_ground_truths: Dict[str, List[Dict]],
    item_features: Dict[str, Dict],
    user_profiles: Dict[str, Dict],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run comprehensive ablation study.

    Studies:
    1. Population size effect
    2. Constraint threshold effect
    3. Mutation rate effect
    4. Generation count effect
    """
    logger.info("\n" + "="*50)
    logger.info("Running Ablation Study")
    logger.info("="*50)

    # Select one user for ablation study (faster)
    user_id = list(user_train_histories.keys())[0]
    user_history = user_train_histories[user_id]
    user_test_ground_truth = user_test_ground_truths[user_id]
    user_profile = user_profiles[user_id]

    ablation_configs = {
        # Population size study
        'Pop_50': DualAgentConfig(population_size=50, max_generations=30, use_llm=False),
        'Pop_100': DualAgentConfig(population_size=100, max_generations=30, use_llm=False),
        'Pop_200': DualAgentConfig(population_size=200, max_generations=30, use_llm=False),

        # Mutation rate study
        'Mutation_0.05': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.05, use_llm=False),
        'Mutation_0.1': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.1, use_llm=False),
        'Mutation_0.2': DualAgentConfig(population_size=100, max_generations=30, mutation_rate=0.2, use_llm=False),

        # Constraint threshold study
        'Strict_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.8, seller_coverage_threshold=0.4, new_item_threshold=0.2
        ),
        'Normal_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.6, seller_coverage_threshold=0.2, new_item_threshold=0.1
        ),
        'Relaxed_Constraints': DualAgentConfig(
            population_size=100, max_generations=30, use_llm=False,
            fairness_threshold=0.4, seller_coverage_threshold=0.1, new_item_threshold=0.05
        ),

        # Generation count study
        'Gen_20': DualAgentConfig(population_size=100, max_generations=20, use_llm=False),
        'Gen_50': DualAgentConfig(population_size=100, max_generations=50, use_llm=False),
        'Gen_100': DualAgentConfig(population_size=100, max_generations=100, use_llm=False),
    }

    results = {}
    for name, config in ablation_configs.items():
        logger.info(f"Running ablation: {name}")
        try:
            metrics = run_single_experiment(
                config=config,
                candidate_items=candidate_items,
                train_history=user_history,
                test_ground_truth=user_test_ground_truth,
                item_features=item_features,
                user_profile=user_profile,
                experiment_name=name
            )
            results[name] = metrics
            logger.info(f"  HV: {metrics.get('hypervolume', 0):.4f}, Acc: {metrics.get('avg_accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"Ablation {name} failed: {e}")

    # Save results
    output_path = os.path.join(output_dir, 'ablation_study.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Ablation results saved to {output_path}")

    return results


def generate_results_table(results: Dict[str, List[Dict]]) -> str:
    """Generate LaTeX-ready results table."""
    table = "\\begin{table}[h]\n\\centering\n"
    table += "\\caption{Main comparison results on Amazon dataset}\n"
    table += "\\label{tab:main_results}\n"
    table += "\\begin{tabular}{lcccc}\n\\toprule\n"
    table += "Method & HV $\\uparrow$ & Accuracy $\\uparrow$ & Diversity $\\uparrow$ & Feasibility $\\uparrow$ \\\\ \\midrule\n"

    for method, runs in results.items():
        if not runs:
            continue
        avg_hv = np.mean([r.get('hypervolume', 0) for r in runs])
        std_hv = np.std([r.get('hypervolume', 0) for r in runs])
        avg_acc = np.mean([r.get('avg_accuracy', 0) for r in runs])
        avg_div = np.mean([r.get('avg_diversity', 0) for r in runs])
        avg_feas = np.mean([r.get('feasibility_rate', 0) for r in runs])

        method_display = method.replace('_', ' ')
        if method == 'DualAgent-Rec':
            method_display = "\\textbf{DualAgent-Rec (Ours)}"

        table += f"{method_display} & "
        table += f"{avg_hv:.4f}$\\pm${std_hv:.4f} & "
        table += f"{avg_acc:.4f} & "
        table += f"{avg_div:.4f} & "
        table += f"{avg_feas:.2%} \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\end{table}"

    return table


def generate_ablation_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for ablation study."""
    table = "\\begin{table}[h]\n\\centering\n"
    table += "\\caption{Ablation study results}\n"
    table += "\\label{tab:ablation}\n"
    table += "\\begin{tabular}{lccccc}\n\\toprule\n"
    table += "Setting & HV & Accuracy & Diversity & Novelty & Runtime(s) \\\\ \\midrule\n"

    for name, metrics in results.items():
        if not metrics:
            continue
        table += f"{name.replace('_', ' ')} & "
        table += f"{metrics.get('hypervolume', 0):.4f} & "
        table += f"{metrics.get('avg_accuracy', 0):.4f} & "
        table += f"{metrics.get('avg_diversity', 0):.4f} & "
        table += f"{metrics.get('avg_novelty', 0):.4f} & "
        table += f"{metrics.get('runtime_seconds', 0):.1f} \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\end{table}"

    return table


def plot_paper_figures(results_json_path: str) -> Dict[str, str]:
    """
    Plot paper figures from main_comparison.json.

    Figures:
    1) Trade-off scatter (NDCG vs Diversity)
    2) Grouped bar (Proxy Accuracy & Diversity)
    3) Grouped bar (HV & Feasibility)
    4) 4D performance radar chart
    5) DualAgent convergence subplots (Feasibility & HV)
    6) Real NDCG@10 bar chart (5 methods)
    """
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open(results_json_path, 'r') as f:
        results = json.load(f)

    output_dir = os.path.dirname(results_json_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style='whitegrid', palette='colorblind')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    figure_paths: Dict[str, str] = {}

    palette = sns.color_palette('colorblind', n_colors=8)
    method_colors = {
        'DualAgent-Rec': palette[0],
        'In-processing (w_B, strong)': palette[1],
        'Greedy_Reranking': palette[2],
        'Single_Population': palette[3],
        'w/o_Constraints': palette[4],
        'In-processing_A': palette[5],
        'In-processing_C': palette[6],
    }
    method_labels = {
        'DualAgent-Rec': 'DualAgent-Rec',
        'In-processing (w_B, strong)': 'In-processing (B, λ=100)',
        'Greedy_Reranking': 'Greedy Re-ranking',
        'Single_Population': 'Single Population',
        'w/o_Constraints': 'w/o Constraints',
    }

    legacy_key_map = {
        'In-processing (w_A, weak)': ['In-processing (w_A, weak)', 'InProc_A_lambda0.1'],
        'In-processing (w_A, strong)': ['In-processing (w_A, strong)', 'InProc_A_lambda100'],
        'In-processing (w_B, weak)': ['In-processing (w_B, weak)', 'InProc_B_lambda0.1'],
        'In-processing (w_B, strong)': ['In-processing (w_B, strong)', 'InProc_B_lambda100'],
        'In-processing (w_C, weak)': ['In-processing (w_C, weak)', 'InProc_C_lambda0.1'],
        'In-processing (w_C, strong)': ['In-processing (w_C, strong)', 'InProc_C_lambda100'],
    }

    def _runs(method_name: str) -> List[Dict[str, Any]]:
        candidate_names = legacy_key_map.get(method_name, [method_name])
        for name in candidate_names:
            runs = results.get(name, [])
            if isinstance(runs, list) and runs:
                return runs
        # If no candidate has non-empty runs, still return any existing empty list.
        for name in candidate_names:
            runs = results.get(name, [])
            if isinstance(runs, list):
                return runs
        return []

    def _mean_metric(method_name: str, metric_name: str) -> Any:
        runs = _runs(method_name)
        if not runs:
            return None
        values = [r.get(metric_name) for r in runs if isinstance(r.get(metric_name), (int, float))]
        if not values:
            return None
        return float(np.mean(values))

    def _extract_frontier_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not points:
            return []
        frontier = []
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i == j:
                    continue
                if (
                    other[0] >= point[0]
                    and other[1] >= point[1]
                    and (other[0] > point[0] or other[1] > point[1])
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(point)
        frontier = sorted(list(set(frontier)), key=lambda x: x[0])
        return frontier

    def _get_combo_point(combo: str) -> Any:
        combo_methods = {
            'A': ['In-processing (w_A, weak)', 'In-processing (w_A, strong)'],
            'B': ['In-processing (w_B, weak)', 'In-processing (w_B, strong)'],
            'C': ['In-processing (w_C, weak)', 'In-processing (w_C, strong)'],
        }
        ndcg_vals = []
        div_vals = []
        for method in combo_methods[combo]:
            for run in _runs(method):
                ndcg = run.get('real_ndcg@10')
                div = run.get('avg_diversity')
                if isinstance(ndcg, (int, float)) and isinstance(div, (int, float)):
                    ndcg_vals.append(float(ndcg))
                    div_vals.append(float(div))
        if not ndcg_vals or not div_vals:
            return None
        return (float(np.mean(div_vals)), float(np.mean(ndcg_vals)))

    def _annotate_na_point(ax: Any, x: float, y: float, marker: str, label: str) -> None:
        ax.scatter([x], [y], s=90, marker=marker, facecolors='none', edgecolors='#808080', linewidths=1.6, label=label)
        ax.text(x + 0.004, y + 0.0015, 'N/A', color='#666666', fontsize=9)

    # Figure 1: Trade-off scatter + real DualAgent Pareto frontier
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.grid(True, color='#E5E5E5', linestyle='--', linewidth=0.8, alpha=0.95)

    dual_points: List[Tuple[float, float]] = []
    for run in _runs('DualAgent-Rec'):
        for point in run.get('pareto_solutions', []):
            ndcg = point.get('real_ndcg@10')
            div = point.get('avg_diversity')
            if isinstance(ndcg, (int, float)) and isinstance(div, (int, float)):
                dual_points.append((float(div), float(ndcg)))
    dual_frontier = _extract_frontier_2d(dual_points)

    plotted_x = []
    plotted_y = []
    if dual_points:
        x_all = [p[0] for p in dual_points]
        y_all = [p[1] for p in dual_points]
        plotted_x.extend(x_all)
        plotted_y.extend(y_all)
        ax.scatter(x_all, y_all, s=20, alpha=0.18, color=method_colors['DualAgent-Rec'], label='DualAgent Pareto Points')
    if dual_frontier:
        x_pf = [p[0] for p in dual_frontier]
        y_pf = [p[1] for p in dual_frontier]
        plotted_x.extend(x_pf)
        plotted_y.extend(y_pf)
        ax.plot(
            x_pf, y_pf, '--o',
            color=method_colors['DualAgent-Rec'],
            linewidth=3.0,
            markersize=5,
            label='DualAgent Pareto Frontier'
        )
        balanced_point = max(dual_frontier, key=lambda p: p[0] + p[1])
        ax.scatter(
            [balanced_point[0]],
            [balanced_point[1]],
            s=260,
            marker='*',
            color=method_colors['DualAgent-Rec'],
            edgecolors='black',
            linewidths=0.7,
            zorder=8,
            label='DualAgent Balanced Point',
        )
    else:
        _annotate_na_point(ax, 0.12, 0.055, '*', 'DualAgent Balanced Point (N/A)')

    combo_styles = {
        'A': ('D', method_colors['In-processing_A']),
        'B': ('s', method_colors['In-processing (w_B, strong)']),
        'C': ('^', method_colors['In-processing_C']),
    }
    combo_na_positions = {
        'A': (0.105, 0.056),
        'B': (0.105, 0.052),
        'C': (0.105, 0.048),
    }
    for combo in ['A', 'B', 'C']:
        point = _get_combo_point(combo)
        marker, color = combo_styles[combo]
        if point is None:
            nx, ny = combo_na_positions[combo]
            _annotate_na_point(ax, nx, ny, marker, f'InProc-{combo} (N/A)')
            continue
        x, y = point
        plotted_x.append(x)
        plotted_y.append(y)
        ax.scatter([x], [y], s=110, marker=marker, color=color, edgecolors='black', linewidths=0.4, label=f'InProc-{combo}')

    method_markers = {
        'Greedy_Reranking': 'P',
        'Single_Population': 'X',
        'w/o_Constraints': 'o',
    }
    na_seed_positions = {
        'Greedy_Reranking': (0.12, 0.044),
        'Single_Population': (0.12, 0.040),
        'w/o_Constraints': (0.12, 0.036),
    }
    for method in ['Greedy_Reranking', 'Single_Population', 'w/o_Constraints']:
        ndcg = _mean_metric(method, 'real_ndcg@10')
        div = _mean_metric(method, 'avg_diversity')
        if ndcg is None or div is None:
            nx, ny = na_seed_positions[method]
            _annotate_na_point(ax, nx, ny, method_markers[method], f"{method_labels.get(method, method)} (N/A)")
            continue
        plotted_x.append(div)
        plotted_y.append(ndcg)
        ax.scatter(
            [div], [ndcg],
            s=130,
            marker=method_markers[method],
            color=method_colors.get(method, '#555555'),
            edgecolors='black',
            linewidths=0.4,
            label=method_labels.get(method, method),
        )

    if plotted_x and plotted_y:
        ax.set_xlim(min(0.1, min(plotted_x) - 0.02), max(0.4, max(plotted_x) + 0.02))
        ax.set_ylim(max(0.0, min(plotted_y) - 0.01), max(0.06, max(plotted_y) + 0.01))
    else:
        ax.set_xlim(0.1, 0.4)
        ax.set_ylim(0.0, 0.06)

    ax.set_xlabel('Intra-list Diversity')
    ax.set_ylabel('real_ndcg@10')
    ax.set_title('Figure 1. Trade-off Scatter Plot (NDCG vs Diversity)')
    ax.legend(loc='best', frameon=True)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'tradeoff_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths['tradeoff_scatter'] = scatter_path

    fig2_methods = [
        'DualAgent-Rec',
        'In-processing (w_B, strong)',
        'Greedy_Reranking',
        'Single_Population',
        'w/o_Constraints',
    ]
    fig2_groups = [
        # [Task 3: 新增真实NDCG图]
        # Figure 2 回归代理指标，真实离线指标单独放到 Figure 6。
        ('Accuracy (Proxy avg_accuracy)', 'avg_accuracy'),
        ('Diversity (Intra-list Diversity)', 'avg_diversity'),
    ]

    # Figure 2: Grouped Bar (Accuracy & Diversity)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(True, axis='y', color='#E5E5E5', linestyle='--', linewidth=0.8, alpha=0.95)

    group_x = np.arange(len(fig2_groups))
    width = 0.14
    all_vals = []
    for i, method in enumerate(fig2_methods):
        offset = (i - (len(fig2_methods) - 1) / 2) * width
        values = []
        missing_flags = []
        for _, metric in fig2_groups:
            v = _mean_metric(method, metric)
            values.append(0.0 if v is None else float(v))
            missing_flags.append(v is None)
            if v is not None:
                all_vals.append(float(v))

        bars = ax.bar(
            group_x + offset,
            values,
            width=width,
            color=method_colors[method],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.92,
            label=method_labels[method],
        )
        for j, bar in enumerate(bars):
            if missing_flags[j]:
                bar.set_facecolor('white')
                bar.set_edgecolor('#808080')
                bar.set_hatch('///')
            h = bar.get_height()
            if missing_flags[j]:
                ax.text(bar.get_x() + bar.get_width() / 2, max(0.002, h + 0.001), 'N/A', ha='center', va='bottom', fontsize=9, color='#666666')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f'{h:.4f}', ha='center', va='bottom', fontsize=9)

    y_max = max(all_vals) * 1.20 if all_vals else 0.1
    ax.set_ylim(0.0, max(y_max, 0.08))
    ax.set_xticks(group_x)
    ax.set_xticklabels([g[0] for g in fig2_groups])
    ax.set_ylabel('Metric Value')
    ax.set_title('Figure 2. Grouped Bar Chart (Accuracy & Diversity)')
    ax.legend(loc='upper center', ncol=3, frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'acc_div_grouped_bar.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths['acc_div_grouped_bar'] = fig2_path

    # Figure 3: Grouped Bar (HV & Feasibility)
    fig3_groups = [
        ('Hypervolume (HV)', 'hypervolume', 'float'),
        ('Feasibility Rate (%)', 'feasibility_rate', 'pct'),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.grid(True, axis='y', color='#E5E5E5', linestyle='--', linewidth=0.8, alpha=0.95)

    group_x = np.arange(len(fig3_groups))
    width = 0.14
    all_vals = []
    for i, method in enumerate(fig2_methods):
        offset = (i - (len(fig2_methods) - 1) / 2) * width
        values = []
        missing_flags = []
        for _, metric, _ in fig3_groups:
            v = _mean_metric(method, metric)
            values.append(0.0 if v is None else float(v))
            missing_flags.append(v is None)
            if v is not None:
                all_vals.append(float(v))

        bars = ax.bar(
            group_x + offset,
            values,
            width=width,
            color=method_colors[method],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.92,
            label=method_labels[method],
        )
        for j, bar in enumerate(bars):
            _, _, value_type = fig3_groups[j]
            if missing_flags[j]:
                bar.set_facecolor('white')
                bar.set_edgecolor('#808080')
                bar.set_hatch('///')
                ax.text(bar.get_x() + bar.get_width() / 2, max(0.01, bar.get_height() + 0.005), 'N/A', ha='center', va='bottom', fontsize=9, color='#666666')
                continue
            h = bar.get_height()
            label_text = f'{h:.4f}' if value_type == 'float' else f'{h * 100:.2f}%'
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, label_text, ha='center', va='bottom', fontsize=9)

    y_max = max(all_vals) * 1.20 if all_vals else 1.0
    ax.set_ylim(0.0, max(y_max, 1.05))
    ax.set_xticks(group_x)
    ax.set_xticklabels([g[0] for g in fig3_groups])
    ax.set_ylabel('Metric Value')
    ax.set_title('Figure 3. Grouped Bar Chart (HV & Feasibility)')
    ax.legend(loc='upper center', ncol=3, frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'hv_feasibility_grouped_bar.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths['hv_feasibility_grouped_bar'] = fig3_path

    # Figure 4: Performance radar chart (normalized)
    radar_metrics = ['real_ndcg@10', 'avg_diversity', 'avg_novelty', 'feasibility_rate']
    radar_labels = ['Real NDCG@10', 'Intra-list Diversity', 'List Novelty', 'Overall Feasibility Rate']
    radar_methods = ['DualAgent-Rec', 'In-processing (w_B, strong)', 'Greedy_Reranking']
    radar_display = {
        'DualAgent-Rec': 'DualAgent-Rec',
        'In-processing (w_B, strong)': 'In-processing (B, λ=100)',
        'Greedy_Reranking': 'Greedy Re-ranking',
    }

    raw_radar: Dict[str, List[Any]] = {}
    for method in radar_methods:
        raw_radar[method] = [_mean_metric(method, metric) for metric in radar_metrics]

    # 按“指标列”做相对 Min-Max 归一化：
    # scaled = 0.1 + 0.9 * ((value - min_val) / (max_val - min_val + 1e-8))
    # 映射到 [0.1, 1.0]，避免最差方法在某轴退化到圆心(0)。
    normalized_radar: Dict[str, List[Any]] = {method: [None] * len(radar_metrics) for method in radar_methods}
    for dim_idx, _ in enumerate(radar_metrics):
        dim_values = [raw_radar[m][dim_idx] for m in radar_methods if raw_radar[m][dim_idx] is not None]
        if not dim_values:
            continue
        min_val = float(min(dim_values))
        max_val = float(max(dim_values))
        denom = max_val - min_val + 1e-8
        for m in radar_methods:
            v = raw_radar[m][dim_idx]
            if v is None:
                continue
            scaled_value = 0.1 + 0.9 * ((float(v) - min_val) / denom)
            normalized_radar[m][dim_idx] = float(max(0.1, min(1.0, scaled_value)))

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    # 使用长方形画布并将雷达图放大居中，改善整体观感。
    fig, ax = plt.subplots(figsize=(13.2, 7.4), subplot_kw=dict(polar=True))
    ax.set_position([0.17, 0.16, 0.66, 0.68])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(True, color='#D8D8D8', linestyle='--', linewidth=0.8, alpha=0.65)
    ax.spines['polar'].set_color('#BDBDBD')
    ax.spines['polar'].set_linewidth(1.2)

    plot_order = ['In-processing (w_B, strong)', 'Greedy_Reranking', 'DualAgent-Rec']
    for method in plot_order:
        vals = normalized_radar[method]
        if not any(v is not None for v in vals):
            placeholder = [0.15] * len(radar_labels)
            vals_closed = placeholder + placeholder[:1]
            ax.plot(angles, vals_closed, linestyle='--', linewidth=1.8, color='#808080', label=f"{radar_display[method]} (N/A)")
            ax.fill(angles, vals_closed, alpha=0.06, color='#B0B0B0')
            continue

        numeric_vals = [0.0 if v is None else float(v) for v in vals]
        vals_closed = numeric_vals + numeric_vals[:1]
        color = method_colors[method]
        line_width = 2.8 if method == 'DualAgent-Rec' else 2.0
        z_order = 5 if method == 'DualAgent-Rec' else 3
        fill_alpha = 0.25 if method == 'DualAgent-Rec' else 0.18
        ax.plot(
            angles,
            vals_closed,
            linewidth=line_width,
            color=color,
            marker='o',
            markersize=4,
            label=radar_display[method],
            zorder=z_order
        )
        ax.fill(angles, vals_closed, alpha=fill_alpha, color=color, zorder=z_order - 1)

    display_radar_labels = ['Real NDCG@10', 'Intra-list\nDiversity', 'List\nNovelty', 'Overall\nFeasibility Rate']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_radar_labels)
    ax.tick_params(axis='x', pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    # 雷达图形状表达“相对强弱”，内部网格保留但不显示具体数字。
    ax.set_yticklabels([])
    if not any(_runs('In-processing (w_B, strong)')):
        ax.text(0.5, 0.12, 'N/A: In-processing (B, λ=100) missing', transform=ax.transAxes, ha='center', fontsize=9, color='#666666')
    fig.suptitle('Figure 4. Performance Radar Chart (Normalized)', y=0.97, fontsize=19)
    # 图例下置并横向排列，避免与轴标签冲突。
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True, edgecolor='#B5B5B5')
    radar_path = os.path.join(output_dir, 'performance_radar.png')
    plt.savefig(radar_path, dpi=300)
    plt.close()
    figure_paths['performance_radar'] = radar_path

    # Figure 5: DualAgent convergence subplots
    generations: List[int] = []
    feasibility_series: List[float] = []
    hv_series: List[float] = []

    # Backward-compatible parsing if generation-level history exists.
    gen_feas_map: Dict[int, List[float]] = {}
    gen_hv_map: Dict[int, List[float]] = {}
    for run in _runs('DualAgent-Rec'):
        history = run.get('history') or run.get('generation_history') or []
        if not isinstance(history, list):
            continue
        for row in history:
            if not isinstance(row, dict):
                continue
            gen = row.get('generation')
            if not isinstance(gen, int):
                continue
            feas = row.get('feasibility_rate')
            if feas is None and isinstance(row.get('constraint_metrics'), dict):
                feas = row['constraint_metrics'].get('overall_feasibility')
            hv = row.get('hypervolume')
            if isinstance(feas, (int, float)):
                gen_feas_map.setdefault(gen, []).append(float(feas))
            if isinstance(hv, (int, float)):
                gen_hv_map.setdefault(gen, []).append(float(hv))

    if gen_feas_map:
        generations = sorted(gen_feas_map.keys())
        feasibility_series = [float(np.mean(gen_feas_map[g])) for g in generations]
    if gen_hv_map:
        if not generations:
            generations = sorted(gen_hv_map.keys())
        hv_series = [float(np.mean(gen_hv_map.get(g, [0.0]))) for g in generations]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    panel_titles = [
        'Panel A: Feasibility Rate Convergence',
        'Panel B: Hypervolume Convergence',
    ]
    for ax, title in zip(axes, panel_titles):
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.grid(True, color='#E5E5E5', linestyle='--', linewidth=0.8, alpha=0.95)
        ax.set_xlim(0, 50)

    if generations and feasibility_series:
        axes[0].plot(generations, feasibility_series, '-o', color=method_colors['DualAgent-Rec'], markersize=4, linewidth=2.0)
        axes[0].set_ylabel('Feasibility Rate')
        axes[0].set_ylim(0.0, 1.05)
    else:
        axes[0].set_ylabel('Feasibility Rate')
        axes[0].set_ylim(0.0, 1.05)
        axes[0].text(
            0.5, 0.5,
            'N/A: generation-level history missing in main_comparison.json',
            transform=axes[0].transAxes,
            ha='center',
            va='center',
            fontsize=10,
            color='#666666',
        )

    if generations and hv_series:
        axes[1].plot(generations, hv_series, '-o', color=method_colors['DualAgent-Rec'], markersize=4, linewidth=2.0)
        axes[1].set_ylabel('Hypervolume (HV)')
        hv_max = max(hv_series) if hv_series else 0.1
        axes[1].set_ylim(0.0, max(0.1, hv_max * 1.15))
    else:
        axes[1].set_ylabel('Hypervolume (HV)')
        axes[1].set_ylim(0.0, 1.0)
        axes[1].text(
            0.5, 0.5,
            'N/A: generation-level history missing in main_comparison.json',
            transform=axes[1].transAxes,
            ha='center',
            va='center',
            fontsize=10,
            color='#666666',
        )

    fig.suptitle('Figure 5. DualAgent Convergence Subplots', y=1.02)
    plt.tight_layout()
    conv_path = os.path.join(output_dir, 'dualagent_convergence_subplots.png')
    plt.savefig(conv_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths['dualagent_convergence_subplots'] = conv_path

    # [Task 3: 新增真实NDCG图]
    # Figure 6: Dedicated bar chart for real offline NDCG@10 (5 methods).
    fig6_methods = [
        'DualAgent-Rec',
        'In-processing (w_B, strong)',
        'Greedy_Reranking',
        'Single_Population',
        'w/o_Constraints',
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.grid(True, axis='y', color='#E5E5E5', linestyle='--', linewidth=0.8, alpha=0.95)

    x = np.arange(len(fig6_methods))
    ndcg_vals: List[float] = []
    missing_flags: List[bool] = []
    for method in fig6_methods:
        v = _mean_metric(method, 'real_ndcg@10')
        if v is None:
            ndcg_vals.append(0.0)
            missing_flags.append(True)
        else:
            ndcg_vals.append(float(v))
            missing_flags.append(False)

    bars = ax.bar(
        x,
        ndcg_vals,
        width=0.62,
        color=[method_colors[m] for m in fig6_methods],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.92,
    )
    for i, bar in enumerate(bars):
        if missing_flags[i]:
            bar.set_facecolor('white')
            bar.set_edgecolor('#808080')
            bar.set_hatch('///')
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(0.0015, bar.get_height() + 0.001),
                'N/A',
                ha='center',
                va='bottom',
                fontsize=9,
                color='#666666',
            )
            continue
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.001,
            f'{h:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    valid_vals = [v for v, m in zip(ndcg_vals, missing_flags) if not m]
    y_max = max(valid_vals) * 1.20 if valid_vals else 0.08
    ax.set_ylim(0.0, max(0.08, y_max))
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in fig6_methods], rotation=12, ha='right')
    ax.set_ylabel('real_ndcg@10')
    ax.set_title('Figure 6. Real Offline NDCG@10 Comparison')
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, 'real_ndcg_bar.png')
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    figure_paths['real_ndcg_bar'] = fig6_path

    logger.info(f"Paper figures saved to: {output_dir}")
    return figure_paths


def main():
    parser = argparse.ArgumentParser(description='Run DualAgent-Rec experiments')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='../experiments/results', help='Output directory')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['All_Beauty'],
                       help='Amazon dataset categories (can specify multiple)')
    parser.add_argument('--n_users', type=int, default=20, help='Number of users per category')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM coordinator')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per method')
    parser.add_argument('--max_reviews', type=int, default=20000, help='Max reviews to load')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'main', 'ablation', 'quick'],
                       help='Which experiments to run')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load real Amazon data from multiple categories
    all_candidate_items = []
    all_user_train_histories = {}
    all_user_test_ground_truths = {}
    all_item_features = {}
    all_user_profiles = {}

    for category in args.categories:
        logger.info(f"\n{'='*50}")
        logger.info(f"Loading category: {category}")
        logger.info(f"{'='*50}")
        try:
            candidate_items, user_train_histories, user_test_ground_truths, item_features, user_profiles = load_amazon_data(
                args.data_dir,
                category=category,
                max_reviews=args.max_reviews,
                n_users=args.n_users
            )
            # Merge data from this category (with consistent prefixing)
            all_candidate_items.extend([f"{category}_{item_id}" for item_id in candidate_items])
            for user_id, history in user_train_histories.items():
                # Update item_ids in history to have prefix
                prefixed_history = []
                for h in history:
                    h_copy = h.copy()
                    if 'item_id' in h_copy:
                        h_copy['item_id'] = f"{category}_{h_copy['item_id']}"
                    prefixed_history.append(h_copy)
                all_user_train_histories[f"{category}_{user_id}"] = prefixed_history
            for user_id, test_history in user_test_ground_truths.items():
                prefixed_test_history = []
                for h in test_history:
                    h_copy = h.copy()
                    if 'item_id' in h_copy:
                        h_copy['item_id'] = f"{category}_{h_copy['item_id']}"
                    prefixed_test_history.append(h_copy)
                all_user_test_ground_truths[f"{category}_{user_id}"] = prefixed_test_history
            for item_id, features in item_features.items():
                features['source_category'] = category
                all_item_features[f"{category}_{item_id}"] = features
            for user_id, profile in user_profiles.items():
                all_user_profiles[f"{category}_{user_id}"] = profile
            logger.info(f"Loaded {len(candidate_items)} items, {len(user_train_histories)} users from {category}")
        except Exception as e:
            logger.error(f"Failed to load {category}: {e}")
            continue

    # Use merged data
    candidate_items = list(set(all_candidate_items))
    user_train_histories = all_user_train_histories
    user_test_ground_truths = all_user_test_ground_truths
    item_features = all_item_features
    user_profiles = all_user_profiles

    logger.info(
        f"\nTotal data loaded: {len(candidate_items)} items, {len(user_train_histories)} users "
        f"across {len(args.categories)} categories"
    )

    # Run experiments
    if args.experiment in ['all', 'main']:
        results = run_main_comparison(
            candidate_items=candidate_items,
            user_train_histories=user_train_histories,
            user_test_ground_truths=user_test_ground_truths,
            item_features=item_features,
            user_profiles=user_profiles,
            output_dir=args.output_dir,
            use_llm=args.use_llm,
            num_runs=args.num_runs
        )

        # Generate table
        table = generate_results_table(results)
        table_path = os.path.join(args.output_dir, 'results_table.tex')
        with open(table_path, 'w') as f:
            f.write(table)
        logger.info(f"LaTeX table saved to {table_path}")

        # Generate paper figures from the unified main comparison JSON.
        plot_paper_figures(os.path.join(args.output_dir, 'main_comparison.json'))

    if args.experiment in ['all', 'ablation']:
        ablation_results = run_ablation_study(
            candidate_items=candidate_items,
            user_train_histories=user_train_histories,
            user_test_ground_truths=user_test_ground_truths,
            item_features=item_features,
            user_profiles=user_profiles,
            output_dir=args.output_dir
        )

        # Generate ablation table
        ablation_table = generate_ablation_table(ablation_results)
        ablation_table_path = os.path.join(args.output_dir, 'ablation_table.tex')
        with open(ablation_table_path, 'w') as f:
            f.write(ablation_table)
        logger.info(f"Ablation table saved to {ablation_table_path}")

    if args.experiment == 'quick':
        # Quick test run with one user
        logger.info("Running quick test...")
        user_id = list(user_train_histories.keys())[0]

        config = DualAgentConfig(
            population_size=30,
            max_generations=10,
            use_llm=False,
        )
        metrics = run_single_experiment(
            config=config,
            candidate_items=candidate_items,
            train_history=user_train_histories[user_id],
            test_ground_truth=user_test_ground_truths[user_id],
            item_features=item_features,
            user_profile=user_profiles[user_id],
            experiment_name='quick_test'
        )
        print("\n=== Quick Test Results ===")
        print(f"Hypervolume: {metrics.get('hypervolume', 0):.4f}")
        print(f"Accuracy: {metrics.get('avg_accuracy', 0):.4f}")
        print(f"Diversity: {metrics.get('avg_diversity', 0):.4f}")
        print(f"Novelty: {metrics.get('avg_novelty', 0):.4f}")
        print(f"Real NDCG@10: {metrics.get('real_ndcg@10', 0):.4f}")
        print(f"Real HR@10: {metrics.get('real_hr@10', 0):.4f}")
        print(f"Pareto size: {metrics.get('pareto_size', 0)}")
        print(f"Feasibility: {metrics.get('feasibility_rate', 0):.2%}")
        print(f"Runtime: {metrics.get('runtime_seconds', 0):.2f}s")

    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
