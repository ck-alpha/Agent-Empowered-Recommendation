"""
DualAgent-Rec: LLM-Coordinated Dual-Agent Framework for
Constrained Multi-Objective E-commerce Recommendation

Main framework integrating all components.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from agents import ExploitationAgent, ExplorationAgent, Individual
from constraints import ConstraintHandler, ConstraintConfig, AdaptiveConstraintHandler
from evaluation import ObjectivesCalculator, RecommendationMetrics, MultiObjectiveMetrics
from llm_coordinator import LLMCoordinator, CoordinatorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class DualAgentConfig:
    """Configuration for DualAgent-Rec framework."""
    # Population settings
    population_size: int = 100
    max_generations: int = 200
    recommendation_size: int = 10

    # Agent settings
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    # 修改点：支持真实单种群基线开关，开启后仅保留 exploitation 分支。
    is_single_population: bool = False

    # Constraint settings
    fairness_threshold: float = 0.7
    seller_coverage_threshold: float = 0.3
    new_item_threshold: float = 0.1

    # LLM settings
    use_llm: bool = True
    llm_model: str = 'qwen2.5:72b'
    llm_update_frequency: int = 10

    # Output settings
    output_dir: str = './results'
    save_history: bool = True


class DualAgentRec:
    """
    DualAgent-Rec Framework.

    Combines:
    1. Exploitation Agent (accuracy-focused)
    2. Exploration Agent (diversity-focused)
    3. LLM Coordinator (resource allocation)
    4. Adaptive Constraint Handler
    """

    def __init__(self, config: Optional[DualAgentConfig] = None):
        """
        Initialize DualAgent-Rec.

        Args:
            config: Framework configuration
        """
        self.config = config or DualAgentConfig()

        # Initialize components
        self._init_agents()
        self._init_coordinator()
        self._init_constraint_handler()
        self._init_objectives_calculator()

        # State
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
        self.best_solutions: List[Individual] = []
        # [新增监控指标: generation monitor]
        self._latest_generation_monitor: Dict[str, Any] = {}

    def _init_agents(self):
        """Initialize dual agents."""
        # 双智能体职责拆分：
        # exploitation 负责在可行域内“做精”；exploration 负责在更广空间“做广”。
        self.exploitation_agent = ExploitationAgent(
            population_size=self.config.population_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate
        )

        self.exploration_agent = ExplorationAgent(
            population_size=self.config.population_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate * 2  # 探索侧提高变异率，增加新区域发现概率
        )

    def _init_coordinator(self):
        """Initialize LLM coordinator."""
        coord_config = CoordinatorConfig(
            model_name=self.config.llm_model,
            update_frequency=self.config.llm_update_frequency,
            use_llm=self.config.use_llm
        )
        self.coordinator = LLMCoordinator(coord_config)

    def _init_constraint_handler(self):
        """Initialize constraint handler."""
        constraint_config = ConstraintConfig(
            fairness_threshold=self.config.fairness_threshold,
            seller_coverage_threshold=self.config.seller_coverage_threshold,
            new_item_threshold=self.config.new_item_threshold
        )
        self.constraint_handler = AdaptiveConstraintHandler(constraint_config)

    def _init_objectives_calculator(self):
        """Initialize objectives calculator."""
        self.objectives_calculator = ObjectivesCalculator()

    def _get_active_individuals(self) -> List[Individual]:
        """Get individuals from active optimization branches."""
        # 修改点：单种群模式只统计 exploitation，避免访问未启用的探索分支。
        all_individuals = list(self.exploitation_agent.population)
        if not self.config.is_single_population:
            all_individuals.extend(self.exploration_agent.population)
        return all_individuals

    def optimize(
        self,
        candidate_items: List[str],
        user_history: List[Dict],
        item_features: Dict[str, Dict],
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
        item_popularity: Optional[Dict[str, float]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Individual], Dict[str, Any]]:
        """
        Run multi-objective optimization.

        Args:
            candidate_items: List of candidate item IDs
            user_history: User's interaction history
            item_features: Item ID -> features mapping
            item_embeddings: Optional item embeddings
            item_popularity: Optional item popularity scores
            user_profile: Optional user profile

        Returns:
            Tuple of (Pareto-optimal solutions, optimization metrics)
        """
        logger.info("Starting DualAgent-Rec optimization...")

        # Update objectives calculator
        if item_embeddings:
            self.objectives_calculator.item_embeddings = item_embeddings
        if item_popularity:
            self.objectives_calculator.item_popularity = item_popularity

        # Initialize populations
        self._initialize_populations(candidate_items)

        # 主循环执行“评估 -> 协调 -> 进化 -> 迁移 -> 约束收紧”的闭环。
        for gen in range(self.config.max_generations):
            self.generation = gen

            # Evaluate populations
            self._evaluate_populations(user_history, item_features)

            # Get agent metrics
            exploit_metrics = self.exploitation_agent.get_performance_metrics()
            explore_metrics = (
                {} if self.config.is_single_population
                else self.exploration_agent.get_performance_metrics()
            )

            # Get constraint metrics
            constraint_metrics = self._get_constraint_metrics()
            # [新增监控指标: objective/diversity snapshot]
            generation_monitor = self._compute_generation_population_snapshot()
            invalid_generated_count = 0
            llm_survivors = 0
            de_survivors = 0

            if self.config.is_single_population:
                # 修改点：严格单分支，固定 exploitation 比例并禁用探索进化与知识迁移。
                exploitation_ratio = 1.0
                reasoning = "Single population mode: fixed exploitation_ratio=1.0"
                exploit_children = self.exploitation_agent.evolve(self.config.population_size)
                # [新增监控指标: risk control]
                invalid_generated_count = self._count_invalid_generated_offspring(exploit_children, item_features)
                combined_exploit = self.exploitation_agent.population + exploit_children
                self.exploitation_agent.population = self.exploitation_agent.environmental_selection(combined_exploit)
                llm_survivors, de_survivors = self._count_survivors_by_source()
            else:
                # 协调器输出 α（exploitation 占比），对应论文中的动态资源分配。
                exploitation_ratio, reasoning = self.coordinator.get_resource_allocation(
                    exploitation_metrics=exploit_metrics,
                    exploration_metrics=explore_metrics,
                    constraint_metrics=constraint_metrics,
                    current_generation=gen,
                    max_generations=self.config.max_generations,
                    user_profile=user_profile
                )

                # 根据 α 将总算力/子代预算切分给两个智能体。
                total_offspring = self.config.population_size
                exploit_offspring = int(total_offspring * exploitation_ratio)
                explore_offspring = total_offspring - exploit_offspring

                # Generate offspring
                exploit_children = self.exploitation_agent.evolve(exploit_offspring)
                explore_children = self.exploration_agent.evolve(explore_offspring)

                # 跨种群交叉：把两侧优势解进行知识迁移，避免两条搜索轨迹完全割裂。
                transfer_children = self._cross_population_breeding(
                    self.exploitation_agent.population,
                    self.exploration_agent.population,
                    num_children=min(10, total_offspring // 5)
                )
                # [新增监控指标: source]
                for child in transfer_children:
                    child.source = 'MIXED'

                # [新增监控指标: risk control]
                invalid_generated_count = (
                    self._count_invalid_generated_offspring(exploit_children, item_features)
                    + self._count_invalid_generated_offspring(explore_children, item_features)
                    + self._count_invalid_generated_offspring(transfer_children, item_features)
                )

                # 父代+子代+迁移子代合并后环境选择，形成下一代精英种群。
                combined_exploit = self.exploitation_agent.population + exploit_children + transfer_children
                combined_explore = self.exploration_agent.population + explore_children + transfer_children

                self.exploitation_agent.population = self.exploitation_agent.environmental_selection(combined_exploit)
                self.exploration_agent.population = self.exploration_agent.environmental_selection(combined_explore)
                llm_survivors, de_survivors = self._count_survivors_by_source()

            # [新增监控指标: epsilon]
            current_epsilon = float(self.constraint_handler.epsilon)
            # 依据全局可行率更新 epsilon，实现“早期放宽、后期收紧”的约束策略。
            self.constraint_handler.update_epsilon(constraint_metrics.get('overall_feasibility', 0.5))

            # Update best solutions
            self._update_best_solutions()
            generation_monitor.update({
                'llm_survivors': int(llm_survivors),
                'de_survivors': int(de_survivors),
                'invalid_generated_count': int(invalid_generated_count),
                'current_epsilon': current_epsilon,
            })
            self._latest_generation_monitor = generation_monitor

            # Log progress
            if gen % 10 == 0:
                self._log_progress(gen, exploit_metrics, explore_metrics, constraint_metrics, reasoning)

            # Save history
            if self.config.save_history:
                self._save_generation_history(
                    gen,
                    exploit_metrics,
                    explore_metrics,
                    constraint_metrics,
                    generation_monitor=generation_monitor,
                )

        # [Task 2: 每代记录]
        # 补齐末端代数快照（例如 max_generations=50 时补充 generation=50），
        # 仅用于收敛曲线可视化，不改变进化过程或 LLM 调用频率。
        if self.config.save_history:
            final_exploit_metrics = self.exploitation_agent.get_performance_metrics()
            final_explore_metrics = (
                {} if self.config.is_single_population
                else self.exploration_agent.get_performance_metrics()
            )
            final_constraint_metrics = self._get_constraint_metrics()
            self._append_final_generation_snapshot(
                final_exploit_metrics,
                final_explore_metrics,
                final_constraint_metrics,
            )

        # Final results
        final_metrics = self._compute_final_metrics()
        logger.info(f"Optimization complete. Found {len(self.best_solutions)} Pareto-optimal solutions.")

        return self.best_solutions, final_metrics

    def _initialize_populations(self, candidate_items: List[str]):
        """Initialize both agent populations."""
        self.exploitation_agent.initialize_population(
            candidate_items,
            k=self.config.recommendation_size
        )
        # 修改点：单种群模式不初始化探索侧种群。
        if not self.config.is_single_population:
            self.exploration_agent.initialize_population(
                candidate_items,
                k=self.config.recommendation_size
            )
        else:
            self.exploration_agent.population = []

    def _evaluate_populations(self, user_history: List[Dict], item_features: Dict[str, Dict]):
        """Evaluate both populations."""
        self.exploitation_agent.evaluate_population(
            user_history,
            item_features,
            self.objectives_calculator,
            self.constraint_handler
        )
        # 修改点：单种群模式只评估 exploitation 分支。
        if not self.config.is_single_population:
            self.exploration_agent.evaluate_population(
                user_history,
                item_features,
                self.objectives_calculator,
                self.constraint_handler
            )

    def _get_constraint_metrics(self) -> Dict[str, float]:
        """Get constraint satisfaction metrics from both populations."""
        # 修改点：按 active populations 聚合约束统计，兼容 single/dual 两种模式。
        all_individuals = self._get_active_individuals()

        if not all_individuals:
            return {}

        feasible_count = sum(1 for ind in all_individuals if ind.is_feasible)
        total_count = len(all_individuals)

        avg_violations = np.mean([ind.constraint_violations for ind in all_individuals], axis=0)

        return {
            'overall_feasibility': feasible_count / total_count,
            'fairness_violation': avg_violations[0] if len(avg_violations) > 0 else 0,
            'seller_violation': avg_violations[1] if len(avg_violations) > 1 else 0,
            'new_item_violation': avg_violations[2] if len(avg_violations) > 2 else 0,
        }

    def _cross_population_breeding(
        self,
        pop1: List[Individual],
        pop2: List[Individual],
        num_children: int
    ) -> List[Individual]:
        """
        Cross-population breeding for knowledge transfer.

        Combines good solutions from both populations.
        """
        children = []

        for _ in range(num_children):
            # 一方偏利用、一方偏探索，跨群父母可把“精度基因”和“多样性基因”组合起来。
            parent1 = self.exploitation_agent.tournament_selection(pop1)
            parent2 = self.exploration_agent.tournament_selection(pop2)

            # Crossover
            child1, child2 = self.exploitation_agent.crossover(parent1, parent2)
            child1.source = 'MIXED'
            child2.source = 'MIXED'
            children.extend([child1, child2])

        return children[:num_children]

    def _update_best_solutions(self):
        """Update Pareto-optimal solutions from both populations."""
        all_feasible = [
            ind for ind in
            self._get_active_individuals()
            if ind.is_feasible
        ]

        if not all_feasible:
            # 若暂时无可行解，保留违反量最小的候选作为过渡前沿，避免最优集为空。
            all_individuals = self._get_active_individuals()
            all_feasible = sorted(all_individuals, key=lambda x: x.total_violation)[:10]

        # 使用非支配排序抽取 Pareto 前沿，作为最终候选推荐集。
        fronts = self.exploitation_agent.non_dominated_sort(all_feasible)
        if fronts:
            self.best_solutions = fronts[0][:self.config.population_size]

    def _log_progress(
        self,
        generation: int,
        exploit_metrics: Dict,
        explore_metrics: Dict,
        constraint_metrics: Dict,
        reasoning: str
    ):
        """Log optimization progress."""
        logger.info(
            f"Gen {generation}: "
            f"Feasibility={constraint_metrics.get('overall_feasibility', 0):.2%}, "
            f"ProxyRel={exploit_metrics.get('proxy_relevance', exploit_metrics.get('ndcg', 0)):.4f}, "
            f"Diversity={explore_metrics.get('diversity', 0):.4f}, "
            f"Pareto={len(self.best_solutions)}"
        )

    def _save_generation_history(
        self,
        generation: int,
        exploit_metrics: Dict,
        explore_metrics: Dict,
        constraint_metrics: Dict,
        generation_monitor: Optional[Dict[str, Any]] = None,
    ):
        """Save generation history for analysis."""
        # [Task 2: 每代记录]
        # 每一代都显式写入 feasibility_rate 与 hypervolume，供平滑收敛曲线使用。
        generation_monitor = generation_monitor or {}
        feasibility_rate = float(constraint_metrics.get('overall_feasibility', 0.0))
        hypervolume = self._compute_current_hypervolume()

        self.history.append({
            'generation': generation,
            'feasibility_rate': feasibility_rate,
            'hypervolume': hypervolume,
            # [新增监控指标: population diversity]
            'pareto_size': int(generation_monitor.get('pareto_size', len(self.best_solutions))),
            'population_spacing': float(generation_monitor.get('population_spacing', 0.0)),
            # [新增监控指标: agent contribution]
            'llm_survivors': int(generation_monitor.get('llm_survivors', 0)),
            'de_survivors': int(generation_monitor.get('de_survivors', 0)),
            # [新增监控指标: risk control]
            'invalid_generated_count': int(generation_monitor.get('invalid_generated_count', 0)),
            'current_epsilon': float(generation_monitor.get('current_epsilon', self.constraint_handler.epsilon)),
            # [新增监控指标: objective trajectories]
            'avg_generation_accuracy': float(generation_monitor.get('avg_generation_accuracy', 0.0)),
            'avg_generation_diversity': float(generation_monitor.get('avg_generation_diversity', 0.0)),
            'exploitation_metrics': exploit_metrics,
            'exploration_metrics': explore_metrics,
            'constraint_metrics': constraint_metrics,
            'coordinator_summary': self.coordinator.get_coordination_summary(),
        })

    def _append_final_generation_snapshot(
        self,
        exploit_metrics: Dict,
        explore_metrics: Dict,
        constraint_metrics: Dict,
    ) -> None:
        """Append a terminal snapshot so history can cover [0, max_generations]."""
        if self.config.max_generations < 0:
            return

        target_generation = int(self.config.max_generations)
        last_generation = None
        if self.history and isinstance(self.history[-1], dict):
            last_generation = self.history[-1].get('generation')

        if last_generation == target_generation:
            return

        self._save_generation_history(
            generation=target_generation,
            exploit_metrics=exploit_metrics,
            explore_metrics=explore_metrics,
            constraint_metrics=constraint_metrics,
            generation_monitor=self._latest_generation_monitor,
        )

    # [新增监控指标: population diversity / objective trajectories]
    def _compute_generation_population_snapshot(self) -> Dict[str, Any]:
        """Compute lightweight generation-level monitoring metrics from current evaluated populations."""
        active_individuals = self._get_active_individuals()
        if not active_individuals:
            return {
                'pareto_size': 0,
                'population_spacing': 0.0,
                'avg_generation_accuracy': 0.0,
                'avg_generation_diversity': 0.0,
            }

        fronts = self.exploitation_agent.non_dominated_sort(active_individuals)
        rank1 = fronts[0] if fronts else []
        pareto_size = len(rank1)

        rank1_scores = [ind.scores for ind in rank1 if len(ind.scores) > 0]
        active_scores = [ind.scores for ind in active_individuals if len(ind.scores) > 0]
        if len(rank1_scores) >= 2:
            population_spacing = float(MultiObjectiveMetrics.spacing(rank1_scores))
        elif len(active_scores) >= 2:
            population_spacing = float(MultiObjectiveMetrics.spacing(active_scores))
        else:
            population_spacing = 0.0

        avg_generation_accuracy = float(np.mean([ind.scores[0] for ind in active_individuals])) if active_individuals else 0.0
        avg_generation_diversity = float(np.mean([ind.scores[1] for ind in active_individuals])) if active_individuals else 0.0

        return {
            'pareto_size': int(pareto_size),
            'population_spacing': population_spacing,
            'avg_generation_accuracy': avg_generation_accuracy,
            'avg_generation_diversity': avg_generation_diversity,
        }

    # [新增监控指标: risk control]
    def _count_invalid_generated_offspring(
        self,
        offspring: List[Individual],
        item_features: Dict[str, Dict],
        severe_violation_threshold: float = 0.5,
    ) -> int:
        """
        Count severely invalid offspring in current generation.
        Note: this method only counts and does not discard offspring.
        """
        invalid_count = 0
        for child in offspring:
            violations = self.constraint_handler.calculate_violations(child.item_ids, item_features)
            total_violation = float(np.sum(np.maximum(0.0, np.array(violations, dtype=float))))
            if total_violation > severe_violation_threshold:
                invalid_count += 1
        return int(invalid_count)

    # [新增监控指标: agent contribution]
    def _count_survivors_by_source(self) -> Tuple[int, int]:
        """Count DE/LLM survivors after environmental selection."""
        active_individuals = self._get_active_individuals()
        llm_survivors = sum(1 for ind in active_individuals if getattr(ind, 'source', 'INIT') == 'LLM')
        de_survivors = sum(1 for ind in active_individuals if getattr(ind, 'source', 'INIT') == 'DE')
        return int(llm_survivors), int(de_survivors)

    def _compute_current_hypervolume(self) -> float:
        """Compute current hypervolume on the maintained best solution set."""
        if not self.best_solutions:
            return 0.0

        pareto_scores = [ind.scores for ind in self.best_solutions if len(ind.scores) > 0]
        if not pareto_scores:
            return 0.0

        reference_point = np.array([1.0, 1.0, 1.0])
        return float(MultiObjectiveMetrics.hypervolume(pareto_scores, reference_point))

    def _compute_final_metrics(self) -> Dict[str, Any]:
        """Compute final optimization metrics."""
        if not self.best_solutions:
            return {}

        # Extract Pareto front scores
        pareto_scores = [ind.scores for ind in self.best_solutions]

        # Compute multi-objective metrics
        reference_point = np.array([1.0, 1.0, 1.0])  # Ideal point
        hv = MultiObjectiveMetrics.hypervolume(pareto_scores, reference_point)
        spacing = MultiObjectiveMetrics.spacing(pareto_scores)

        # Compute recommendation metrics
        avg_scores = np.mean(pareto_scores, axis=0)

        return {
            'hypervolume': hv,
            'spacing': spacing,
            'pareto_size': len(self.best_solutions),
            'avg_accuracy': float(avg_scores[0]),
            'avg_diversity': float(avg_scores[1]),
            'avg_novelty': float(avg_scores[2]),
            'feasibility_rate': sum(1 for ind in self.best_solutions if ind.is_feasible) / len(self.best_solutions),
            'coordinator_summary': self.coordinator.get_coordination_summary(),
            'total_generations': self.generation + 1,
        }

    def get_recommendation(self, strategy: str = 'balanced') -> Optional[Individual]:
        """
        Get final recommendation from Pareto front.

        Args:
            strategy: Selection strategy
                - 'balanced': Best average score
                - 'accuracy': Best accuracy
                - 'diversity': Best diversity
                - 'novelty': Best novelty

        Returns:
            Selected recommendation solution
        """
        if not self.best_solutions:
            return None

        if strategy == 'balanced':
            return max(self.best_solutions, key=lambda x: np.mean(x.scores))
        elif strategy == 'accuracy':
            return max(self.best_solutions, key=lambda x: x.scores[0])
        elif strategy == 'diversity':
            return max(self.best_solutions, key=lambda x: x.scores[1])
        elif strategy == 'novelty':
            return max(self.best_solutions, key=lambda x: x.scores[2])
        else:
            return self.best_solutions[0]

    def save_results(self, output_path: str):
        """Save optimization results to file."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        results = {
            'config': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'recommendation_size': self.config.recommendation_size,
                'is_single_population': self.config.is_single_population,
                'use_llm': self.config.use_llm,
                'llm_model': self.config.llm_model,
            },
            'final_metrics': self._compute_final_metrics(),
            'history': self.history,
            'generation_history': self.history,
            'best_solutions': [
                {
                    'items': ind.item_ids,
                    'scores': ind.scores.tolist(),
                    'violations': ind.constraint_violations.tolist(),
                    'is_feasible': ind.is_feasible,
                }
                for ind in self.best_solutions
            ],
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def run_demo():
    """Run a simple demo of DualAgent-Rec."""
    import random

    # Create synthetic data
    num_items = 500
    num_history = 50

    candidate_items = [f"item_{i}" for i in range(num_items)]
    categories = ['Electronics', 'Computers', 'Phone', 'Camera', 'Audio', 'Gaming']
    sellers = [f"seller_{i}" for i in range(50)]

    item_features = {
        item_id: {
            'category': random.choice(categories),
            'seller_id': random.choice(sellers),
            'is_new': random.random() < 0.2,
            'popularity': random.random(),
        }
        for item_id in candidate_items
    }

    user_history = [
        {
            'item_id': random.choice(candidate_items),
            'category': random.choice(categories),
            'rating': random.randint(1, 5),
        }
        for _ in range(num_history)
    ]

    user_profile = {
        'interaction_count': num_history,
        'category_diversity': 0.7,
        'avg_rating': 4.2,
    }

    # Run optimization
    config = DualAgentConfig(
        population_size=50,
        max_generations=30,
        recommendation_size=10,
        use_llm=False,  # Disable LLM for quick demo
    )

    framework = DualAgentRec(config)

    best_solutions, metrics = framework.optimize(
        candidate_items=candidate_items,
        user_history=user_history,
        item_features=item_features,
        user_profile=user_profile
    )

    print("\n=== Demo Results ===")
    print(f"Pareto-optimal solutions: {len(best_solutions)}")
    print(f"Hypervolume: {metrics.get('hypervolume', 0):.4f}")
    print(f"Average accuracy: {metrics.get('avg_accuracy', 0):.4f}")
    print(f"Average diversity: {metrics.get('avg_diversity', 0):.4f}")
    print(f"Feasibility rate: {metrics.get('feasibility_rate', 0):.2%}")

    # Get recommendation
    rec = framework.get_recommendation('balanced')
    if rec:
        print(f"\nRecommended items: {rec.item_ids[:5]}...")
        print(f"Scores: {rec.scores}")


if __name__ == "__main__":
    run_demo()
