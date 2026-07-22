"""Closed-world domain capabilities and their extensible registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from copa.core import CandidateRecord


AttributeKind = Literal["numeric", "categorical", "boolean", "identifier", "set"]


@dataclass(frozen=True)
class AttributeCapability:
    logical_name: str
    executable_attribute: str
    kind: AttributeKind
    operators: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    value_aliases: Mapping[str, Any] = field(default_factory=dict)
    special_transform: Optional[str] = None
    executor_type: Optional[str] = None
    parameter_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    slate_aggregations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectiveCapability:
    ir_name: str
    executable_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    parameter_schema: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""


class DomainSchema:
    def __init__(
        self,
        name: str,
        attributes: Sequence[AttributeCapability],
        objectives: Sequence[ObjectiveCapability],
        *,
        currency: str = "USD",
        enumerate_limit: int = 40,
    ) -> None:
        self.name = name
        self.attributes = list(attributes)
        self.objectives = list(objectives)
        self.currency = currency.upper()
        self.enumerate_limit = enumerate_limit
        self._attributes: Dict[str, AttributeCapability] = {}
        self._objectives: Dict[str, ObjectiveCapability] = {}
        for capability in self.attributes:
            for key in (capability.logical_name, capability.executable_attribute, *capability.aliases):
                self._attributes[self._key(key)] = capability
        for capability in self.objectives:
            for key in (capability.ir_name, *capability.aliases):
                self._objectives[self._key(key)] = capability

    @staticmethod
    def _key(value: str) -> str:
        return str(value).strip().casefold().replace("-", "_").replace(" ", "_")

    def resolve_attribute(self, name: str) -> Optional[AttributeCapability]:
        return self._attributes.get(self._key(name))

    def resolve_objective(self, name: str) -> Optional[ObjectiveCapability]:
        return self._objectives.get(self._key(name))

    def candidate_values(self, candidates: Sequence[CandidateRecord], capability: AttributeCapability) -> List[Any]:
        values: List[Any] = []
        for candidate in candidates:
            if capability.executable_attribute == "item_id":
                value = candidate.item_id
            elif capability.executable_attribute == "base_score":
                value = candidate.base_score
            elif capability.executable_attribute in candidate.metadata:
                value = candidate.metadata[capability.executable_attribute]
            else:
                continue
            if capability.kind == "set" and isinstance(value, (list, tuple, set, frozenset)):
                for member in value:
                    if member is not None and member not in values:
                        values.append(member)
            elif value is not None and value not in values:
                values.append(value)
        return values

    def prompt_summary(self, candidates: Sequence[CandidateRecord]) -> Dict[str, Any]:
        attributes = []
        for capability in self.attributes:
            values = self.candidate_values(candidates, capability)
            payload: Dict[str, Any] = {
                "name": capability.logical_name,
                "kind": capability.kind,
                "operators": list(capability.operators),
                "aliases": list(capability.aliases),
                "value_schema": dict(capability.parameter_schema),
                "description": capability.description,
                "slate_aggregations": list(capability.slate_aggregations),
            }
            if capability.kind in {"categorical", "boolean"} and len(values) <= self.enumerate_limit:
                payload["available_values"] = values
            elif capability.kind == "numeric" and values:
                try:
                    numeric = [float(value) for value in values]
                except (TypeError, ValueError):
                    payload["value_policy"] = "candidate_metadata_type_checked_after_generation"
                else:
                    payload["observed_range"] = [min(numeric), max(numeric)]
            else:
                payload["value_policy"] = "catalog_value_checked_after_generation"
            attributes.append(payload)
        return {
            "domain": self.name,
            "currency": self.currency,
            "attributes": attributes,
            "soft_objectives": [
                {
                    "name": capability.ir_name,
                    "aliases": list(capability.aliases),
                    "parameter_schema": dict(capability.parameter_schema),
                    "description": capability.description,
                }
                for capability in self.objectives
            ],
        }


class DomainSchemaRegistry:
    """Registers semantic schemas without coupling new domains to compiler code."""

    def __init__(self) -> None:
        self._schemas: Dict[str, DomainSchema] = {}
        self._canonical_names: Dict[str, str] = {}

    @staticmethod
    def _key(value: str) -> str:
        return DomainSchema._key(value)

    def register(
        self,
        name: str,
        schema: DomainSchema,
        *,
        aliases: Sequence[str] = (),
        replace: bool = False,
    ) -> None:
        canonical = self._key(name)
        if not canonical:
            raise ValueError("Domain schema name cannot be empty")
        keys = {canonical, self._key(schema.name), *(self._key(alias) for alias in aliases)}
        collisions = sorted(key for key in keys if key in self._schemas and self._schemas[key] is not schema)
        if collisions and not replace:
            raise KeyError(f"Domain schema aliases already registered: {collisions}")
        for key in keys:
            self._schemas[key] = schema
            self._canonical_names[key] = canonical

    def get(self, name: str) -> DomainSchema:
        key = self._key(name)
        if key not in self._schemas:
            raise KeyError(f"Unknown COPA compiler domain: {name}")
        return self._schemas[key]

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._canonical_names.values())))


# The requirements call this the Constraint Schema Registry. Keep the more
# precise DomainSchemaRegistry name while exposing the requested public alias.
ConstraintSchemaRegistry = DomainSchemaRegistry


RELEVANCE = ObjectiveCapability(
    "relevance", "relevance", aliases=("accuracy", "相关性", "准确性"),
    description="Maximize the deterministic recommender base score.",
)
NOVELTY = ObjectiveCapability(
    "novelty", "novelty", aliases=("新颖性", "冷门", "少见"),
    description="Prefer candidates with lower normalized popularity.",
)
FAIRNESS = ObjectiveCapability(
    "fairness", "fairness", aliases=("公平性", "公平"),
    description="Reduce exposure-distribution distance for the configured auditable group.",
)


def _diversity(name: str, logical_attribute: str, aliases: Tuple[str, ...]) -> ObjectiveCapability:
    return ObjectiveCapability(
        name,
        "diversity",
        {"logical_attribute": logical_attribute},
        aliases=aliases,
        description=f"Maximize intra-slate diversity over {logical_attribute}.",
    )


def synthetic_domain_schema() -> DomainSchema:
    return DomainSchema(
        "synthetic",
        [
            AttributeCapability("price", "price", "numeric", ("<", "<=", ">", ">=", "==", "!=", "between"), ("价格",), parameter_schema={"type": "number", "currency": "USD"}, description="Candidate price in USD.", slate_aggregations=("aggregate_sum",)),
            AttributeCapability("category", "category", "categorical", ("==", "!=", "in", "not_in"), ("类别", "品类"), parameter_schema={"type": "catalog_value_or_array"}, description="Synthetic product category.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("brand", "brand_id", "categorical", ("==", "!=", "in", "not_in"), ("brand_id", "品牌"), parameter_schema={"type": "catalog_value_or_array"}, description="Synthetic product brand identifier.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("group", "group", "categorical", ("==", "!=", "in", "not_in"), ("群体",), parameter_schema={"type": "catalog_value_or_array"}, description="Auditable synthetic group.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("popularity", "popularity", "numeric", ("<", "<=", ">", ">=", "between"), ("流行度", "热度"), parameter_schema={"type": "number", "range": [0, 1]}, description="Normalized item popularity."),
            AttributeCapability("availability", "availability", "boolean", ("==", "!="), ("available", "有货", "库存可用"), parameter_schema={"type": "boolean"}, description="Whether the candidate is available."),
            AttributeCapability("item_id", "item_id", "identifier", ("==", "!=", "in", "not_in"), ("商品id", "物品id"), parameter_schema={"type": "catalog_identifier_or_array"}, description="Candidate identifier; only user-supplied exclusions are allowed."),
        ],
        [
            RELEVANCE,
            _diversity("brand_diversity", "brand", ("品牌多样性",)),
            _diversity("category_diversity", "category", ("diversity", "类别多样性", "品类多样性")),
            NOVELTY,
            FAIRNESS,
        ],
    )


def beauty_domain_schema() -> DomainSchema:
    return DomainSchema(
        "all_beauty",
        [
            AttributeCapability("price", "price_filled", "numeric", ("<", "<=", ">", ">=", "==", "!=", "between"), ("price_filled", "价格"), parameter_schema={"type": "number", "currency": "USD"}, description="Filled Amazon price in USD.", slate_aggregations=("aggregate_sum",)),
            AttributeCapability("category", "main_category", "categorical", ("==", "!=", "in", "not_in"), ("main_category", "类别", "品类"), parameter_schema={"type": "catalog_value_or_array"}, description="Amazon main category.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("brand", "brand_id", "categorical", ("==", "!=", "in", "not_in"), ("brand_id", "品牌"), parameter_schema={"type": "catalog_value_or_array"}, description="Normalized Amazon brand identifier.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("seller", "seller_id", "categorical", ("==", "!=", "in", "not_in"), ("seller_id", "卖家"), parameter_schema={"type": "catalog_value_or_array"}, description="Normalized seller identifier.", slate_aggregations=("distinct_count", "per_group_count", "group_count")),
            AttributeCapability("popularity", "popularity", "numeric", ("<", "<=", ">", ">=", "between"), ("流行度", "热度"), parameter_schema={"type": "number", "range": [0, 1]}, description="Normalized training-set popularity."),
            AttributeCapability("availability", "inventory_initial", "boolean", ("==", "!="), ("available", "有货", "库存可用"), special_transform="positive_inventory", parameter_schema={"type": "boolean"}, description="Availability derived from inventory_initial > 0."),
            AttributeCapability("item_id", "item_id", "identifier", ("==", "!=", "in", "not_in"), ("商品id", "物品id"), parameter_schema={"type": "catalog_identifier_or_array"}, description="Candidate identifier; only user-supplied exclusions are allowed."),
        ],
        [
            RELEVANCE,
            _diversity("brand_diversity", "brand", ("品牌多样性",)),
            _diversity("category_diversity", "category", ("diversity", "类别多样性", "品类多样性")),
            _diversity("seller_diversity", "seller", ("卖家多样性",)),
            NOVELTY,
        ],
    )


def mind_domain_schema() -> DomainSchema:
    """MIND contract; adapters must derive entity_ids and age_hours metadata."""

    return DomainSchema(
        "mind",
        [
            AttributeCapability("topic", "category", "categorical", ("==", "!=", "in", "not_in"), ("category", "主题", "话题"), parameter_schema={"type": "catalog_value_or_array"}, description="MIND top-level news category."),
            AttributeCapability("subtopic", "subcategory", "categorical", ("==", "!=", "in", "not_in"), ("subcategory", "子主题", "子话题"), parameter_schema={"type": "catalog_value_or_array"}, description="MIND news subcategory."),
            AttributeCapability("entity", "entity_ids", "set", ("contains_any", "contains_all", "not_contains"), ("entity_id", "实体"), executor_type="set_membership", parameter_schema={"type": "catalog_value_or_array"}, description="Wikidata IDs parsed from title_entities and abstract_entities."),
            AttributeCapability("freshness", "age_hours", "numeric", ("<", "<=", ">", ">=", "between"), ("age", "时效", "新鲜度", "发布时间"), parameter_schema={"type": "number", "unit": "hours"}, description="News age in hours at recommendation time; requires an adapter-provided reference timestamp."),
            AttributeCapability("popularity", "popularity", "numeric", ("<", "<=", ">", ">=", "between"), ("流行度", "热度"), parameter_schema={"type": "number", "range": [0, 1]}, description="Normalized training-impression popularity."),
            AttributeCapability("item_id", "item_id", "identifier", ("==", "!=", "in", "not_in"), ("news_id", "新闻id"), parameter_schema={"type": "catalog_identifier_or_array"}, description="MIND news identifier; only user-supplied exclusions are allowed."),
        ],
        [
            RELEVANCE,
            _diversity("topic_diversity", "topic", ("diversity", "主题多样性", "话题多样性")),
            _diversity("entity_diversity", "entity", ("实体多样性",)),
            NOVELTY,
        ],
    )


DEFAULT_DOMAIN_SCHEMA_REGISTRY = DomainSchemaRegistry()
DEFAULT_DOMAIN_SCHEMA_REGISTRY.register("synthetic", synthetic_domain_schema(), aliases=("demo",))
DEFAULT_DOMAIN_SCHEMA_REGISTRY.register("all_beauty", beauty_domain_schema(), aliases=("beauty", "allbeauty"))
DEFAULT_DOMAIN_SCHEMA_REGISTRY.register("mind", mind_domain_schema(), aliases=("news",))


def get_domain_schema(name: str, registry: Optional[DomainSchemaRegistry] = None) -> DomainSchema:
    return (registry or DEFAULT_DOMAIN_SCHEMA_REGISTRY).get(name)
