"""Minimal checkpointed Phase 3 Agent example using local Qwen."""

from copa.phase2.cli import load_config
from copa.phase3.cli import build_agent_pipeline
from copa.phase3.models import AgentRecommendationRequest
from copa.data import build_synthetic_case


def main() -> None:
    config = load_config("COPA/configs/phase3_agent.yaml")
    pipeline = build_agent_pipeline(config)
    case = build_synthetic_case()
    try:
        result = pipeline.run(
            AgentRecommendationRequest(
                case.user_id,
                "推荐5个价格不超过60美元的商品，品牌尽量多样",
                case.candidates,
                base_constraints=case.constraints,
                context=case.context,
            ),
            thread_id="phase3-example",
        )
        print(result.to_dict())
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
