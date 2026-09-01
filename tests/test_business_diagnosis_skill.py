from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_models import AgentModelTurn, AgentToolCall
from agent_runtime import AgentConfig, run_agent
from data_processor import (
    calculate_metrics,
    detect_data_capabilities,
    process_order_csv,
    summarize_orders,
)
from deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError
from skill_loader import (
    LoadedSkill,
    SkillLoader,
    SkillLoadError,
    SkillMetadata,
    deterministic_skill_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tool_turn(call_id: str, name: str, arguments=None) -> AgentModelTurn:
    arguments = arguments or {}
    call = AgentToolCall(call_id, name, arguments)
    return AgentModelTurn(
        None,
        (call,),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        },
    )


def final_turn(answer: str) -> AgentModelTurn:
    return AgentModelTurn(answer, (), {"role": "assistant", "content": answer})


class SkillAwareClient:
    def __init__(self, selection, turns):
        self.selection = selection
        self.turns = list(turns)
        self.selection_requests = []
        self.agent_requests = []

    def select_skill(self, question, *, skills, conversation_context, capabilities):
        self.selection_requests.append({"question": question, "skills": skills})
        if isinstance(self.selection, Exception):
            raise self.selection
        return self.selection

    def agent_completion(self, messages, *, tools):
        self.agent_requests.append([dict(item) for item in messages])
        return self.turns.pop(0)


def bundle_from_bytes(content: bytes):
    orders, quality = process_order_csv(content)
    return (
        orders,
        quality,
        summarize_orders(orders),
        calculate_metrics(orders),
        detect_data_capabilities(orders),
    )


def run(client, bundle, question, *, database_id="db-a", context=None, loader=None):
    orders, _, overview, metrics, caps = bundle
    return run_agent(
        question,
        database_id=database_id,
        orders=orders,
        overview=overview,
        metrics=metrics,
        capabilities=caps,
        llm_client=client,
        conversation_context=context if context is not None else {},
        config=AgentConfig(True, 4),
        skill_loader=loader or SkillLoader(PROJECT_ROOT / "skills"),
    )


def aov_decline_bundle():
    rows = ["order_id,date,user_id,product,price,quantity,gender,grade"]
    for day in range(1, 15):
        quantity = 2 if day <= 7 else 1
        rows.append(f"o{day},2026-08-{day:02d},u{day},同款欧包,15,{quantity},女,大二")
    return bundle_from_bytes(("\n".join(rows) + "\n").encode("utf-8"))


def demo_bundle():
    return bundle_from_bytes((Path("/Users/wyy/Desktop/Datalume/data/demo_orders.csv")).read_bytes())


def test_skill_loader_progressively_exposes_metadata_then_full_instructions():
    loader = SkillLoader(PROJECT_ROOT / "skills")
    metadata = loader.list_metadata()
    assert metadata == (SkillMetadata(
        "business-diagnosis",
        "Diagnose changes or problems in business performance using available trend, order, product, channel and user data. Use when the user asks why business performance changed, what is driving a decline or anomaly, or where an operating problem may be coming from.",
    ),)
    assert "Objective" not in metadata[0].description
    loaded = loader.load("business-diagnosis")
    assert "## Objective" in loaded.instructions
    assert "Hypothesis Testing" in loaded.instructions
    assert "停止条件" in loaded.instructions


def test_deepseek_skill_selector_receives_metadata_not_full_skill():
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content='{"skill":"business-diagnosis"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FakeCompletions()
    client = DeepSeekClient(
        DeepSeekConfig(api_key="mock"),
        sdk_client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    metadata = SkillLoader(PROJECT_ROOT / "skills").list_metadata()
    selected = client.select_skill(
        "最近销售为什么下降？",
        skills=[item.as_prompt_payload() for item in metadata],
        conversation_context={},
        capabilities=None,
    )
    assert selected == "business-diagnosis"
    payload = completions.kwargs["messages"][1]["content"]
    assert "Diagnose changes" in payload
    assert "Hypothesis Testing" not in payload


def test_diagnostic_question_selects_and_loads_business_skill():
    client = SkillAwareClient(
        "business-diagnosis",
        [tool_turn("c1", "analyze_trend", {"metric": "revenue"}), final_turn("诊断完成。")],
    )
    response = run(client, demo_bundle(), "最近销售为什么下降？")
    assert response.skills_used == ("business-diagnosis",)
    assert response.public_trace()["skills_used"] == ["business-diagnosis"]
    assert "Revenue ≈ Orders × Average Order Value" in client.agent_requests[0][0]["content"]


def test_simple_product_query_does_not_activate_business_skill(default_bundle):
    client = SkillAwareClient(
        None,
        [tool_turn("c1", "analyze_product", {"metric": "quantity"}), final_turn("商品查询完成。")],
    )
    response = run(client, default_bundle, "哪个商品销量最高？")
    assert response.skills_used == ()
    assert [step.tool_name for step in response.steps] == ["analyze_product"]
    assert "## Objective" not in client.agent_requests[0][0]["content"]


def test_order_decline_observation_can_drive_channel_hypothesis():
    class OrderDrivenClient(SkillAwareClient):
        def __init__(self):
            super().__init__("business-diagnosis", [])
            self.calls = 0

        def agent_completion(self, messages, *, tools):
            self.agent_requests.append([dict(item) for item in messages])
            self.calls += 1
            if self.calls == 1:
                return tool_turn("c1", "analyze_trend", {"metric": "revenue"})
            if self.calls == 2:
                observation = json.loads([m for m in messages if m.get("role") == "tool"][-1]["content"])
                assert observation["metrics"]["recent_orders"] < observation["metrics"]["previous_orders"]
                return tool_turn("c2", "analyze_channel", {"metric": "orders"})
            return final_turn("订单量下降后，进一步检查了渠道订单表现。")

    response = run(OrderDrivenClient(), demo_bundle(), "最近销售为什么下降？")
    assert [step.tool_name for step in response.steps] == ["analyze_trend", "analyze_channel"]


def test_stable_orders_and_lower_aov_prioritize_product_hypothesis():
    class AovDrivenClient(SkillAwareClient):
        def __init__(self):
            super().__init__("business-diagnosis", [])
            self.calls = 0

        def agent_completion(self, messages, *, tools):
            self.calls += 1
            if self.calls == 1:
                return tool_turn("c1", "analyze_trend", {"metric": "revenue"})
            if self.calls == 2:
                observation = json.loads([m for m in messages if m.get("role") == "tool"][-1]["content"])
                metrics = observation["metrics"]
                assert metrics["recent_orders"] == metrics["previous_orders"]
                assert metrics["recent_aov"] < metrics["previous_aov"]
                return tool_turn("c2", "analyze_product", {"metric": "revenue"})
            return final_turn("客单价下降后，已检验商品结构假设。")

    response = run(AovDrivenClient(), aov_decline_bundle(), "最近销售为什么下降？")
    assert [step.tool_name for step in response.steps] == ["analyze_trend", "analyze_product"]


def test_unsupported_product_hypothesis_is_not_forced_and_limits_are_stated():
    answer = (
        "可以确认销售下降主要与客单价下降有关，但当前商品数据没有发现足以解释该变化的结构性原因。"
        "由于缺少优惠、促销和实际支付字段，暂时无法进一步确认客单价下降的具体原因。"
    )
    client = SkillAwareClient(
        "business-diagnosis",
        [
            tool_turn("c1", "analyze_trend", {"metric": "revenue"}),
            tool_turn("c2", "analyze_product", {"metric": "revenue"}),
            final_turn(answer),
        ],
    )
    response = run(client, aov_decline_bundle(), "最近销售为什么下降？")
    assert "商品结构导致" not in response.answer
    assert "没有发现足以解释" in response.answer
    assert "缺少优惠、促销和实际支付字段" in response.answer
    assert response.stop_reason == "final_answer"


def test_skill_selection_failure_uses_deterministic_fallback():
    client = SkillAwareClient(
        DeepSeekError("selection unavailable"),
        [tool_turn("c1", "analyze_trend", {"metric": "revenue"}), final_turn("完成。")],
    )
    response = run(client, demo_bundle(), "帮我诊断一下最近的经营问题")
    assert response.skills_used == ("business-diagnosis",)


def test_skill_loader_failure_continues_with_agent_v1(default_bundle):
    class BrokenLoader(SkillLoader):
        def list_metadata(self):
            raise SkillLoadError("broken")

    client = SkillAwareClient(
        "business-diagnosis",
        [tool_turn("c1", "analyze_product", {"metric": "quantity"}), final_turn("原 Agent 正常完成。")],
    )
    response = run(
        client,
        default_bundle,
        "最近销售为什么下降？",
        loader=BrokenLoader(PROJECT_ROOT / "skills"),
    )
    assert response.answer == "原 Agent 正常完成。"
    assert response.skills_used == ()
    assert [step.tool_name for step in response.steps] == ["analyze_product"]


def test_skill_state_is_isolated_by_database(default_bundle):
    diagnosis_context, query_context = {}, {}
    diagnosis_client = SkillAwareClient(
        "business-diagnosis",
        [tool_turn("c1", "analyze_trend", {"metric": "revenue"}), final_turn("A")],
    )
    query_client = SkillAwareClient(
        None,
        [tool_turn("c2", "analyze_product", {"metric": "quantity"}), final_turn("B")],
    )
    response_a = run(
        diagnosis_client, default_bundle, "最近销售为什么下降？",
        database_id="db-a", context=diagnosis_context,
    )
    response_b = run(
        query_client, default_bundle, "哪个商品销量最高？",
        database_id="db-b", context=query_context,
    )
    assert response_a.skills_used == ("business-diagnosis",)
    assert response_b.skills_used == ()
    assert diagnosis_context["last_skills_used"] == ["business-diagnosis"]
    assert query_context["last_skills_used"] == []


def test_deterministic_selector_does_not_trigger_on_sales_word_alone():
    metadata = SkillLoader(PROJECT_ROOT / "skills").list_metadata()
    assert deterministic_skill_selection("销售额是多少？", metadata) is None
    assert deterministic_skill_selection("哪个渠道销售额最高？", metadata) is None
    assert deterministic_skill_selection("最近销售为什么下降？", metadata) == "business-diagnosis"
    assert deterministic_skill_selection("最近哪里出了问题？", metadata) == "business-diagnosis"
