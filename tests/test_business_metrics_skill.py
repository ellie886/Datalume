from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from datalume.agent_runtime import AgentConfig, run_agent
from datalume.deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError
from datalume.skill_loader import SkillLoader, SkillLoadError, deterministic_skill_selection


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def planning_message(**changes):
    payload = {
        "action": "continue",
        "skill": "business-metrics",
        "uses_pending": False,
        **changes,
    }
    return SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))


def tool_message(name, metric):
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="metric-call",
        function=SimpleNamespace(name=name, arguments=json.dumps({"metric": metric})),
    )])


def scripted_client(*messages):
    class Completions:
        def __init__(self):
            self.messages = list(messages)
            self.requests = []

        def create(self, **kwargs):
            self.requests.append(kwargs)
            message = self.messages.pop(0)
            if callable(message):
                message = message(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = Completions()
    client = DeepSeekClient(
        DeepSeekConfig(api_key="mock"),
        sdk_client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return client, completions


def run(client, bundle, question, *, context=None, loader=None):
    orders, _, overview, metrics, capabilities = bundle
    return run_agent(
        question,
        database_id="metrics-test",
        orders=orders,
        overview=overview,
        metrics=metrics,
        capabilities=capabilities,
        llm_client=client,
        conversation_context=context if context is not None else {},
        config=AgentConfig(True, 4),
        skill_loader=loader or SkillLoader(PROJECT_ROOT / "skills"),
    )


def observed_conclusion(request):
    observation = json.loads(next(
        item["content"] for item in reversed(request["messages"])
        if item["role"] == "tool"
    ))
    return SimpleNamespace(content=observation["summary"], tool_calls=[])


@pytest.mark.parametrize(("question", "tool", "metric"), [
    ("哪个商品表现最好？", "analyze_product", "revenue"),
    ("哪些是高价值用户？", "analyze_user", "revenue"),
    ("当前用户复购情况怎么样？", "analyze_repurchase", "repurchase_rate"),
])
def test_metrics_skill_reaches_planner_and_selected_tool_loop(
    default_bundle, question, tool, metric,
):
    context = {}
    client, sdk = scripted_client(
        planning_message(), tool_message(tool, metric), observed_conclusion,
    )

    response = run(client, default_bundle, question, context=context)

    instructions = SkillLoader().load("business-metrics").instructions
    assert instructions in sdk.requests[0]["messages"][0]["content"]
    assert instructions in sdk.requests[1]["messages"][0]["content"]
    assert response.skills_used == ("business-metrics",)
    assert [step.tool_name for step in response.steps] == [tool]
    assert response.plan.metric == metric
    assert response.primary_result.supported
    assert response.answer == response.primary_result.conclusion
    assert context["last_skills_used"] == ["business-metrics"]
    assert context["previous_metric"] == metric
    assert len(sdk.requests) == 3  # 一次规划、一次工具选择、一次基于 Observation 回答。
    assert len(sdk.requests[1]["tools"]) == 9


def test_explicit_metric_keeps_simple_query_and_no_business_skill(default_bundle):
    client, sdk = scripted_client(
        planning_message(skill=None),
        tool_message("analyze_product", "quantity"),
        observed_conclusion,
    )

    response = run(client, default_bundle, "哪个商品销量最高？")

    assert response.skills_used == ()
    assert SkillLoader().load("business-metrics").instructions not in (
        sdk.requests[1]["messages"][0]["content"]
    )
    assert response.primary_result.metrics["product_quantity"]["原味欧包"] == 16
    assert response.plan.metric == "quantity"


def test_metric_clarification_resumes_with_exact_user_choice(default_bundle):
    context = {}
    client, sdk = scripted_client(
        planning_message(
            action="clarify", ambiguity_type="metric", options=["销量", "销售额"],
        ),
        planning_message(uses_pending=True),
        tool_message("analyze_product", "quantity"),
        observed_conclusion,
    )
    goal = "我只保留一个商品，哪个最好？"

    clarification = run(client, default_bundle, goal, context=context)
    assert clarification.needs_clarification
    assert clarification.steps == ()
    assert len(sdk.requests) == 1
    assert context["pending_clarification"]["original_question"] == goal

    response = run(client, default_bundle, "按销量看", context=context)

    user_payload = json.loads(sdk.requests[2]["messages"][1]["content"])
    assert user_payload["user_goal"] == goal + "\n用户补充：按销量看"
    assert not response.needs_clarification
    assert response.plan.metric == "quantity"
    assert "pending_clarification" not in context


def test_explicit_profit_with_missing_fields_uses_existing_refusal(default_bundle):
    client, sdk = scripted_client(
        planning_message(),
        tool_message("analyze_profit", "profit"),
        SimpleNamespace(content="虚构利润 9999 元", tool_calls=[]),
    )

    response = run(client, default_bundle, "按利润判断哪个商品最好？")

    assert not response.needs_clarification
    assert not response.primary_result.supported
    assert [step.tool_name for step in response.steps] == ["analyze_profit"]
    assert "缺少成本" in response.answer
    assert "9999" not in response.answer
    planning_payload = json.loads(sdk.requests[0]["messages"][1]["content"])
    assert not planning_payload["data_capabilities"]["profit_supported"]


def test_metrics_policy_does_not_displace_diagnosis(default_bundle):
    client, sdk = scripted_client(
        planning_message(skill="business-diagnosis"),
        tool_message("analyze_trend", "revenue"),
        observed_conclusion,
    )

    response = run(client, default_bundle, "最近销售为什么下降？")

    assert response.skills_used == ("business-diagnosis",)
    assert SkillLoader().load("business-metrics").instructions in (
        sdk.requests[0]["messages"][0]["content"]
    )
    assert SkillLoader().load("business-diagnosis").instructions in (
        sdk.requests[1]["messages"][0]["content"]
    )


def test_missing_metrics_body_does_not_disable_request_planning(default_bundle):
    class MissingMetricsLoader(SkillLoader):
        def load(self, name):
            if name == "business-metrics":
                raise SkillLoadError("metrics body unavailable")
            return super().load(name)

    client, sdk = scripted_client(
        planning_message(skill=None),
        tool_message("analyze_product", "quantity"),
        observed_conclusion,
    )

    response = run(
        client, default_bundle, "哪个商品销量最高？",
        loader=MissingMetricsLoader(PROJECT_ROOT / "skills"),
    )

    assert response.fallback_level == "agent"
    assert len(sdk.requests) == 3
    assert response.primary_result.supported


def test_metrics_selection_survives_planner_and_selector_failure(default_bundle):
    client, sdk = scripted_client(
        tool_message("analyze_product", "revenue"), observed_conclusion,
    )

    def unavailable(*args, **kwargs):
        raise DeepSeekError("selection unavailable")

    client.plan_request = unavailable
    client.select_skill = unavailable

    response = run(client, default_bundle, "哪个商品表现最好？")

    assert response.skills_used == ("business-metrics",)
    assert response.primary_result.supported
    assert len(sdk.requests) == 2


@pytest.mark.parametrize(("reply", "uses_pending"), [
    ("按利润额看", True),
    ("按复购率看", True),
    ("换个问题，哪个商品利润最高？", False),
])
def test_planning_failure_distinguishes_metric_supplement_from_new_goal(
    default_bundle, reply, uses_pending,
):
    original = "我只保留一个商品，哪个最好？"
    context = {
        "pending_clarification": {
            "original_question": original, "attempts": 1, "supplements": [],
        },
    }
    client, sdk = scripted_client(
        tool_message("analyze_profit", "profit"), observed_conclusion,
    )

    def unavailable(*args, **kwargs):
        raise DeepSeekError("selection unavailable")

    client.plan_request = unavailable
    client.select_skill = unavailable

    run(client, default_bundle, reply, context=context)

    payload = json.loads(sdk.requests[0]["messages"][1]["content"])
    expected = original + "\n用户补充：" + reply if uses_pending else reply
    assert payload["user_goal"] == expected
    assert "pending_clarification" not in context


@pytest.mark.parametrize(("question", "expected"), [
    ("哪个渠道表现最好？", "business-metrics"),
    ("客单价怎么算？", "business-metrics"),
    ("高价值用户有哪些？", "business-metrics"),
    ("销售额高就是利润高吗？", "business-metrics"),
    ("哪个商品销量最高？", None),
    ("销售额是多少？", None),
    ("最近销售为什么下降？", "business-diagnosis"),
])
def test_deterministic_selection_stays_conservative(question, expected):
    metadata = SkillLoader(PROJECT_ROOT / "skills").list_metadata()
    assert deterministic_skill_selection(question, metadata) == expected
