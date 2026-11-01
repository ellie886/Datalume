from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from datalume.agent_models import AgentModelTurn, AgentRequestDecision, AgentToolCall
from datalume.agent_runtime import AgentConfig, run_agent
from datalume.database_manager import create_database, initialize_database_state
from datalume.deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError, IntentPlan
from datalume.skill_loader import SkillLoader


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


class PlanningClient:
    def __init__(self, decisions, turns=()):
        self.decisions = list(decisions)
        self.turns = list(turns)
        self.planning_requests = []
        self.agent_requests = []

    def plan_request(self, question, **kwargs):
        self.planning_requests.append({"question": question, **kwargs})
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision

    def select_skill(self, question, *, skills, conversation_context, capabilities):
        return None

    def agent_completion(self, messages, *, tools):
        self.agent_requests.append([dict(item) for item in messages])
        return self.turns.pop(0)


def run(client, bundle, question, *, database_id="db-a", context=None):
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
        skill_loader=SkillLoader(PROJECT_ROOT / "skills"),
    )


def test_clarification_skill_is_small_and_loadable():
    loader = SkillLoader(PROJECT_ROOT / "skills")
    skill = loader.load("clarify-analysis-request")
    assert "关键歧义" in skill.instructions
    assert "一个现有分析 Tool" in skill.instructions
    assert "数据缺少必要字段" in skill.instructions
    assert "不计算指标" in skill.instructions


def test_missing_referent_requests_one_clarification_without_tools(default_bundle):
    context = {}
    client = PlanningClient([
        AgentRequestDecision(
            "clarify",
            "",
            clarification_question="你想比较哪两个商品？",
            options=("原味欧包和全麦欧包", "其他商品"),
            reason="缺少比较对象",
            ambiguity_type="referent",
        )
    ])

    response = run(client, default_bundle, "帮我对比一下这两个", context=context)

    assert response.needs_clarification
    assert response.stop_reason == "needs_clarification"
    assert response.steps == ()
    assert response.skills_used == ("clarify-analysis-request",)
    assert "你提到的对象具体指什么" in response.answer
    assert context["pending_clarification"]["original_question"] == "帮我对比一下这两个"
    assert client.planning_requests[0]["available_tools"]


def test_clarification_answer_resumes_original_goal_and_clears_pending(default_bundle):
    context = {
        "pending_clarification": {
            "original_question": "帮我对比一下这两个",
            "clarification_question": "你想比较哪两个商品？",
            "options": [],
            "attempts": 1,
        }
    }
    client = PlanningClient(
        [AgentRequestDecision(
            "continue",
            "模型改写不应被执行",
            uses_pending=True,
        )],
        [
            tool_turn("c1", "analyze_product", {"metric": "quantity"}),
            final_turn("已完成两个商品的销量比较。"),
        ],
    )

    response = run(client, default_bundle, "原味欧包和全麦欧包", context=context)

    assert not response.needs_clarification
    assert [step.tool_name for step in response.steps] == ["analyze_product"]
    assert "pending_clarification" not in context
    first_payload = json.loads(client.agent_requests[0][1]["content"])
    assert first_payload["user_goal"] == (
        "帮我对比一下这两个\n用户补充：原味欧包和全麦欧包"
    )


def test_clear_new_goal_overrides_pending_request(default_bundle):
    context = {
        "pending_clarification": {
            "original_question": "帮我对比一下这两个",
            "clarification_question": "你想比较哪两个商品？",
            "attempts": 1,
        }
    }
    new_goal = "最近一周销售为什么下降？"
    client = PlanningClient(
        [AgentRequestDecision("continue", new_goal, skill="business-diagnosis")],
        [tool_turn("c1", "analyze_trend", {"metric": "revenue"}), final_turn("趋势分析完成。")],
    )

    response = run(client, default_bundle, new_goal, context=context)

    assert [step.tool_name for step in response.steps] == ["analyze_trend"]
    assert response.skills_used == ("business-diagnosis",)
    assert "pending_clarification" not in context


def test_planning_failure_keeps_existing_agent_fallback(default_bundle):
    client = PlanningClient(
        [DeepSeekError("planning unavailable")],
        [tool_turn("c1", "analyze_product", {"metric": "quantity"}), final_turn("商品分析完成。")],
    )

    response = run(client, default_bundle, "哪个商品销量最高？")

    assert response.answer == "商品分析完成。"
    assert [step.tool_name for step in response.steps] == ["analyze_product"]


def test_planning_failure_preserves_pending_user_supplement(default_bundle):
    context = {
        "pending_clarification": {
            "original_question": "帮我对比一下这两个",
            "supplements": [],
            "attempts": 1,
        }
    }
    client = PlanningClient(
        [DeepSeekError("planning unavailable")],
        [tool_turn("c1", "analyze_product", {"metric": "quantity"}), final_turn("完成。")],
    )

    run(
        client,
        default_bundle,
        "原味欧包和全麦欧包",
        context=context,
    )

    payload = json.loads(client.agent_requests[0][1]["content"])
    assert payload["user_goal"] == (
        "帮我对比一下这两个\n用户补充：原味欧包和全麦欧包"
    )
    assert "pending_clarification" not in context


def test_pending_clarification_is_isolated_by_database(default_bundle):
    state = {}
    initialize_database_state(state)
    content = (PROJECT_ROOT / "data" / "orders.csv").read_bytes()
    first = create_database(state, "A", content)
    second = create_database(state, "B", content)
    client = PlanningClient([
        AgentRequestDecision(
            "clarify", "", clarification_question="你指的是哪两个商品？"
        )
    ])

    run(
        client,
        default_bundle,
        "帮我比较这两个",
        database_id=first.database_id,
        context=first.conversation_context,
    )

    assert "pending_clarification" in first.conversation_context
    assert "pending_clarification" not in second.conversation_context


def test_deepseek_request_planner_parses_structured_clarification():
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            content = json.dumps({
                "action": "clarify",
                "resolved_question": "",
                "skill": None,
                "ambiguity_type": "referent",
                "clarification_question": "你想比较哪两个商品？",
                "options": ["商品 A 和 B", "商品 C 和 D"],
                "reason": "比较对象不明确",
                "uses_pending": False,
            }, ensure_ascii=False)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions = FakeCompletions()
    client = DeepSeekClient(
        DeepSeekConfig(api_key="mock"),
        sdk_client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    decision = client.plan_request(
        "帮我对比一下这两个",
        skills=[{"name": "business-diagnosis", "description": "diagnose"}],
        clarification_instructions="仅在关键歧义时澄清。",
        conversation_context={},
        capabilities=None,
        available_fields=("product", "quantity"),
    )

    assert decision.action == "clarify"
    assert decision.clarification_question == "你想比较哪两个商品？"
    assert decision.options == ("商品 A 和 B", "商品 C 和 D")
    assert decision.ambiguity_type == "referent"
    assert not decision.uses_pending
    system_prompt = completions.kwargs["messages"][0]["content"]
    assert "一个现有分析工具能综合返回" in system_prompt
    assert "数据缺少成本、渠道" in system_prompt


def test_clarification_never_displays_unverified_claims(default_bundle):
    context = {}
    client = PlanningClient([
        AgentRequestDecision(
            "clarify",
            "",
            clarification_question="销售额下降 32%，你想分析为什么吗？",
            options=(
                "销售额下降 32%",
                "按 ¥999 计算",
                "原味欧包 603 元",
                "原味欧包",
            ),
            ambiguity_type="object",
        )
    ])

    response = run(client, default_bundle, "帮我看看它", context=context)

    assert response.answer.startswith("你希望分析哪个对象？")
    assert "32%" not in response.answer
    assert "¥999" not in response.answer
    assert "603 元" not in response.answer
    assert "原味欧包" in response.answer
    assert response.clarification_options == ("原味欧包",)


def test_planner_cannot_rewrite_clear_user_goal(default_bundle):
    question = "哪个商品销量最高？"
    client = PlanningClient(
        [AgentRequestDecision("continue", "改成分析利润最低的商品")],
        [tool_turn("c1", "analyze_product", {"metric": "quantity"}), final_turn("完成。")],
    )

    run(client, default_bundle, question)

    payload = json.loads(client.agent_requests[0][1]["content"])
    assert payload["user_goal"] == question


def test_repeated_clarification_preserves_user_supplements(default_bundle):
    context = {
        "pending_clarification": {
            "original_question": "帮我比较一下",
            "supplements": ["比较两个商品"],
            "attempts": 1,
        }
    }
    client = PlanningClient([
        AgentRequestDecision(
            "clarify",
            "",
            ambiguity_type="metric",
            options=("销量", "销售额"),
        )
    ])

    run(client, default_bundle, "原味欧包和全麦欧包", context=context)

    pending = context["pending_clarification"]
    assert pending["original_question"] == "帮我比较一下"
    assert pending["supplements"] == ["比较两个商品", "原味欧包和全麦欧包"]
    assert pending["attempts"] == 2


def test_clarification_limit_clears_stale_pending_state(default_bundle):
    context = {
        "pending_clarification": {
            "original_question": "帮我比较一下",
            "supplements": ["两个商品", "按表现", "最近"],
            "attempts": 3,
        }
    }
    client = PlanningClient([
        AgentRequestDecision("clarify", "", ambiguity_type="object")
    ])

    response = run(client, default_bundle, "还是那个", context=context)

    assert response.needs_clarification
    assert response.stop_reason == "clarification_limit"
    assert "重新完整描述" in response.answer
    assert "pending_clarification" not in context


def test_agent_failure_uses_controlled_resolved_goal_and_clears_pending(default_bundle):
    class FailingClient(PlanningClient):
        def __init__(self):
            super().__init__([
                AgentRequestDecision(
                    "continue", "模型改写", uses_pending=True
                )
            ])
            self.legacy_question = None

        def agent_completion(self, messages, *, tools):
            raise DeepSeekError("agent unavailable")

        def parse_intent(self, question, *, conversation_context, capabilities):
            self.legacy_question = question
            return IntentPlan("product", metric="quantity")

        def generate_answer(self, question, **kwargs):
            return "已安全回退。"

    context = {
        "pending_clarification": {
            "original_question": "帮我对比一下这两个",
            "supplements": [],
            "attempts": 1,
        }
    }
    client = FailingClient()

    response = run(
        client,
        default_bundle,
        "原味欧包和全麦欧包",
        context=context,
    )

    assert response.fallback_used
    assert client.legacy_question == (
        "帮我对比一下这两个\n用户补充：原味欧包和全麦欧包"
    )
    assert "pending_clarification" not in context
