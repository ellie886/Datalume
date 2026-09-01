from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_agent import analyze_question
from database_manager import create_database, initialize_database_state
from deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError, IntentPlan
from router import QuestionType
from visualization import build_visualization


class MockDeepSeekClient:
    def __init__(self, plans, answers=None, *, fail_parse=False, fail_answer=False):
        self.plans = list(plans)
        self.answers = list(answers or ["自然语言回答"] * len(self.plans))
        self.fail_parse = fail_parse
        self.fail_answer = fail_answer
        self.received_results = []
        self.received_contexts = []

    def parse_intent(self, question, *, conversation_context, capabilities):
        if self.fail_parse:
            raise DeepSeekError("mock failure")
        self.received_contexts.append(dict(conversation_context))
        return self.plans.pop(0)

    def generate_answer(self, question, *, plan, result, conversation_context, capabilities):
        if self.fail_answer:
            raise DeepSeekError("mock failure")
        self.received_results.append(result)
        return self.answers.pop(0)


def product_plan(metric="quantity", action="analyze", follow_up=False, visualization=True, entity=None):
    return IntentPlan("product", metric, action, visualization, follow_up, None, entity, action == "explain")


def test_llm_intent_drives_real_pandas_result(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient([product_plan()], ["原味欧包销量最高，共 16 件。"])
    context = {}
    analysis = analyze_question(
        "哪个商品销量最高？", metrics, orders=orders, capabilities=caps,
        llm_client=client, conversation_context=context,
    )
    assert analysis.question_type is QuestionType.PRODUCT
    assert analysis.answer == "原味欧包销量最高，共 16 件。"
    assert client.received_results[0].metrics["product_quantity"]["原味欧包"] == 16
    assert context["previous_entity"] == "原味欧包"


def test_visual_follow_up_inherits_product_quantity(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient(
        [product_plan(), product_plan(None, "visualize", True, True)],
        ["原味欧包销量最高。", "可以，下面是各商品销量对比。"],
    )
    context = {}
    analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    follow_up = analyze_question("帮我画一下", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    assert follow_up.plan.follow_up
    assert follow_up.plan.metric == "quantity"
    assert follow_up.plan.action == "visualize"
    assert follow_up.plan.visualization
    assert follow_up.result.metrics["product_quantity"]["原味欧包"] == 16
    assert build_visualization(
        follow_up.result.intent,
        orders,
        follow_up.result,
        caps,
        metric=follow_up.plan.metric,
        action=follow_up.plan.action,
        visualization=follow_up.plan.visualization,
    ) is not None


def test_follow_up_can_switch_metric_to_revenue(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient(
        [product_plan(), product_plan("revenue", "analyze", True, True)],
        ["销量回答", "销售额回答"],
    )
    context = {}
    analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    answer = analyze_question("那销售额呢？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    assert answer.plan.intent == "product"
    assert answer.plan.metric == "revenue"
    assert answer.result.metrics["product_revenue"]["原味欧包"] == 240


def test_entity_is_inherited_for_reasoning_follow_up(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient(
        [product_plan(), product_plan(None, "explain", True, False)],
        ["销量回答", "原因回答"],
    )
    context = {}
    analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    answer = analyze_question("为什么它卖得最好？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=context)
    assert answer.plan.entity == "原味欧包"
    assert answer.plan.reasoning_request
    assert client.received_contexts[-1]["previous_entity"] == "原味欧包"


def test_llm_api_failure_falls_back_without_crashing(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient([product_plan()], fail_parse=True)
    answer = analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context={})
    assert answer.fallback_used
    assert answer.question_type is QuestionType.PRODUCT
    assert "原味欧包" in answer.answer


def test_invalid_llm_json_raises_and_orchestrator_falls_back(default_bundle):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
            )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = DeepSeekClient(DeepSeekConfig(api_key=""), sdk_client=sdk)
    with pytest.raises(DeepSeekError):
        client.parse_intent("问题", conversation_context={}, capabilities=None)

    orders, _, _, metrics, caps = default_bundle
    answer = analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context={})
    assert answer.fallback_used
    assert "原味欧包" in answer.answer


def test_database_conversation_contexts_are_isolated(rich_csv, default_bundle):
    state = {}
    initialize_database_state(state)
    campus = create_database(state, "校园团购", rich_csv)
    recruitment = create_database(state, "秋招", rich_csv)
    orders, _, _, metrics, caps = default_bundle
    client = MockDeepSeekClient([product_plan(), IntentPlan("user", "users")], ["商品回答", "用户回答"])
    analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=campus.conversation_context)
    analyze_question("哪些用户值得维护？", metrics, orders=orders, capabilities=caps, llm_client=client, conversation_context=recruitment.conversation_context)
    assert campus.conversation_context["previous_intent"] == "product"
    assert recruitment.conversation_context["previous_intent"] == "user"
    assert recruitment.conversation_context.get("previous_entity") != "原味欧包"
