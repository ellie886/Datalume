from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_models import AgentModelTurn, AgentToolCall
from agent_runtime import AgentConfig, run_agent
from agent_tools import AgentToolRegistry
from ai_agent import analyze_question
from database_manager import create_database, initialize_database_state
from data_processor import calculate_metrics, detect_data_capabilities, process_order_csv, summarize_orders
from deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekError, IntentPlan
from visualization import build_visualization


def tool_turn(call_id, name, arguments=None):
    call = AgentToolCall(call_id, name, arguments or {})
    return AgentModelTurn(
        None, (call,),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {}, ensure_ascii=False)},
            }],
        },
    )


def final_turn(content="分析完成。"):
    return AgentModelTurn(content, (), {"role": "assistant", "content": content})


class ScriptedClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def agent_completion(self, messages, *, tools):
        self.requests.append({"messages": [dict(item) for item in messages], "tools": tools})
        return self.turns.pop(0)


def run(client, bundle, question="哪个商品销量最高？", *, database_id="db-a", context=None, max_steps=4):
    orders, _, overview, metrics, caps = bundle
    return run_agent(
        question, database_id=database_id, orders=orders, metrics=metrics,
        capabilities=caps, overview=overview, llm_client=client,
        conversation_context=context if context is not None else {},
        config=AgentConfig(True, max_steps),
    )


def rich_bundle(rich_csv):
    orders, quality = process_order_csv(rich_csv)
    return orders, quality, summarize_orders(orders), calculate_metrics(orders), detect_data_capabilities(orders)


def test_single_tool_uses_real_analysis_result(default_bundle):
    client = ScriptedClient([
        tool_turn("c1", "analyze_product", {"metric": "quantity", "top_n": 5}),
        final_turn("原味欧包销量最高，共 16 件。"),
    ])
    response = run(client, default_bundle)
    assert [step.tool_name for step in response.steps] == ["analyze_product"]
    assert response.primary_result.metrics["product_quantity"]["原味欧包"] == 16
    assert response.answer == "原味欧包销量最高，共 16 件。"
    assert response.stop_reason == "final_answer"


def test_multi_tool_calls_execute_in_model_selected_order(rich_csv):
    bundle = rich_bundle(rich_csv)
    client = ScriptedClient([
        tool_turn("c1", "analyze_trend", {"metric": "revenue", "time_range": "recent_week"}),
        tool_turn("c2", "analyze_product", {"metric": "revenue", "top_n": 5}),
        tool_turn("c3", "analyze_channel", {"metric": "revenue"}),
        final_turn("趋势、商品和渠道分析已完成。"),
    ])
    response = run(client, bundle, "最近销售为什么下降？从商品和渠道两个角度看看。")
    assert [step.tool_name for step in response.steps] == [
        "analyze_trend", "analyze_product", "analyze_channel"
    ]
    assert len(client.requests) == 4
    assert any(message.get("role") == "tool" for message in client.requests[1]["messages"])


def test_observation_can_drive_next_tool(rich_csv):
    bundle = rich_bundle(rich_csv)

    class ObservationDrivenClient:
        def __init__(self):
            self.calls = 0

        def agent_completion(self, messages, *, tools):
            self.calls += 1
            if self.calls == 1:
                return tool_turn("c1", "analyze_trend", {"metric": "revenue"})
            if self.calls == 2:
                observations = [item for item in messages if item.get("role") == "tool"]
                assert observations and "recent_orders" in observations[-1]["content"]
                return tool_turn("c2", "analyze_channel", {"metric": "orders"})
            return final_turn("根据趋势 observation，继续检查了渠道订单。")

    response = run(ObservationDrivenClient(), bundle, "最近销售为什么下降？")
    assert [step.tool_name for step in response.steps] == ["analyze_trend", "analyze_channel"]


def test_visualization_tool_uses_existing_visualization_builder(default_bundle):
    client = ScriptedClient([
        tool_turn("c1", "analyze_product", {"metric": "quantity"}),
        tool_turn("c2", "create_visualization", {
            "analysis_type": "product", "metric": "quantity", "chart_type": "bar"
        }),
        final_turn("已生成商品销量图。"),
    ])
    response = run(client, default_bundle, "哪个商品销量最高？帮我画一下。")
    assert len(response.visualizations) == 1
    spec = response.visualizations[0]
    orders, _, _, _, caps = default_bundle
    chart = build_visualization(
        response.primary_result.intent, orders, response.primary_result, caps,
        metric=spec.metric, action="visualize", visualization=True,
    )
    assert chart is not None


def test_unsupported_profit_never_uses_invented_model_answer(default_bundle):
    client = ScriptedClient([
        tool_turn("c1", "analyze_profit", {"metric": "profit"}),
        final_turn("虚构商品利润是 999 元。"),
    ])
    response = run(client, default_bundle, "哪个商品利润最高？")
    assert not response.primary_result.supported
    assert "缺少成本" in response.answer
    assert "999" not in response.answer


def test_max_steps_stops_without_infinite_loop(default_bundle):
    client = ScriptedClient([
        tool_turn(f"c{index}", "analyze_product", {"metric": "quantity"})
        for index in range(1, 7)
    ])
    response = run(client, default_bundle, max_steps=4)
    assert len(response.steps) == 4
    assert response.stop_reason == "max_steps"
    assert "原味欧包" in response.answer


def test_agent_api_failure_falls_back_to_single_step(default_bundle):
    class FailingAgentClient:
        def agent_completion(self, messages, *, tools):
            raise DeepSeekError("tool API unavailable")

        def parse_intent(self, question, *, conversation_context, capabilities):
            return IntentPlan("product", "quantity")

        def generate_answer(self, question, *, plan, result, conversation_context, capabilities):
            return "标准模式：原味欧包 16 件。"

    response = run(FailingAgentClient(), default_bundle)
    assert response.fallback_used
    assert response.fallback_level == "single_step"
    assert response.primary_result.metrics["product_quantity"]["原味欧包"] == 16


def test_agent_and_single_step_failure_reaches_deterministic(default_bundle):
    class FullyFailingClient:
        def agent_completion(self, messages, *, tools):
            raise DeepSeekError("agent failed")

        def parse_intent(self, question, *, conversation_context, capabilities):
            raise DeepSeekError("single failed")

    response = run(FullyFailingClient(), default_bundle)
    assert response.fallback_level == "deterministic"
    assert "原味欧包" in response.answer


def test_unknown_tool_is_not_executed(default_bundle):
    client = ScriptedClient([
        tool_turn("c1", "delete_database", {}),
        final_turn("数据库已删除。"),
    ])
    response = run(client, default_bundle)
    assert response.steps[0].error == "unknown_tool"
    assert "不在 Datalume 白名单" in response.answer
    assert "已删除" not in response.answer


def test_database_agent_context_and_history_are_isolated(default_bundle):
    state = {}
    initialize_database_state(state)
    content = open("data/orders.csv", "rb").read()
    first = create_database(state, "A", content)
    second = create_database(state, "B", content)
    client_a = ScriptedClient([tool_turn("a1", "analyze_product", {"metric": "quantity"}), final_turn("A")])
    client_b = ScriptedClient([tool_turn("b1", "analyze_user", {"metric": "users"}), final_turn("B")])
    response_a = run(client_a, default_bundle, database_id=first.database_id, context=first.conversation_context)
    response_b = run(client_b, default_bundle, "哪些用户值得维护？", database_id=second.database_id, context=second.conversation_context)
    first.agent_history.append(response_a.public_trace())
    second.agent_history.append(response_b.public_trace())
    assert first.conversation_context["last_tools_used"] == ["analyze_product"]
    assert second.conversation_context["last_tools_used"] == ["analyze_user"]
    assert first.agent_history != second.agent_history


def test_payload_never_contains_raw_csv(default_bundle):
    client = ScriptedClient([
        tool_turn("c1", "analyze_product", {"metric": "quantity"}),
        final_turn("完成"),
    ])
    run(client, default_bundle)
    payload = json.dumps(client.requests, ensure_ascii=False, default=str)
    assert "csv_content" not in payload
    assert "date,user_id,product,price,quantity" not in payload
    assert "DataFrame" not in payload


def test_generate_report_tool_is_deterministic(default_bundle):
    client = ScriptedClient([tool_turn("c1", "generate_report", {}), final_turn("报告已生成。")])
    response = run(client, default_bundle, "帮我生成一份经营报告。")
    assert response.report is not None
    assert response.report.total_sales == 464
    assert response.report.to_markdown().startswith("# Datalume AI 经营诊断报告")


def test_feature_flag_disabled_keeps_old_flow(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    response = run_agent(
        "哪个商品销量最高？", database_id="db", orders=orders, metrics=metrics,
        capabilities=caps, overview=overview, llm_client=None,
        conversation_context={}, config=AgentConfig(False, 4),
    )
    legacy = analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps)
    assert response.primary_result.metrics == legacy.result.metrics
    assert response.stop_reason == "agent_disabled"


def test_tool_registry_contains_only_safe_whitelist():
    names = {schema["function"]["name"] for schema in AgentToolRegistry().schemas()}
    assert names == {
        "analyze_product", "analyze_user", "analyze_repurchase",
        "analyze_profit", "analyze_channel", "analyze_trend",
        "analyze_strategy", "create_visualization", "generate_report",
    }
    assert not names & {"shell", "python", "delete_database", "http_request"}


def test_deepseek_client_parses_openai_compatible_tool_calls():
    raw_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="analyze_product", arguments='{"metric":"quantity"}'),
    )

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content=None, tool_calls=[raw_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = FakeCompletions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = DeepSeekClient(DeepSeekConfig(api_key="mock"), sdk_client=sdk)
    turn = client.agent_completion([{"role": "user", "content": "问题"}], tools=AgentToolRegistry().schemas())
    assert turn.tool_calls[0].name == "analyze_product"
    assert turn.tool_calls[0].arguments == {"metric": "quantity"}
    assert completions.kwargs["tool_choice"] == "auto"
    assert "tools" in completions.kwargs


def test_malformed_tool_arguments_raise_safe_client_error():
    raw_call = SimpleNamespace(
        id="call-bad",
        function=SimpleNamespace(name="analyze_product", arguments="not-json"),
    )

    class FakeCompletions:
        def create(self, **kwargs):
            message = SimpleNamespace(content=None, tool_calls=[raw_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client = DeepSeekClient(DeepSeekConfig(api_key="mock"), sdk_client=sdk)
    with pytest.raises(DeepSeekError, match="参数无法解析"):
        client.agent_completion([], tools=AgentToolRegistry().schemas())
