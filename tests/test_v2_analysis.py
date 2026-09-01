import pytest

from ai_agent import analyze_question
from data_processor import calculate_metrics, detect_data_capabilities, process_order_csv
from router import QuestionType, classify_question


def bundle(content):
    orders, _ = process_order_csv(content)
    return orders, calculate_metrics(orders), detect_data_capabilities(orders)


def test_v2_router_intents():
    assert classify_question("销量最高的商品利润也最高吗？") is QuestionType.PROFIT
    assert classify_question("哪个获客渠道效果最好？") is QuestionType.CHANNEL
    assert classify_question("哪些用户的复购率最高？") is QuestionType.REPURCHASE
    assert classify_question("最近一周销售额为什么下降了？") is QuestionType.TREND


def test_profit_missing_cost_refuses(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("销量最高的商品利润也最高吗？", metrics, orders=orders, capabilities=caps)
    assert not result.result.supported
    assert "缺少成本" in result.answer
    assert "等同" in result.answer


def test_profit_is_really_calculated(rich_csv):
    orders, metrics, caps = bundle(rich_csv)
    result = analyze_question("哪个商品利润最高？", metrics, orders=orders, capabilities=caps).result
    assert result.supported
    assert result.metrics["product_profit"]["欧包"]["profit"] == pytest.approx(168)
    assert result.metrics["product_profit"]["酸奶"]["profit"] == pytest.approx(70)


def test_channel_missing_refuses(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("哪个获客渠道效果最好？", metrics, orders=orders, capabilities=caps).result
    assert not result.supported
    assert "缺少渠道" in result.conclusion


def test_channel_analysis_uses_real_metrics(rich_csv):
    orders, metrics, caps = bundle(rich_csv)
    result = analyze_question("哪个获客渠道效果最好？", metrics, orders=orders, capabilities=caps).result
    assert result.supported
    assert set(result.metrics["channel_summary"]) == {"社群", "海报"}
    assert "CAC" in result.to_markdown()


def test_repurchase_counts_users(rich_csv):
    orders, metrics, caps = bundle(rich_csv)
    result = analyze_question("哪些用户的复购率最高？", metrics, orders=orders, capabilities=caps).result
    assert result.supported
    assert result.metrics["repeat_users"] == 12
    assert result.metrics["repeat_user_rate"] == 1
    assert result.metrics["reliable"] is True


def test_default_repurchase_states_order_limitation(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("复购率怎么样？", metrics, orders=orders, capabilities=caps).result
    assert result.supported
    assert result.metrics["reliable"] is False
    assert any("order_id" in item for item in result.limitations)


def test_trend_compares_two_weeks(rich_csv):
    orders, metrics, caps = bundle(rich_csv)
    result = analyze_question("最近一周销售额为什么下降了？", metrics, orders=orders, capabilities=caps).result
    assert result.supported
    assert result.metrics["recent_orders"] == 7
    assert result.metrics["previous_orders"] == 7
    assert "订单量" in result.to_markdown()


def test_trend_short_cycle_refuses(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("最近一周销售额为什么下降了？", metrics, orders=orders, capabilities=caps).result
    assert not result.supported
    assert "不足" in result.conclusion


def test_optional_capabilities_detected(rich_csv):
    orders, _, caps = bundle(rich_csv)
    assert caps.repurchase_reliable
    assert caps.profit_supported
    assert caps.channel_supported
    assert caps.channel_cost_supported
