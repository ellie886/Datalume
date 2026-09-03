from datalume.ai_agent import analyze_question, generate_ai_answer
from datalume.router import QuestionType, classify_question


def test_default_core_metrics(default_bundle):
    _, _, overview, metrics, _ = default_bundle
    assert metrics.total_sales == 464
    assert metrics.order_count == 14
    assert overview.user_count == 14
    assert metrics.top_product == "原味欧包"


def test_default_user_metrics(default_bundle):
    _, _, _, metrics, _ = default_bundle
    assert metrics.top_gender == "女"
    assert metrics.top_grade == "大一"
    assert metrics.product_sales["原味欧包"] == 240


def test_legacy_question_routes():
    assert classify_question("哪个商品销量最高？") is QuestionType.PRODUCT
    assert classify_question("哪些用户值得重点维护？") is QuestionType.USER
    assert classify_question("现在应该怎么提升销售额？") is QuestionType.STRATEGY


def test_legacy_answers(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    assert "原味欧包" in generate_ai_answer("哪个商品销量最高？", metrics, orders=orders, capabilities=caps)
    assert "大一" in generate_ai_answer("哪些用户值得重点维护？", metrics, orders=orders, capabilities=caps)
    assert "建议" in generate_ai_answer("现在应该怎么提升销售额？", metrics, orders=orders, capabilities=caps)


def test_contextual_followups(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    history = [{"role": "user", "content": "哪个商品销量最高？"}, {"role": "assistant", "content": "原味欧包"}]
    why = analyze_question("为什么？", metrics, history, orders, caps)
    history.extend([{"role": "user", "content": "为什么？"}, {"role": "assistant", "content": why.answer}])
    promotion = analyze_question("那应该怎么推广它？", metrics, history, orders, caps)
    assert why.question_type is QuestionType.PRODUCT
    assert "¥240.00" in why.answer
    assert "原味欧包" in promotion.answer
    assert "推广方案" in promotion.answer


def test_current_explicit_question_beats_history(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    history = [{"role": "user", "content": "哪些用户值得维护？"}, {"role": "assistant", "content": "大一"}]
    answer = analyze_question("最近一周销售额为什么下降了？", metrics, history, orders, caps)
    assert answer.question_type is QuestionType.TREND
