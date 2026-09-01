from database_manager import create_database, get_active_database, initialize_database_state, replace_database_data, search_databases, switch_database, toggle_database_pin
from data_processor import calculate_metrics, detect_data_capabilities, process_order_csv
from router import QuestionType
from visualization import build_visualization, visualization_kind
from deepseek_client import IntentPlan
from ui_components import (
    _answer_html,
    _meaningful_diagnoses,
    _should_show_suggestions,
    _visible_limitations,
)
from app import _quick_questions


def test_database_creation_and_switch_isolated(rich_csv):
    state = {}
    initialize_database_state(state)
    first = create_database(state, "校园团购", b"default")
    first.chat_messages.append({"role": "user", "content": "旧问题"})
    second = create_database(state, "新数据库", rich_csv)
    second.chat_messages.append({"role": "user", "content": "新问题"})
    switch_database(state, first.database_id)
    assert get_active_database(state).chat_messages[0]["content"] == "旧问题"
    switch_database(state, second.database_id)
    assert get_active_database(state).chat_messages[0]["content"] == "新问题"


def test_database_reports_are_isolated(rich_csv):
    state = {}
    initialize_database_state(state)
    first = create_database(state, "校园团购", b"default")
    first.diagnostic_report = "报告一"
    second = create_database(state, "二", rich_csv)
    assert second.diagnostic_report is None
    assert first.diagnostic_report == "报告一"


def test_replacing_data_clears_only_current_project(rich_csv):
    state = {}
    initialize_database_state(state)
    project = create_database(state, "校园团购", b"default")
    project.chat_messages.append({"role": "user", "content": "问题"})
    project.agent_history.append({"tools": ["analyze_product"]})
    project.diagnostic_report = "报告"
    assert replace_database_data(project, rich_csv)
    assert project.chat_messages == []
    assert project.agent_history == []
    assert project.diagnostic_report is None


def test_same_data_does_not_clear(rich_csv):
    state = {}
    initialize_database_state(state)
    project = create_database(state, "校园团购", rich_csv)
    project.chat_messages.append({"role": "user", "content": "保留"})
    assert not replace_database_data(project, rich_csv)
    assert project.chat_messages


def test_pin_toggle(rich_csv):
    state = {}
    initialize_database_state(state)
    project = create_database(state, "校园团购", rich_csv)
    before = project.pinned
    assert toggle_database_pin(state, project.database_id) is (not before)
    assert search_databases(state)[0].database_id == project.database_id
    assert toggle_database_pin(state, project.database_id) is before


def test_initial_database_state_is_truly_empty():
    state = {}
    initialize_database_state(state)
    assert state["databases"] == {}
    assert state["active_database_id"] is None
    assert get_active_database(state) is None


def test_created_databases_and_search_use_real_names(rich_csv):
    state = {}
    initialize_database_state(state)
    campus = create_database(state, "校园团购", rich_csv)
    recruitment = create_database(state, "秋招", rich_csv)
    assert set(state["databases"]) == {campus.database_id, recruitment.database_id}
    assert [project.name for project in search_databases(state, "秋")] == ["秋招"]
    assert {project.name for project in search_databases(state, "")} == {"校园团购", "秋招"}


def test_visualization_mapping():
    assert visualization_kind(QuestionType.TREND) == "line"
    assert visualization_kind(QuestionType.PRODUCT) == "bar"
    assert visualization_kind(QuestionType.PROFIT) == "profit"
    assert visualization_kind(QuestionType.CHANNEL) == "channel"
    assert visualization_kind(QuestionType.REPURCHASE) == "repurchase"
    assert visualization_kind(QuestionType.USER) == "user"


def test_product_visualization_builds(default_bundle):
    from ai_agent import analyze_question
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps).result
    assert build_visualization(result.intent, orders, result, caps) is not None


def test_unsupported_analysis_has_no_chart(default_bundle):
    from ai_agent import analyze_question
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("哪个渠道最好？", metrics, orders=orders, capabilities=caps).result
    assert build_visualization(result.intent, orders, result, caps) is None


def test_meaningless_unchanged_diagnoses_are_filtered():
    diagnoses = _meaningful_diagnoses(
        (
            "主力商品由 原味欧包 变为 原味欧包；主要年级由 大一 变为 大二。",
            "主渠道由 社群 变为 社群。",
            "订单量下降 20%。",
        )
    )
    assert diagnoses == ("主要年级由 大一 变为 大二。", "订单量下降 20%。")


def test_answer_html_keeps_natural_language_in_one_text_flow():
    rendered = _answer_html("最近一周销售额下降，主要由订单量减少导致。\n\n客单价相对稳定。")
    assert rendered.count("<p>") == 2
    assert "st.columns" not in rendered
    assert "最近一周销售额下降" in rendered


def test_suggestions_are_only_shown_for_recommendation_context(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    product = __import__("analysis_engine").product_analysis("哪个商品销量最高？", orders, metrics)
    assert not _should_show_suggestions(
        product, IntentPlan("product", "quantity", "analyze"), (("product", product),)
    )
    assert _should_show_suggestions(
        product, IntentPlan("product", "quantity", "recommend"), (("product", product),)
    )


def test_irrelevant_profit_caveat_is_hidden_outside_profit(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    product = __import__("analysis_engine").product_analysis("哪个商品销量最高？", orders, metrics)
    assert _visible_limitations((("product", product),)) == ()


def test_quick_questions_follow_current_data_capability(default_bundle, rich_csv):
    default_orders, _, _, _, default_caps = default_bundle
    default_questions = _quick_questions(default_orders, default_caps)
    assert not any("最近一周" in question for question in default_questions)

    rich_orders, _ = process_order_csv(rich_csv)
    rich_questions = _quick_questions(rich_orders, detect_data_capabilities(rich_orders))
    assert any("最近一周" in question for question in rich_questions)
