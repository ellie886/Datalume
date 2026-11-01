import pytest

from datalume.database_manager import create_database, delete_database, get_active_database, initialize_database_state, rename_database, replace_database_data, search_databases, switch_database, toggle_database_pin
from datalume.data_processor import calculate_metrics, detect_data_capabilities, process_order_csv
from datalume.ai_agent import analyze_question
from datalume.analysis_engine import product_analysis
from datalume.router import QuestionType
from datalume.visualization import build_visualization, visualization_kind
from datalume.deepseek_client import IntentPlan
from datalume.ui_components import (
    _answer_html,
    _meaningful_diagnoses,
    _wants_visualization,
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


def test_rename_database_preserves_identity_and_all_project_state(rich_csv):
    state = {}
    initialize_database_state(state)
    project = create_database(state, "校园团购", rich_csv, pinned=True)
    project.chat_messages.append({"role": "user", "content": "问题"})
    project.conversation_context["previous_intent"] = "trend"
    project.agent_history.append({"tools": ["analyze_trend"]})
    project.diagnostic_report = {"report": "保留"}
    project.diagnostic_report_source_id = project.data_source_id
    original_id = project.database_id
    original_source = project.data_source_id

    renamed = rename_database(state, original_id, "  校园零售经营数据  ")

    assert renamed is project
    assert renamed.name == "校园零售经营数据"
    assert state["active_database_id"] == original_id
    assert state["databases"][original_id] is project
    assert project.pinned is True
    assert project.csv_content == rich_csv
    assert project.data_source_id == original_source
    assert project.chat_messages == [{"role": "user", "content": "问题"}]
    assert project.conversation_context == {"previous_intent": "trend"}
    assert project.agent_history == [{"tools": ["analyze_trend"]}]
    assert project.diagnostic_report == {"report": "保留"}
    assert project.diagnostic_report_source_id == original_source


def test_rename_database_rejects_empty_and_duplicate_names(rich_csv):
    state = {}
    initialize_database_state(state)
    first = create_database(state, "校园团购", rich_csv)
    second = create_database(state, "秋招", rich_csv)
    with pytest.raises(ValueError, match="不能为空"):
        rename_database(state, second.database_id, "   ")
    with pytest.raises(ValueError, match="同名"):
        rename_database(state, second.database_id, "  校园团购 ")
    with pytest.raises(ValueError, match="同名"):
        rename_database(state, second.database_id, "校园团购".upper())
    assert first.name == "校园团购"
    assert second.name == "秋招"
    assert state["active_database_id"] == second.database_id


def test_delete_non_active_database_keeps_active_project(rich_csv):
    state = {}
    initialize_database_state(state)
    first = create_database(state, "校园团购", rich_csv)
    second = create_database(state, "秋招", rich_csv)
    active = delete_database(state, first.database_id)
    assert active is second
    assert state["active_database_id"] == second.database_id
    assert set(state["databases"]) == {second.database_id}


def test_delete_active_database_selects_pinned_before_other_databases(rich_csv):
    state = {}
    initialize_database_state(state)
    pinned = create_database(state, "置顶项目", rich_csv, pinned=True)
    create_database(state, "普通项目", rich_csv)
    active = create_database(state, "待删除", rich_csv)
    selected = delete_database(state, active.database_id)
    assert selected is pinned
    assert state["active_database_id"] == pinned.database_id


def test_delete_last_database_clears_private_state_and_returns_to_empty(rich_csv):
    state = {}
    initialize_database_state(state)
    project = create_database(state, "唯一项目", rich_csv, pinned=True)
    project.chat_messages.append({"role": "user", "content": "问题"})
    project.conversation_context["previous_intent"] = "product"
    project.agent_history.append({"tools": ["analyze_product"]})
    project.diagnostic_report = "报告"
    project.diagnostic_report_source_id = project.data_source_id

    assert delete_database(state, project.database_id) is None
    assert state["databases"] == {}
    assert state["active_database_id"] is None
    assert project.csv_content == b""
    assert project.data_source_id == ""
    assert project.pinned is False
    assert project.chat_messages == []
    assert project.conversation_context == {}
    assert project.agent_history == []
    assert project.diagnostic_report is None
    assert project.diagnostic_report_source_id is None


def test_delete_pinned_database_removes_it_from_navigation(rich_csv):
    state = {}
    initialize_database_state(state)
    pinned = create_database(state, "置顶项目", rich_csv, pinned=True)
    remaining = create_database(state, "保留项目", rich_csv)
    delete_database(state, pinned.database_id)
    assert [project.database_id for project in search_databases(state)] == [
        remaining.database_id
    ]
    assert all(not project.pinned for project in state["databases"].values())


def test_visualization_mapping():
    assert visualization_kind(QuestionType.TREND) == "line"
    assert visualization_kind(QuestionType.PRODUCT) == "bar"
    assert visualization_kind(QuestionType.PROFIT) == "profit"
    assert visualization_kind(QuestionType.CHANNEL) == "channel"
    assert visualization_kind(QuestionType.REPURCHASE) == "repurchase"
    assert visualization_kind(QuestionType.USER) == "user"


def test_product_visualization_builds(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    result = analyze_question("哪个商品销量最高？", metrics, orders=orders, capabilities=caps).result
    assert build_visualization(result.intent, orders, result, caps) is not None


def test_unsupported_analysis_has_no_chart(default_bundle):
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


def test_answer_html_renders_safe_headings_lists_and_inline_code():
    rendered = _answer_html(
        "## 结论\n\n- **订单量**下降\n- 检查 `channel` 字段\n\n1. 优先分析渠道"
    )
    assert 'class="answer-heading"' in rendered
    assert "<ul>" in rendered and rendered.count("<li>") == 3
    assert "<strong>订单量</strong>" in rendered
    assert "<code>channel</code>" in rendered
    assert "<script" not in _answer_html("<script>alert(1)</script>")


def test_visualization_request_detection_is_explicit():
    assert _wants_visualization("帮我画一张趋势图")
    assert _wants_visualization("把销量可视化")
    assert not _wants_visualization("哪个商品销量最高？")


def test_suggestions_are_only_shown_for_recommendation_context(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    product = product_analysis("哪个商品销量最高？", orders, metrics)
    assert not _should_show_suggestions(
        product, IntentPlan("product", "quantity", "analyze"), (("product", product),)
    )
    assert _should_show_suggestions(
        product, IntentPlan("product", "quantity", "recommend"), (("product", product),)
    )


def test_irrelevant_profit_caveat_is_hidden_outside_profit(default_bundle):
    orders, _, _, metrics, caps = default_bundle
    product = product_analysis("哪个商品销量最高？", orders, metrics)
    assert _visible_limitations((("product", product),)) == ()


def test_quick_questions_follow_current_data_capability(default_bundle, rich_csv):
    default_orders, _, _, _, default_caps = default_bundle
    default_questions = _quick_questions(default_orders, default_caps)
    assert not any("最近一周" in question for question in default_questions)

    rich_orders, _ = process_order_csv(rich_csv)
    rich_questions = _quick_questions(rich_orders, detect_data_capabilities(rich_orders))
    assert any("最近一周" in question for question in rich_questions)
