from pathlib import Path

from streamlit.testing.v1 import AppTest

from datalume.agent_models import AgentResponse, AgentStep, VisualizationSpec
from datalume.analysis_engine import run_analysis
from datalume.database_manager import create_database
from datalume.data_processor import calculate_metrics, detect_data_capabilities, process_order_csv
from datalume.deepseek_client import IntentPlan
from datalume.router import QuestionType


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _vega_charts(app):
    """兼容 Streamlit AppTest 新旧 Vega-Lite 元素名称。"""

    return app.get("vega_lite_chart") or app.get("arrow_vega_lite_chart")


def test_streamlit_initial_view_runs():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    assert len(app.chat_input) == 0
    assert len(app.session_state["databases"]) == 0
    assert not any("从数据开始" in item.value for item in app.markdown)
    assert any(button.label == "新建数据库" for button in app.sidebar.button)
    assert not any(button.label == "新建数据库" for button in app.main.button)
    assert not any(button.label == "生成 AI 经营诊断报告" for button in app.button)
    assert not any("校园团购" in button.label for button in app.button)
    assert not app.expander


def test_streamlit_create_database_from_csv():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    next(button for button in app.button if button.label == "新建数据库").click()
    app.run()
    next(item for item in app.text_input if item.label == "数据库名称").input("校园团购")
    uploader = app.get("file_uploader")[0]
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    if hasattr(uploader, "upload"):
        uploader.upload("orders.csv", content, "text/csv")
        app.run()
        next(button for button in app.button if button.label == "创建数据库").click()
    else:
        # Streamlit 1.40 AppTest 尚不能驱动 file_uploader；直接注入同一 CSV
        # 后继续验证数据库创建后的完整 Streamlit 状态与导航。
        create_database(app.session_state, "校园团购", content)
    app.run()
    assert not app.exception
    assert len(app.session_state["databases"]) == 1
    assert any(button.label == "校园团购" for button in app.sidebar.button)
    assert not any(button.label == "校园团购" for button in app.main.button)
    assert len(app.chat_input) == 1


def test_streamlit_chat_chart_and_report_download():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    create_database(app.session_state, "校园团购", (APP_PATH.parent / "data" / "orders.csv").read_bytes())
    app.run()
    assert any(button.label == "校园团购" for button in app.button)
    assert len(app.chat_input) == 1
    next(button for button in app.button if button.label == "哪个商品销量最高？").click()
    app.run()
    assert not app.exception
    assert not _vega_charts(app)
    assert any("基础分析模式" in item.value for item in app.markdown)
    assert not any("AI 诊断" in item.value for item in app.markdown)
    assert not any("建议优先关注" in item.value for item in app.markdown)

    app.chat_input[0].set_value("哪个商品销量最高？帮我画出来。").run()
    assert not app.exception
    assert _vega_charts(app)
    next(button for button in app.button if button.label == "生成 AI 经营诊断报告").click()
    app.run()
    assert not app.exception
    assert any(button.label == "下载经营诊断报告" for button in app.get("download_button"))


def test_streamlit_group_creation_has_real_feedback():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    next(button for button in app.sidebar.button if button.label == "新建分组").click()
    app.run()
    group_name = next(item for item in app.text_input if item.label == "分组名称")
    group_name.input("本周项目")
    next(button for button in app.button if button.label == "创建分组").click()
    app.run()
    assert not app.exception
    assert app.session_state["database_groups"] == ["本周项目"]
    assert any("本周项目" in item.value for item in app.sidebar.markdown)


def test_workspace_toolbar_uses_one_clear_database_menu():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    create_database(app.session_state, "实验1", content)
    app.run()
    assert not app.exception
    source = APP_PATH.read_text(encoding="utf-8")
    assert '"更多", icon=":material/more_horiz:"' in source
    assert 'st.popover("⌄"' not in source


def test_streamlit_database_rename_updates_navigation_and_preserves_state():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    project = create_database(app.session_state, "校园团购", content, pinned=True)
    project.chat_messages.append({"role": "user", "content": "保留的问题"})
    project.conversation_context["previous_intent"] = "product"
    project.agent_history.append({"tools": ["analyze_product"]})
    project.diagnostic_report = "保留的报告"
    source_id = project.data_source_id
    app.run()

    next(button for button in app.button if button.label == "重命名").click()
    app.run()
    name_input = next(item for item in app.text_input if item.label == "数据库名称")
    assert name_input.value == "校园团购"
    name_input.input("  校园零售经营数据  ")
    app.run()
    next(button for button in app.button if button.label == "保存").click()
    app.run()

    assert not app.exception
    assert project.name == "校园零售经营数据"
    assert app.session_state["active_database_id"] == project.database_id
    assert project.pinned is True
    assert project.csv_content == content
    assert project.data_source_id == source_id
    assert project.chat_messages == [{"role": "user", "content": "保留的问题"}]
    assert project.conversation_context == {"previous_intent": "product"}
    assert project.agent_history == [{"tools": ["analyze_product"]}]
    assert project.diagnostic_report == "保留的报告"
    assert any(button.label == "校园零售经营数据" for button in app.sidebar.button)
    assert not any(button.label == "校园团购" for button in app.sidebar.button)


def test_streamlit_delete_requires_confirmation_and_selects_pinned_database():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    pinned = create_database(app.session_state, "置顶项目", content, pinned=True)
    active = create_database(app.session_state, "待删除项目", content)
    active.chat_messages.append({"role": "user", "content": "即将删除的问题"})
    active.conversation_context["previous_intent"] = "trend"
    active.agent_history.append({"tools": ["analyze_trend"]})
    active.diagnostic_report = "即将删除的报告"
    app.run()

    next(button for button in app.button if button.label == "删除数据库").click()
    app.run()
    assert active.database_id in app.session_state["databases"]
    assert any("删除“待删除项目”" in item.value for item in app.markdown)
    next(button for button in app.button if button.label == "删除").click()
    app.run()

    assert not app.exception
    assert active.database_id not in app.session_state["databases"]
    assert app.session_state["active_database_id"] == pinned.database_id
    assert active.csv_content == b""
    assert active.chat_messages == []
    assert active.conversation_context == {}
    assert active.agent_history == []
    assert active.diagnostic_report is None
    assert any(button.label == "置顶项目" for button in app.sidebar.button)
    assert not any("即将删除的问题" in item.value for item in app.markdown)


def test_streamlit_delete_last_database_returns_to_initial_empty_state():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    project = create_database(app.session_state, "最后一个项目", content)
    app.run()
    next(button for button in app.button if button.label == "删除数据库").click()
    app.run()
    next(button for button in app.button if button.label == "删除").click()
    app.run()

    assert not app.exception
    assert project.database_id not in app.session_state["databases"]
    assert app.session_state["active_database_id"] is None
    assert len(app.chat_input) == 0
    assert not any("从数据开始" in item.value for item in app.markdown)
    assert any(button.label == "新建数据库" for button in app.sidebar.button)
    assert not any(button.label == "新建数据库" for button in app.main.button)


def test_complex_multi_dimension_answer_uses_dynamic_single_column_layout(rich_csv):
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    project = create_database(app.session_state, "多维分析", rich_csv)
    orders, _ = process_order_csv(rich_csv)
    metrics = calculate_metrics(orders)
    caps = detect_data_capabilities(orders)
    trend = run_analysis(QuestionType.TREND, "最近销售为什么下降？", orders, metrics, caps)
    product = run_analysis(QuestionType.PRODUCT, "从商品角度分析", orders, metrics, caps)
    channel = run_analysis(QuestionType.CHANNEL, "从渠道角度分析", orders, metrics, caps)
    plan = IntentPlan("trend", "revenue", "visualize", True)
    response = AgentResponse(
        answer="最近一周销售变化主要需要从订单量、商品和渠道结构共同判断。",
        steps=(
            AgentStep(1, "analyze_trend", {}, trend.conclusion, True, 1),
            AgentStep(2, "analyze_product", {}, product.conclusion, True, 1),
            AgentStep(3, "analyze_channel", {}, channel.conclusion, True, 1),
            AgentStep(4, "create_visualization", {}, "趋势图", True, 1),
        ),
        visualizations=(VisualizationSpec("trend", "revenue", "line"),),
        fallback_used=False,
        fallback_level="agent",
        stop_reason="final_answer",
        primary_result=trend,
        plan=plan,
        analysis_results=(("trend", trend), ("product", product), ("channel", channel)),
        skills_used=("business-diagnosis",),
    )
    project.chat_messages.extend((
        {"role": "user", "content": "最近销售为什么下降？从商品和渠道两个角度帮我分析，并画一张图。"},
        {
            "role": "assistant", "content": response.answer, "result": trend,
            "plan": plan, "agent_response": response, "user_goal": "最近销售为什么下降？",
        },
    ))
    app.run()
    assert not app.exception
    html = "\n".join(item.value for item in app.markdown)
    assert "assistant-answer" in html
    kpi_markup = next(
        item.value for item in app.markdown
        if '<div class="kpi-summary">' in item.value
    )
    assert kpi_markup.count('class="kpi-summary-item"') == 4
    assert "销售趋势" in html and "商品表现" in html and "渠道表现" in html
    assert "AI 诊断 · 销售趋势" in html
    assert "建议优先关注" not in html
    assert len(app.expander) == 1
    assert any("使用经营诊断 Skill" in item.value for item in app.markdown)
    assert _vega_charts(app)


def test_explicit_chart_request_renders_even_without_agent_visualization_spec(rich_csv):
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    project = create_database(app.session_state, "图表兑现", rich_csv)
    orders, _ = process_order_csv(rich_csv)
    metrics = calculate_metrics(orders)
    caps = detect_data_capabilities(orders)
    product = run_analysis(QuestionType.PRODUCT, "哪个商品销量最高？", orders, metrics, caps)
    response = AgentResponse(
        answer="已完成商品销量分析。",
        steps=(AgentStep(1, "analyze_product", {}, product.conclusion, True, 1),),
        visualizations=(),
        fallback_used=False,
        fallback_level="agent",
        stop_reason="final_answer",
        primary_result=product,
        plan=IntentPlan("product", "quantity", "analyze", False),
        analysis_results=(("product", product),),
    )
    project.chat_messages.extend((
        {"role": "user", "content": "哪个商品销量最高？帮我画出来。"},
        {
            "role": "assistant", "content": response.answer, "result": product,
            "plan": response.plan, "agent_response": response,
            "user_goal": "哪个商品销量最高？帮我画出来。",
        },
    ))

    app.run()

    assert not app.exception
    assert _vega_charts(app)


def test_clarification_message_renders_without_analysis_blocks(rich_csv):
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    project = create_database(app.session_state, "澄清测试", rich_csv)
    project.chat_messages.extend((
        {"role": "user", "content": "帮我比较一下这两个"},
        {
            "role": "assistant",
            "kind": "clarification",
            "content": "你想比较哪两个商品？\n\n1. 欧包和酸奶",
        },
    ))

    app.run()

    assert not app.exception
    html = "\n".join(item.value for item in app.markdown)
    assert "你想比较哪两个商品" in html
    assert "assistant-answer" in html
    assert '<div class="kpi-summary">' not in html
    assert not _vega_charts(app)


def test_streamlit_database_search_and_switch_are_isolated():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    content = (APP_PATH.parent / "data" / "orders.csv").read_bytes()
    campus = create_database(app.session_state, "校园团购", content)
    campus.chat_messages.append({"role": "user", "content": "校园数据库问题"})
    recruitment = create_database(app.session_state, "秋招", content)
    recruitment.chat_messages.append({"role": "user", "content": "秋招数据库问题"})

    app.run()
    assert any("秋招数据库问题" in item.value for item in app.markdown)
    next(button for button in app.button if button.label == "校园团购").click()
    app.run()
    assert app.session_state["active_database_id"] == campus.database_id
    assert any("校园数据库问题" in item.value for item in app.markdown)
    assert not any("秋招数据库问题" in item.value for item in app.markdown)

    next(item for item in app.text_input if item.label == "搜索数据库").input("秋招")
    app.run()
    database_buttons = {button.label for button in app.sidebar.button}
    assert "秋招" in database_buttons
    assert "校园团购" not in database_buttons
