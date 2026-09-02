"""Datalume：AI-first 的多数据库经营分析工作区。"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, MutableMapping

import altair as alt
import pandas as pd
import streamlit as st

from agent_runtime import run_agent
from brand_assets import ensure_logo_asset
from deepseek_client import DeepSeekClient
from data_processor import (
    DataValidationError,
    calculate_metrics,
    detect_data_capabilities,
    process_order_csv,
    summarize_orders,
)
from database_manager import (
    DatabaseProject,
    create_database,
    delete_database,
    get_active_database,
    initialize_database_state,
    rename_database,
    replace_database_data,
    search_databases,
    switch_database,
    toggle_database_pin,
)
from insight import generate_auto_insights
from report_generator import DiagnosticReport, generate_diagnostic_report
from ui_components import (
    comparison_kpis,
    inject_global_styles,
    render_analysis,
    render_brand,
    render_data_start_state,
    render_empty_state,
    render_kpi_cards,
    render_user_message,
    render_workspace_header,
)
from visualization import PALETTE


@st.cache_data(show_spinner=False)
def load_dataset(content: bytes):
    orders, quality = process_order_csv(content)
    return (
        orders,
        quality,
        summarize_orders(orders),
        calculate_metrics(orders),
        detect_data_capabilities(orders),
    )


@st.cache_resource(show_spinner=False)
def load_llm_client() -> DeepSeekClient:
    return DeepSeekClient.from_env()


def sync_chat_history_with_data_source(
    state: MutableMapping[str, Any], source_id: str
) -> bool:
    """兼容旧会话状态：数据源变化时清空聊天与报告。"""

    if state.get("chat_data_source_id") == source_id:
        return False
    state["chat_data_source_id"] = source_id
    state["chat_messages"] = []
    state["diagnostic_report"] = None
    state["diagnostic_report_source_id"] = None
    return True


def _render_database_item(project: DatabaseProject, active_id: str) -> None:
    name_col, pin_col = st.columns([5.5, 1], vertical_alignment="center")
    with name_col:
        if st.button(
            project.name,
            key=f"db_{project.database_id}",
            icon=":material/database:",
            use_container_width=True,
            type="primary" if project.database_id == active_id else "secondary",
        ):
            switch_database(st.session_state, project.database_id)
            st.rerun()
    with pin_col:
        if st.button(
            " ",
            icon=":material/push_pin:" if project.pinned else ":material/keep_off:",
            key=f"pin_{project.database_id}",
            help="取消置顶" if project.pinned else "置顶",
        ):
            toggle_database_pin(st.session_state, project.database_id)
            st.rerun()


@st.dialog("新建数据库", width="small")
def _create_database_dialog() -> None:
    name = st.text_input(
        "数据库名称",
        placeholder="输入数据库名称",
        key="create_database_name",
    )
    uploaded = st.file_uploader(
        "上传订单 CSV",
        type=["csv"],
        key="create_database_csv",
    )
    if st.button("创建数据库", type="primary", use_container_width=True):
        if not name.strip() or uploaded is None:
            st.error("请填写数据库名称并上传 CSV。")
            return
        content = uploaded.getvalue()
        try:
            process_order_csv(content)
        except DataValidationError as error:
            st.error(str(error))
            return
        create_database(st.session_state, name, content)
        st.session_state["show_create_database"] = False
        del st.session_state["create_database_name"]
        del st.session_state["create_database_csv"]
        st.rerun()


def _render_create_database() -> None:
    if st.button("新建数据库", icon=":material/add:", use_container_width=True):
        st.session_state["show_create_group"] = False
        st.session_state["show_create_database"] = True
    if st.session_state.get("show_create_database", False):
        _create_database_dialog()


@st.dialog("重命名数据库", width="small")
def _rename_database_dialog(project: DatabaseProject) -> None:
    input_key = f"rename_database_name_{project.database_id}"
    name = st.text_input(
        "数据库名称",
        value=project.name,
        key=input_key,
    )
    cancel_col, save_col = st.columns(2)
    if cancel_col.button(
        "取消", key=f"cancel_rename_{project.database_id}",
        use_container_width=True,
    ):
        st.session_state["show_rename_database_id"] = None
        st.rerun()
    if save_col.button(
        "保存", key=f"save_rename_{project.database_id}", type="primary",
        use_container_width=True,
    ):
        try:
            rename_database(st.session_state, project.database_id, name)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["show_rename_database_id"] = None
            st.rerun()


@st.dialog("删除数据库", width="small")
def _delete_database_dialog(project: DatabaseProject) -> None:
    st.markdown(f"### 删除“{escape(project.name)}”？")
    st.write(
        "删除后，该数据库在当前会话中的上传数据、对话记录、Agent 上下文、"
        "分析结果和报告都将被移除。"
    )
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button(
        "取消", key=f"cancel_delete_{project.database_id}",
        use_container_width=True,
    ):
        st.session_state["show_delete_database_id"] = None
        st.rerun()
    if delete_col.button(
        "删除", key=f"confirm_delete_{project.database_id}", type="primary",
        use_container_width=True,
    ):
        database_id = project.database_id
        delete_database(st.session_state, database_id)
        st.session_state["show_delete_database_id"] = None
        st.session_state["show_rename_database_id"] = None
        st.session_state.pop(f"rename_database_name_{database_id}", None)
        st.rerun()


@st.dialog("新建分组", width="small")
def _create_group_dialog() -> None:
    name = st.text_input(
        "分组名称", placeholder="输入分组名称", key="create_group_name"
    )
    if st.button("创建分组", type="primary", use_container_width=True):
        cleaned = name.strip()
        if not cleaned:
            st.error("请输入分组名称。")
            return
        groups = st.session_state.setdefault("database_groups", [])
        if cleaned in groups:
            st.info("这个分组已经存在。")
            return
        groups.append(cleaned)
        st.session_state["show_create_group"] = False
        del st.session_state["create_group_name"]
        st.rerun()


def _render_database_groups() -> None:
    for group in st.session_state.get("database_groups", []):
        st.markdown(
            f'<div class="database-group"><span class="group-glyph">⌑</span>'
            f'<span>{escape(group)}</span></div>',
            unsafe_allow_html=True,
        )
    if st.button(
        "新建分组", icon=":material/add:", key="new_database_group",
        use_container_width=True,
    ):
        st.session_state["show_create_database"] = False
        st.session_state["show_create_group"] = True
    if st.session_state.get("show_create_group", False):
        _create_group_dialog()


def render_sidebar(logo_path: Path) -> None:
    with st.sidebar:
        render_brand(logo_path)
        _render_create_database()
        query = st.text_input("搜索数据库", placeholder="搜索数据库", label_visibility="collapsed")
        projects = search_databases(st.session_state, query)
        pinned = [project for project in projects if project.pinned]
        others = [project for project in projects if not project.pinned]
        st.markdown('<div class="sidebar-section">置顶</div>', unsafe_allow_html=True)
        for project in pinned:
            _render_database_item(project, st.session_state["active_database_id"])
        st.markdown('<div class="sidebar-section">全部数据库</div>', unsafe_allow_html=True)
        for project in others:
            _render_database_item(project, st.session_state["active_database_id"])
        _render_database_groups()
        st.markdown(
            '<div class="sidebar-footer"><span class="user-avatar">L</span>'
            '<span>当前用户</span><span class="settings-glyph">⚙</span></div>',
            unsafe_allow_html=True,
        )


def _render_data_management(project: DatabaseProject, overview, quality, capabilities) -> None:
    st.caption(
        f"{overview.row_count} 行 · {overview.user_count} 位用户 · "
        f"{overview.product_count} 个商品 · {overview.start_date:%m/%d}–{overview.end_date:%m/%d}"
    )
    supported = "、".join(label for label, status, _ in capabilities.status_items() if status == "✓")
    limited = "、".join(label for label, status, _ in capabilities.status_items() if status != "✓")
    st.caption(f"支持：{supported}" + (f"　受限：{limited}" if limited else ""))
    for warning in quality.warnings:
        st.caption(warning)
    replacement = st.file_uploader(
        "替换当前数据库 CSV",
        type=["csv"],
        key=f"replace_{project.database_id}",
    )
    if replacement is not None:
        try:
            process_order_csv(replacement.getvalue())
        except DataValidationError as error:
            st.error(str(error))
        else:
            if st.button("确认切换数据源", key=f"confirm_replace_{project.database_id}"):
                if replace_database_data(project, replacement.getvalue()):
                    st.rerun()
                else:
                    st.info("上传文件与当前数据源相同。")


def _render_dashboard(orders, metrics, capabilities) -> None:
    """保留原 Dashboard 能力，在按需弹层中紧凑展示。"""

    render_kpi_cards(comparison_kpis(orders, capabilities))
    product_tab, trend_tab, profile_tab, value_tab, insight_tab = st.tabs(
        ["商品销售", "销售趋势", "用户画像", "用户价值", "自动洞察"]
    )
    with product_tab:
        frame = metrics.product_sales.rename("销售额").reset_index()
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[4]).encode(
            x=alt.X("销售额:Q", title=None),
            y=alt.Y("product:N", sort="-x", title=None),
            tooltip=["product", "销售额"],
        ).properties(height=245)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            frame.rename(columns={"product": "商品"}),
            use_container_width=True,
            hide_index=True,
        )
    with trend_tab:
        frame = metrics.daily_sales.rename("销售额").reset_index()
        chart = alt.Chart(frame).mark_line(point=True, color=PALETTE[3]).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("销售额:Q", title=None),
            tooltip=["date:T", "销售额:Q"],
        ).properties(height=245)
        st.altair_chart(chart, use_container_width=True)
    with profile_tab:
        left, right = st.columns(2)
        left.markdown("**性别用户数**")
        left.dataframe(metrics.gender_count.rename("用户数"), use_container_width=True)
        right.markdown("**年级用户数**")
        right.dataframe(metrics.grade_count.rename("用户数"), use_container_width=True)
    with value_tab:
        user_value = orders.groupby("user_id").agg(消费金额=("sales", "sum"), 购买记录数=("sales", "size"), 最近购买=("date", "max")).sort_values("消费金额", ascending=False)
        st.dataframe(user_value, use_container_width=True)
    with insight_tab:
        for item in generate_auto_insights(metrics):
            st.write(f"- {item}")


def _render_report(project: DatabaseProject, report: DiagnosticReport | None) -> None:
    if report is None:
        return
    st.markdown(report.to_markdown())
    st.download_button(
        "下载经营诊断报告",
        report.to_markdown(),
        file_name="datalume_ai_report.md",
        mime="text/markdown",
        key=f"download_report_{project.database_id}",
    )


@st.dialog("经营分析", width="large")
def _dashboard_dialog(orders, metrics, capabilities) -> None:
    _render_dashboard(orders, metrics, capabilities)


@st.dialog("AI 经营诊断报告", width="large")
def _report_dialog(project: DatabaseProject) -> None:
    _render_report(project, project.diagnostic_report)


def _render_workspace_toolbar(project, orders, quality, overview, metrics, capabilities) -> None:
    title_col, menu_col = st.columns([12, 2], vertical_alignment="center")
    with title_col:
        render_workspace_header(project.name)
    with menu_col:
        with st.popover(
            "更多", icon=":material/more_horiz:", help="数据库操作",
            use_container_width=True,
        ):
            st.markdown("**数据库**")
            if st.button(
                "重命名", icon=":material/edit:", key=f"rename_{project.database_id}",
                use_container_width=True,
            ):
                st.session_state["show_delete_database_id"] = None
                st.session_state["show_rename_database_id"] = project.database_id
            if st.button(
                "删除数据库", icon=":material/delete:", key=f"delete_{project.database_id}",
                use_container_width=True,
            ):
                st.session_state["show_rename_database_id"] = None
                st.session_state["show_delete_database_id"] = project.database_id
            st.divider()
            _render_data_management(project, overview, quality, capabilities)
            st.divider()
            if st.button(
                "查看经营分析",
                icon=":material/monitoring:",
                use_container_width=True,
            ):
                _dashboard_dialog(orders, metrics, capabilities)
            if st.button(
                "生成 AI 经营诊断报告",
                icon=":material/description:",
                use_container_width=True,
            ):
                project.diagnostic_report = generate_diagnostic_report(
                    orders, metrics, overview, capabilities
                )
                project.diagnostic_report_source_id = project.data_source_id
                _report_dialog(project)
            elif project.diagnostic_report is not None:
                if st.button(
                    "查看已有报告",
                    icon=":material/article:",
                    use_container_width=True,
                ):
                    _report_dialog(project)
    if st.session_state.get("show_rename_database_id") == project.database_id:
        _rename_database_dialog(project)
    if st.session_state.get("show_delete_database_id") == project.database_id:
        _delete_database_dialog(project)
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


def _answer_prompt(
    prompt: str, project: DatabaseProject, orders, overview, metrics, capabilities
) -> None:
    project.chat_messages.append({"role": "user", "content": prompt})
    with st.spinner("正在分析数据…"):
        response = run_agent(
            prompt,
            database_id=project.database_id,
            orders=orders,
            overview=overview,
            metrics=metrics,
            capabilities=capabilities,
            llm_client=load_llm_client(),
            conversation_context=project.conversation_context,
        )
    project.agent_history.append({"goal": prompt, **response.public_trace()})
    del project.agent_history[:-20]
    if response.report is not None:
        project.diagnostic_report = response.report
        project.diagnostic_report_source_id = project.data_source_id
    project.chat_messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "result": response.primary_result,
            "plan": response.plan,
            "fallback_used": response.fallback_level == "deterministic",
            "agent_fallback": response.fallback_used and response.fallback_level != "deterministic",
            "agent_response": response,
            "user_goal": prompt,
        }
    )


def _quick_questions(orders, capabilities) -> tuple[str, ...]:
    questions = ["哪个商品销量最高？", "哪些用户值得重点维护？"]
    date_span = int(
        (pd.Timestamp(orders["date"].max()).normalize() - pd.Timestamp(orders["date"].min()).normalize()).days
    ) + 1
    if date_span >= 14:
        questions.append("最近一周销售额为什么变化？")
    elif capabilities.profit_supported:
        questions.append("哪个商品利润最高？")
    elif capabilities.channel_supported:
        questions.append("哪个获客渠道效果最好？")
    elif capabilities.repurchase_reliable:
        questions.append("哪些用户的复购率最高？")
    else:
        questions.append("现在应该怎么提升销售额？")
    return tuple(questions)


def main() -> None:
    st.set_page_config(page_title="Datalume", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_global_styles()
    logo_path = ensure_logo_asset()
    initialize_database_state(st.session_state)
    st.session_state.setdefault("database_groups", [])
    st.session_state.setdefault("show_create_group", False)
    st.session_state.setdefault("show_rename_database_id", None)
    st.session_state.setdefault("show_delete_database_id", None)
    render_sidebar(logo_path)
    project = get_active_database(st.session_state)
    if project is None:
        render_data_start_state()
        action_cols = st.columns([5, 2, 5])
        with action_cols[1]:
            if st.button(
                "新建数据库",
                icon=":material/add:",
                type="primary",
                use_container_width=True,
                key="empty_create_database",
            ):
                st.session_state["show_create_database"] = True
                st.rerun()
        return

    try:
        orders, quality, overview, metrics, capabilities = load_dataset(project.csv_content)
    except DataValidationError as error:
        st.error(str(error))
        st.stop()

    _render_workspace_toolbar(project, orders, quality, overview, metrics, capabilities)

    if not project.chat_messages:
        render_empty_state()
        st.markdown("**你可以这样问**")
        quick_cols = st.columns(3)
        quick_questions = _quick_questions(orders, capabilities)
        for index, question in enumerate(quick_questions):
            if quick_cols[index].button(
                question,
                key=f"quick_{project.database_id}_{index}",
                use_container_width=True,
            ):
                _answer_prompt(question, project, orders, overview, metrics, capabilities)
                st.rerun()
    else:
        for message_index, message in enumerate(project.chat_messages):
            if message["role"] == "user":
                render_user_message(message["content"])
            else:
                render_analysis(
                    message["result"],
                    orders,
                    metrics,
                    capabilities,
                    answer=message.get("content"),
                    plan=message.get("plan"),
                    fallback_used=bool(message.get("fallback_used", False)),
                    agent_fallback=bool(message.get("agent_fallback", False)),
                    agent_response=message.get("agent_response"),
                    user_goal=message.get("user_goal"),
                    message_key=f"{project.database_id}_{message_index}",
                )

    prompt = st.chat_input("继续提问你的经营问题...")
    if prompt:
        _answer_prompt(prompt, project, orders, overview, metrics, capabilities)
        st.rerun()


if __name__ == "__main__":
    main()
