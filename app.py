"""Datalume：AI-first 的多数据库经营分析工作区。"""

from __future__ import annotations

from html import escape
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any, MutableMapping

import altair as alt
import pandas as pd
import streamlit as st

from datalume.agent_runtime import run_agent
from datalume.brand_assets import ensure_logo_asset
from datalume.deepseek_client import DeepSeekClient
from datalume.data_processor import (
    DataValidationError,
    calculate_metrics,
    detect_data_capabilities,
    excel_sheet_names,
    fingerprint_csv,
    prepare_order_upload,
    process_order_csv,
    summarize_orders,
)
from datalume.data_preparation import FIELD_ALIASES, inspect_quality, prepare_table, read_table, suggest_mappings
from datalume.partial_analysis import render_partial_analysis
from datalume.import_recommendation import ImportRecommendationError, make_preview_payload, recommend_import
from datalume.database_manager import (
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
from datalume.insight import generate_auto_insights
from datalume.report_generator import DiagnosticReport, generate_diagnostic_report
from datalume.ui_components import (
    comparison_kpis,
    inject_global_styles,
    render_analysis,
    render_assistant_message,
    render_brand,
    render_empty_state,
    render_kpi_cards,
    render_user_message,
    render_workspace_header,
)
from datalume.visualization import PALETTE, style_chart


@st.cache_data(show_spinner=False)
def load_dataset(content: bytes, remove_duplicates: bool = True):
    orders, quality = process_order_csv(content, remove_duplicates=remove_duplicates)
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


def _set_flash(message: str) -> None:
    """在下一次 rerun 后显示一次轻量操作反馈。"""

    st.session_state["ui_flash_message"] = message


def _render_flash() -> None:
    message = st.session_state.pop("ui_flash_message", None)
    if message:
        st.toast(message, icon=":material/check_circle:")


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
            "取消置顶" if project.pinned else "置顶",
            icon=":material/push_pin:" if project.pinned else ":material/keep_off:",
            key=f"pin_{project.database_id}",
            help="取消置顶" if project.pinned else "置顶",
        ):
            toggle_database_pin(st.session_state, project.database_id)
            st.rerun()


def _prepare_uploaded_orders(uploaded, *, key_prefix: str):
    """Inspect locally, optionally request an AI suggestion, and confirm import."""
    if uploaded is None:
        st.session_state.pop(f"{key_prefix}_recommendation", None)
        st.session_state.pop(f"{key_prefix}_recommendation_scope", None)
        st.session_state.pop(f"{key_prefix}_pending_recommendation", None)
        return None
    content, filename = uploaded.getvalue(), uploaded.name
    file_key = fingerprint_csv(content)
    try:
        sheet_name = None
        if Path(filename).suffix.lower() in {".xlsx", ".xls"}:
            sheets = excel_sheet_names(content, filename=filename)
            sheet_name = (st.selectbox("选择工作表", sheets, key=f"{key_prefix}_sheet_{file_key}")
                          if len(sheets) > 1 else sheets[0])
        selection_key = sha256(f"{file_key}:{sheet_name}".encode()).hexdigest()
        header_key = f"{key_prefix}_header_{selection_key}"
        pending = st.session_state.pop(f"{key_prefix}_pending_recommendation", None)
        if pending is not None and pending["selection_key"] == selection_key:
            # Apply before the header and mapping widgets are instantiated.
            st.session_state[header_key] = pending["header_row"] + 1
            target_key = sha256(f"{selection_key}:{pending['header_row']}".encode()).hexdigest()
            st.session_state[f"{key_prefix}_mapping_seed_{target_key}"] = pending["mapping"]
            version_key = f"{key_prefix}_mapping_version_{target_key}"
            st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
        with st.expander("预览表格与调整表头", expanded=False):
            header_row = int(st.number_input(
                "表头位于第几行", min_value=1, value=1, step=1, key=header_key,
                help="如果第一行是标题，请选择真正包含字段名称的行。",
            )) - 1
            table = read_table(content, filename=filename, sheet_name=sheet_name, header_row=header_row)
            if table.frame.empty:
                st.error("表头下面没有数据。")
                return None
            st.caption(f"识别到 {len(table.columns)} 列、{len(table.frame)} 行；请核对原始内容。")
            preview = table.frame.head(8).copy()
            preview.columns = [f"{name}（第{i + 1}列）" for i, name in enumerate(table.columns)]
            st.dataframe(preview, hide_index=True, use_container_width=True)
        mapping_key = sha256(f"{selection_key}:{header_row}".encode()).hexdigest()
        scope_key = f"{key_prefix}_recommendation_scope"
        previous_scope = st.session_state.get(scope_key)
        if previous_scope != mapping_key:
            st.session_state.pop(f"{key_prefix}_recommendation", None)
            if previous_scope:
                st.session_state.pop(f"{key_prefix}_mapping_seed_{previous_scope}", None)
                st.session_state.pop(f"{key_prefix}_mapping_version_{previous_scope}", None)
                st.session_state.pop(f"{key_prefix}_ai_consent_{previous_scope}", None)
                for state_key in list(st.session_state):
                    if state_key.startswith(f"{key_prefix}_confirm_{previous_scope}_"):
                        del st.session_state[state_key]
            st.session_state[scope_key] = mapping_key
        labels = {
            "date": "日期", "revenue": "销售额", "channel": "渠道", "user_id": "用户编号",
            "product": "商品", "price": "单价", "quantity": "数量", "gender": "性别",
            "grade": "年级", "order_id": "订单编号", "unit_cost": "单位成本",
            "total_cost": "总成本", "profit": "利润", "acquisition_cost": "获客成本",
        }
        st.markdown("**推荐导入方案**")
        st.caption("已根据表头生成本地建议。可以修改字段；金额单位和业务口径仍需你核对。")
        with st.expander("让 AI 帮我理解表格", expanded=False):
            st.caption(
                "经你同意并点击后，会向已配置的模型服务（默认为 DeepSeek）发送所选工作表名称、"
                "前 15 行及前 12 列的单元格内容（每格最多 80 字），以及表格行列数。"
                "样本可能包含个人信息、商业或薪酬数据，请先确认适合发送。"
            )
            raw_table = read_table(content, filename=filename, sheet_name=sheet_name, header_row=0)
            if st.checkbox("查看将发送的表格样本", key=f"{key_prefix}_show_sample_{mapping_key}"):
                payload = make_preview_payload(raw_table)
                st.caption(f"工作表：{payload['sheet_name'] or 'CSV'}；展示行数 {payload['sample_row_count']}，列数 {payload['sample_column_count']}。")
                st.dataframe(pd.DataFrame([item["cells"] for item in payload["sample_rows"]]),
                             hide_index=True, use_container_width=True)
            consent = st.checkbox("同意发送表格样本给 AI", key=f"{key_prefix}_ai_consent_{mapping_key}")
            if st.button("生成 AI 推荐方案", disabled=not consent, key=f"{key_prefix}_request_ai_{mapping_key}"):
                if consent:
                    try:
                        with st.spinner("AI 正在理解表格结构…"):
                            result = recommend_import(raw_table, load_llm_client())
                        st.session_state[f"{key_prefix}_recommendation"] = result
                    except (ImportRecommendationError, DataValidationError):
                        st.session_state.pop(f"{key_prefix}_recommendation", None)
                        st.warning("AI 推荐暂不可用，可继续使用本地建议和手动修改。")
            recommendation = st.session_state.get(f"{key_prefix}_recommendation")
            if recommendation is not None:
                st.caption("AI 建议仅供核对，不会自动改动数据或字段。")
                if recommendation.structure:
                    st.text(f"表格类型：{recommendation.structure}")
                if recommendation.description:
                    st.text(recommendation.description)
                if recommendation.header_row is not None:
                    st.caption(f"建议表头：第 {recommendation.header_row + 1} 行。")
                    proposed_table = read_table(content, filename=filename, sheet_name=sheet_name,
                                                header_row=recommendation.header_row)
                    if recommendation.mapping:
                        st.text("建议字段：" + "；".join(
                            f"{labels[field]} ← {proposed_table.columns[index]}（第 {index + 1} 列）"
                            for field, index in recommendation.mapping.items()
                        ))
                    if st.button("应用 AI 推荐", key=f"{key_prefix}_apply_ai_{mapping_key}"):
                        st.session_state[f"{key_prefix}_pending_recommendation"] = {
                            "selection_key": selection_key,
                            "header_row": recommendation.header_row,
                            "mapping": dict(recommendation.mapping),
                        }
                        st.rerun()
                for concern in recommendation.concerns:
                    st.text("待核对：" + concern)
        summary = st.container()
        suggestions = st.session_state.get(f"{key_prefix}_mapping_seed_{mapping_key}", suggest_mappings(table))
        version = st.session_state.get(f"{key_prefix}_mapping_version_{mapping_key}", 0)
        mapping = {}
        options = [-1, *range(len(table.columns))]
        with st.expander("修改方案", expanded=False):
            st.caption("可手动调整字段对应关系；不确定的字段保持未选择。")
            for field in ("date", "revenue", "channel", "user_id", "product", "price", "quantity",
                          "gender", "grade", "order_id", "unit_cost", "total_cost", "profit", "acquisition_cost"):
                suggested = suggestions.get(field, -1)
                if suggested not in options:
                    suggested = -1
                chosen = st.selectbox(
                    labels[field], options, index=options.index(suggested),
                    format_func=lambda i: "未选择" if i == -1 else f"{table.columns[i]}（第{i + 1}列）",
                    key=f"{key_prefix}_map_{mapping_key}_{version}_{field}",
                )
                if chosen != -1:
                    mapping[field] = chosen
            remove_duplicates = st.checkbox(
                "删除完全重复的行", value=False, key=f"{key_prefix}_dedupe_{mapping_key}",
                help="重复行可能是真实交易；默认保留。",
            )
        with summary:
            st.caption(f"表头：第 {header_row + 1} 行；检测到 {len(table.columns)} 列、{len(table.frame)} 行。")
            if mapping:
                st.text("字段建议：" + "；".join(
                    f"{labels[field]} ← {table.columns[index]}（第 {index + 1} 列）"
                    for field, index in mapping.items()
                ))
            else:
                st.info("未能确定匹配字段。可以展开“修改方案”选择，或让 AI 给出建议。")
            try:
                quality = inspect_quality(table, mapping)
            except DataValidationError as error:
                st.error(str(error))
                return None
            st.caption(f"完全空白行：{len(quality.empty_rows)}；完全重复行：{len(quality.duplicate_rows)}。"
                       + ("将删除重复行。" if remove_duplicates else "重复行默认保留。"))
            if quality.duplicate_rows:
                st.caption("疑似重复行：" + "、".join(map(str, quality.duplicate_rows[:20])))
            for field, rows in quality.invalid_numeric.items():
                if rows:
                    st.warning(f"{labels[field]}有 {len(rows)} 行为空或不是可识别的数字（如第 {rows[0]} 行）；请检查原表或映射。")
            if quality.invalid_dates or quality.ambiguous_dates:
                st.warning(f"日期有 {len(quality.invalid_dates)} 行无效、{len(quality.ambiguous_dates)} 行格式有歧义，请核对。")
            try:
                prepared = prepare_table(table, mapping, remove_duplicates=remove_duplicates)
                if prepared.profile == "orders":
                    process_order_csv(prepared.csv_content, remove_duplicates=False)
                    st.caption("当前方案可使用完整订单分析。")
                else:
                    st.caption("当前方案可使用基础销售额趋势" + ("和渠道汇总。" if "channel" in mapping else "。"))
                    st.caption("请核对金额单位、退款和统计范围；无法确认的口径请先返回原表核实。")
            except DataValidationError as error:
                st.info("当前方案暂不能导入：" + str(error))
                st.caption("当前仅支持完整订单数据，或含日期和销售额的数据。其他结构仍需整理或扩展分析能力。")
                return None
        signature = sha256(json.dumps([mapping, remove_duplicates, header_row, version], sort_keys=True).encode()).hexdigest()
        if not st.checkbox(
            "我已核对字段、清洗方式和金额口径", key=f"{key_prefix}_confirm_{mapping_key}_{signature}",
        ):
            st.info("请核对方案并勾选确认后继续。")
            return None
        return prepared
    except DataValidationError as error:
        st.error(str(error))
        return None


@st.dialog("新建数据库", width="small")
def _create_database_dialog() -> None:
    name = st.text_input(
        "数据库名称",
        placeholder="输入数据库名称",
        key="create_database_name",
    )
    uploaded = st.file_uploader(
        "上传 CSV 或 Excel",
        type=["csv", "xlsx", "xls"],
        key="create_database_csv",
    )
    content = _prepare_uploaded_orders(uploaded, key_prefix="create_order")
    if st.button("创建数据库", type="primary", use_container_width=True):
        cleaned_name = name.strip()
        if not cleaned_name or uploaded is None:
            st.error("请填写数据库名称并上传 CSV 或 Excel。")
            return
        if any(
            project.name.casefold() == cleaned_name.casefold()
            for project in st.session_state.get("databases", {}).values()
        ):
            st.error("数据库名称已存在，请换一个名称。")
            return
        if content is None:
            return
        create_database(st.session_state, cleaned_name, content.csv_content,
                        profile="orders" if content.profile == "orders" else "partial",
                        remove_duplicates=False)
        _set_flash(f"已创建数据库“{cleaned_name}”。")
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
        st.session_state.pop(input_key, None)
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
            _set_flash(f"数据库已重命名为“{project.name}”。")
            st.session_state["show_rename_database_id"] = None
            st.session_state.pop(input_key, None)
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
        project_name = project.name
        delete_database(st.session_state, database_id)
        _set_flash(f"已删除数据库“{project_name}”。")
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
        with st.container(key="sidebar_navigation"):
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
            if query.strip() and not projects:
                st.caption("没有匹配的数据库")
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
        "替换当前数据库 CSV 或 Excel",
        type=["csv", "xlsx", "xls"],
        key=f"replace_{project.database_id}",
    )
    if replacement is not None:
        st.warning("替换数据后，当前数据库的聊天记录、Agent 上下文和报告将被清空。")
        content = _prepare_uploaded_orders(
            replacement, key_prefix=f"replace_order_{project.database_id}"
        )
        if content is not None:
            if st.button(
                "确认替换数据",
                key=f"confirm_replace_{project.database_id}",
                use_container_width=True,
            ):
                if replace_database_data(project, content.csv_content,
                                         profile="orders" if content.profile == "orders" else "partial",
                                         remove_duplicates=False):
                    _set_flash("数据源已更新，旧对话与报告已清空。")
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
        st.altair_chart(style_chart(chart, height=245), use_container_width=True, theme=None)
        st.dataframe(
            frame.rename(columns={"product": "商品"}),
            use_container_width=True,
            hide_index=True,
        )
    with trend_tab:
        frame = metrics.daily_sales.rename("销售额").reset_index()
        chart = alt.Chart(frame).mark_line(point=True, color=PALETTE[4]).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("销售额:Q", title=None),
            tooltip=["date:T", "销售额:Q"],
        ).properties(height=245)
        st.altair_chart(style_chart(chart, height=245), use_container_width=True, theme=None)
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
    with st.container(key="workspace_toolbar"):
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
                    with st.spinner("正在生成经营诊断报告…"):
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
    render_user_message(prompt)
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
            "kind": "clarification" if response.needs_clarification else "analysis",
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


def _render_partial_workspace(project: DatabaseProject) -> None:
    """Limited aggregates for tables that cannot safely use the order agent."""
    title_col, menu_col = st.columns([12, 2], vertical_alignment="center")
    with title_col:
        render_workspace_header(project.name)
    with menu_col:
        with st.popover("更多", icon=":material/more_horiz:", use_container_width=True):
            if st.button("重命名", key=f"rename_{project.database_id}"):
                st.session_state["show_rename_database_id"] = project.database_id
            if st.button("删除数据库", key=f"delete_{project.database_id}"):
                st.session_state["show_delete_database_id"] = project.database_id
            replacement = st.file_uploader("替换当前数据库 CSV 或 Excel", type=["csv", "xlsx", "xls"], key=f"replace_{project.database_id}")
            if replacement is not None:
                st.warning("替换数据后，当前数据库的聊天记录和报告将被清空。")
                prepared = _prepare_uploaded_orders(replacement, key_prefix=f"replace_order_{project.database_id}")
                if prepared is not None and st.button("确认替换数据", key=f"confirm_replace_{project.database_id}"):
                    if replace_database_data(project, prepared.csv_content,
                                             profile="orders" if prepared.profile == "orders" else "partial",
                                             remove_duplicates=False):
                        _set_flash("数据源已更新。")
                        st.rerun()
                    else:
                        st.info("上传数据与当前数据源相同。")
    if st.session_state.get("show_rename_database_id") == project.database_id:
        _rename_database_dialog(project)
    if st.session_state.get("show_delete_database_id") == project.database_id:
        _delete_database_dialog(project)
    frame = pd.read_csv(BytesIO(project.csv_content), dtype="string", keep_default_na=False)
    render_partial_analysis(frame)


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
    _render_flash()
    project = get_active_database(st.session_state)
    if project is None:
        return

    if getattr(project, "profile", "orders") == "partial":
        _render_partial_workspace(project)
        return

    try:
        orders, quality, overview, metrics, capabilities = load_dataset(
            project.csv_content, getattr(project, "remove_duplicates", True)
        )
    except DataValidationError as error:
        st.error(str(error))
        st.stop()

    _render_workspace_toolbar(project, orders, quality, overview, metrics, capabilities)

    if not project.chat_messages:
        render_empty_state()
        st.markdown("**你可以这样问**")
        with st.container(key="quick_questions"):
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
            elif message.get("kind") == "clarification":
                render_assistant_message(message["content"])
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
