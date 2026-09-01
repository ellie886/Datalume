"""Datalume 的 Streamlit 视觉组件与结构化分析呈现。"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import streamlit as st

from agent_models import AgentResponse
from analysis_engine import AnalysisResult, order_count, percent_change
from data_processor import BusinessMetrics, DataCapabilities
from deepseek_client import IntentPlan
from visualization import build_visualization


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
:root { --ink:#171717; --muted:#737373; --line:#e8e8e5; --soft:#f7f7f5; }
html, body, [class*="css"] { color:var(--ink); }
.stApp { background:#fff; }
[data-testid="stSidebar"] { background:#f7f7f5; border-right:1px solid #e5e5e2; min-width:274px; max-width:274px; }
[data-testid="stSidebar"] > div:first-child { padding:.9rem .8rem .75rem; }
[data-testid="stSidebar"] [data-testid="stImage"] img { border-radius:10px; }
.block-container { max-width:1240px; padding:.45rem 2rem 5.75rem; }
h1,h2,h3,p { letter-spacing:-.012em; }
h1 { font-size:1.35rem !important; font-weight:650 !important; margin:.15rem 0 .9rem !important; }
h2 { font-size:1.05rem !important; font-weight:650 !important; }
h3 { font-size:.98rem !important; font-weight:650 !important; }
[data-testid="stHeader"] { background:transparent; height:0; }
[data-testid="stToolbar"] { visibility:hidden; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stElementToolbar"], .stElementToolbar { display:none !important; }
.brand-word { font-size:1.48rem; font-weight:720; letter-spacing:-.04em; line-height:42px; }
.sidebar-section { margin:1rem .2rem .25rem; font-size:.75rem; font-weight:650; color:#666; }
.db-caption { color:#999; font-size:.72rem; margin:-.35rem 0 .55rem; }
.database-group { display:flex; align-items:center; gap:.55rem; color:#555; font-size:.82rem; padding:.5rem .55rem; }
.group-glyph { color:#888; font-size:.9rem; }
.sidebar-footer { position:fixed; bottom:.8rem; left:.8rem; width:247px; display:flex; align-items:center; gap:.65rem; padding:.65rem .45rem; border-top:1px solid #e3e3e0; background:#f7f7f5; font-size:.82rem; z-index:10; }
.user-avatar { width:27px; height:27px; display:grid; place-items:center; border-radius:50%; background:#e5e5e2; color:#555; }
.settings-glyph { margin-left:auto; color:#555; font-size:1rem; }
.workspace-head { background:rgba(255,255,255,.97); padding:.8rem .1rem .7rem; }
.workspace-title { font-size:1.03rem; font-weight:680; display:flex; gap:.6rem; align-items:center; white-space:nowrap; }
.database-glyph { width:22px; height:22px; border:1.4px solid #222; border-radius:50%/25%; display:inline-flex; align-items:center; justify-content:center; font-size:.7rem; }
.header-rule { border-bottom:1px solid var(--line); margin:0 0 .75rem; }
.empty-state { padding:3.4rem 1rem 1.8rem; text-align:center; }
.empty-orb { display:inline-grid; place-items:center; width:38px; height:38px; border-radius:50%; color:white; background:#111; font-size:1rem; margin-bottom:.8rem; }
.empty-state h2 { font-size:1.25rem !important; margin-bottom:.35rem; }
.empty-state p { color:var(--muted); margin:0; }
.data-start-state { min-height:48vh; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; text-align:center; }
.data-start-state .empty-orb { width:42px; height:42px; }
.data-start-state h1 { font-size:1.35rem !important; margin:.1rem 0 .4rem !important; }
.data-start-state p { color:var(--muted); font-size:.9rem; margin:0; }
.user-bubble { margin:.2rem 0 .7rem auto; padding:.62rem .9rem; background:#f4f4f3; border-radius:11px; width:fit-content; max-width:72%; font-size:.9rem; }
.assistant-response { display:grid; grid-template-columns:29px minmax(0,1fr); align-items:start; column-gap:.65rem; width:100%; margin:.05rem 0 .45rem; }
.assistant-answer { min-width:0; width:100%; color:var(--ink); font-size:.94rem; line-height:1.68; text-align:left; white-space:normal; overflow-wrap:break-word; word-break:normal; }
.assistant-answer p { margin:0 0 .45rem; }
.assistant-answer p:last-child { margin-bottom:0; }
.ai-label { display:grid; place-items:center; width:29px; height:29px; background:#111; color:#fff; border-radius:50%; margin-top:.05rem; font-size:.8rem; }
.kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:.5rem 0 .65rem; }
.kpi-card { border:1px solid var(--line); border-radius:10px; padding:.72rem .9rem; background:#fff; min-height:90px; box-shadow:0 1px 1px rgba(0,0,0,.015); }
.kpi-label { font-size:.76rem; color:#666; margin-bottom:.35rem; }
.kpi-value { font-size:1.43rem; line-height:1.15; font-weight:650; letter-spacing:-.04em; }
.kpi-delta { font-size:.73rem; color:#888; margin-top:.4rem; }
.delta-up { color:#c55555; } .delta-down { color:#38875b; } .delta-flat { color:#777; }
.kpi-summary { display:flex; flex-wrap:wrap; gap:.45rem 1.2rem; margin:.35rem 0 .65rem 2.4rem; padding:.55rem .7rem; border:1px solid var(--line); border-radius:9px; background:#fff; }
.kpi-summary-item { display:flex; align-items:baseline; gap:.35rem; min-width:0; }
.kpi-summary-value { font-size:1.02rem; font-weight:680; letter-spacing:-.025em; }
.kpi-summary-label { color:#666; font-size:.76rem; }
.kpi-summary-delta { font-size:.72rem; white-space:nowrap; }
.limit-box { border-top:1px solid var(--line); color:#777; padding:.45rem .15rem 0; margin:.5rem 0 0 2.4rem; max-width:calc(100% - 2.4rem); font-size:.75rem; line-height:1.45; }
.fallback-note { color:#888; font-size:.73rem; margin:-.25rem 0 .4rem 2.45rem; }
[data-testid="stChatInput"] { border:1px solid #dededb; border-radius:13px; background:#fff; box-shadow:0 5px 18px rgba(0,0,0,.045); }
[data-testid="stChatInput"]:focus-within { border-color:#bdbdb9; box-shadow:0 5px 18px rgba(0,0,0,.055); }
[data-testid="stChatInput"] textarea { min-height:48px !important; padding-left:.9rem !important; }
[data-testid="stChatInputSubmitButton"] { border-radius:50% !important; background:#111 !important; color:#fff !important; margin-right:.35rem !important; }
button[kind="secondary"] { border-color:#e4e4e1 !important; background:#fff !important; border-radius:8px !important; }
button[kind="secondary"]:hover { border-color:#cacac6 !important; color:#111 !important; }
button[kind="primary"] { background:#111 !important; color:#fff !important; border-color:#111 !important; box-shadow:none !important; }
[data-testid="stSidebar"] button { min-height:2.15rem; font-size:.84rem; }
[data-testid="stSidebar"] button[kind="primary"] { background:#ececea !important; color:#171717 !important; border-color:#ececea !important; box-shadow:none !important; }
[data-testid="stSidebar"] [data-testid="stTextInput"] input { background:#fff; border-color:#e4e4e1; min-height:2.25rem; }
[data-testid="stPopover"] button { border:0 !important; background:transparent !important; box-shadow:none !important; }
[data-testid="stFileUploaderDropzone"] { background:#fafaf9; border-color:#e3e3e0; padding:.65rem; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span { font-size:.78rem; }
[data-testid="stMetric"] { border:1px solid var(--line); border-radius:10px; padding:.8rem; }
[data-testid="stExpander"] { border-color:var(--line); border-radius:10px; }
[class*="st-key-viz_"] { margin-left:2.4rem; max-width:calc(100% - 2.4rem); border-color:var(--line) !important; border-radius:10px !important; padding:.5rem .7rem .1rem !important; }
[class*="st-key-viz_"] [data-testid="stMarkdownContainer"] p { font-size:.84rem; margin:0; }
[class*="st-key-viz_"] [data-testid="stVegaLiteChart"] { margin-top:-.15rem; }
.analysis-section { margin:.45rem 0 .55rem 2.4rem; max-width:calc(100% - 2.4rem); }
.analysis-section-title { color:#333; font-size:.82rem; font-weight:660; margin:0 0 .25rem; }
.analysis-line { display:grid; grid-template-columns:10px minmax(0,1fr); column-gap:.4rem; color:#4f4f4f; font-size:.81rem; line-height:1.55; margin:.18rem 0; }
.analysis-dot { color:#aaa; }
.suggestions { margin:.5rem 0 .55rem 2.4rem; padding-top:.4rem; border-top:1px solid var(--line); }
.suggestion-row { display:grid; grid-template-columns:22px minmax(0,1fr); gap:.5rem; align-items:start; margin:.32rem 0; color:#3f3f3f; font-size:.82rem; line-height:1.5; }
.suggestion-index { width:22px; height:22px; display:grid; place-items:center; border-radius:50%; background:#f1f1ef; color:#555; font-size:.7rem; }
.trace-list { color:#666; font-size:.78rem; line-height:1.65; }
.chart-title { color:#333; font-size:.82rem; font-weight:660; margin:0 0 .1rem; }
[data-testid="stExpander"] { margin-left:2.4rem; max-width:calc(100% - 2.4rem); background:#fff; }
[data-testid="stExpander"] summary { font-size:.76rem; color:#777; }
[data-testid="stVerticalBlock"] { gap:.55rem; }
[data-testid="stDialog"] [data-testid="stVerticalBlock"] { gap:.7rem; }
@media(max-width:900px){ .kpi-grid{grid-template-columns:repeat(2,1fr)} .block-container{padding-left:1rem;padding-right:1rem} .kpi-summary,.analysis-section,.suggestions,.limit-box,[data-testid="stExpander"],[class*="st-key-viz_"]{margin-left:0;max-width:100%} }
</style>
""",
        unsafe_allow_html=True,
    )


def render_brand(logo_path: Path) -> None:
    logo, word = st.columns([1, 4.4], vertical_alignment="center")
    with logo:
        st.image(str(logo_path), width=44)
    with word:
        st.markdown('<div class="brand-word">Datalume</div>', unsafe_allow_html=True)


def render_workspace_header(name: str) -> None:
    st.markdown(
        f'<div class="workspace-head"><div class="workspace-title">'
        f'<span class="database-glyph">≡</span>{escape(name)}</div></div>',
        unsafe_allow_html=True,
    )


def _delta_html(value: float | None) -> str:
    if value is None:
        return '<span class="delta-flat">— 数据周期不足</span>'
    if value > 0:
        return f'<span class="delta-up">↑ {abs(value):.1%}</span>'
    if value < 0:
        return f'<span class="delta-down">↓ {abs(value):.1%}</span>'
    return '<span class="delta-flat">— 0.0%</span>'


def comparison_kpis(
    orders: pd.DataFrame, caps: DataCapabilities
) -> list[tuple[str, str, float | None]]:
    latest = pd.Timestamp(orders["date"].max()).normalize()
    recent = orders[orders["date"].dt.normalize().between(latest - pd.offsets.Day(6), latest)]
    previous = orders[
        orders["date"].dt.normalize().between(
            latest - pd.offsets.Day(13), latest - pd.offsets.Day(7)
        )
    ]
    recent_orders, previous_orders = order_count(recent, caps), order_count(previous, caps)
    recent_sales, previous_sales = float(recent["sales"].sum()), float(previous["sales"].sum())
    recent_aov = recent_sales / recent_orders if recent_orders else 0.0
    previous_aov = previous_sales / previous_orders if previous_orders else 0.0
    recent_users, previous_users = int(recent["user_id"].nunique()), int(previous["user_id"].nunique())
    complete = not previous.empty
    return [
        ("销售额", f"¥{recent_sales:,.0f}", percent_change(recent_sales, previous_sales) if complete else None),
        ("订单量", f"{recent_orders}", percent_change(recent_orders, previous_orders) if complete else None),
        ("客单价", f"¥{recent_aov:,.2f}", percent_change(recent_aov, previous_aov) if complete else None),
        ("用户数", f"{recent_users}", percent_change(recent_users, previous_users) if complete else None),
    ]


def render_kpi_cards(cards: Iterable[tuple[str, str, float | None]]) -> None:
    body = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div>'
        f'<div class="kpi-delta">较上周&nbsp; {_delta_html(delta)}</div></div>'
        for label, value, delta in cards
    )
    st.markdown(f'<div class="kpi-grid">{body}</div>', unsafe_allow_html=True)


def render_kpi_summary(cards: Iterable[tuple[str, str, float | None]]) -> None:
    body = "".join(
        f'<div class="kpi-summary-item"><span class="kpi-summary-value">{escape(value)}</span>'
        f'<span class="kpi-summary-label">{escape(label)}</span>'
        f'<span class="kpi-summary-delta">{_delta_html(delta)}</span></div>'
        for label, value, delta in cards
    )
    if body:
        st.markdown(f'<div class="kpi-summary">{body}</div>', unsafe_allow_html=True)


def render_user_message(content: str) -> None:
    st.markdown(f'<div class="user-bubble">{escape(content)}</div>', unsafe_allow_html=True)


def _inline_markdown(content: str) -> str:
    safe = escape(content)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def _answer_html(content: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    if not paragraphs:
        return ""
    return "".join(
        f"<p>{_inline_markdown(part).replace(chr(10), '<br>')}</p>"
        for part in paragraphs
    )


def _meaningful_diagnoses(items: tuple[str, ...]) -> tuple[str, ...]:
    """仅过滤无信息增量的“由 X 变为 X”，不改动分析结果。"""

    filtered: list[str] = []
    for item in items:
        meaningful_parts: list[str] = []
        for part in re.split(r"[；;]", item):
            cleaned = part.strip().rstrip("。")
            change = re.search(r"^(.*?)由\s*(.*?)\s*变为\s*(.*?)$", cleaned)
            if change:
                before = change.group(2).strip()
                after = change.group(3).strip()
                if before == after:
                    continue
            if cleaned:
                meaningful_parts.append(cleaned)
        if meaningful_parts:
            filtered.append("；".join(meaningful_parts) + "。")
    return tuple(filtered)


def _analysis_entries(
    result: AnalysisResult, agent_response: AgentResponse | None
) -> tuple[tuple[str, AnalysisResult], ...]:
    if agent_response and agent_response.analysis_results:
        return tuple(agent_response.analysis_results)
    return ((result.intent.value, result),)


def _unique_items(*groups: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for group in groups for item in group if item.strip()))


def _novel_items(items: Iterable[str], answer: str, limit: int) -> tuple[str, ...]:
    normalized_answer = re.sub(r"[\s*]", "", answer)
    novel = []
    for item in _meaningful_diagnoses(tuple(items)):
        normalized_item = re.sub(r"[\s*。]", "", item)
        if normalized_item and normalized_item not in normalized_answer:
            novel.append(item)
    return tuple(novel[:limit])


def _render_analysis_lines(title: str, items: Iterable[str]) -> None:
    values = tuple(items)
    if not values:
        return
    rows = "".join(
        f'<div class="analysis-line"><span class="analysis-dot">·</span>'
        f'<span>{escape(item)}</span></div>' for item in values
    )
    st.markdown(
        f'<div class="analysis-section"><div class="analysis-section-title">'
        f'{escape(title)}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def _should_show_suggestions(
    result: AnalysisResult,
    plan: IntentPlan | None,
    entries: tuple[tuple[str, AnalysisResult], ...],
    user_goal: str | None = None,
) -> bool:
    asks_for_advice = any(
        keyword in (user_goal or "")
        for keyword in ("怎么办", "怎么做", "建议", "下一步", "接下来", "应该怎么")
    )
    return (
        result.intent.value == "strategy"
        or (plan is not None and plan.action == "recommend")
        or any(name == "strategy" or item.intent.value == "strategy" for name, item in entries)
        or asks_for_advice
    )


def _visible_limitations(
    entries: tuple[tuple[str, AnalysisResult], ...]
) -> tuple[str, ...]:
    limitations = _unique_items(*(item.limitations for _, item in entries))
    return tuple(
        item for item in limitations
        if item != "销售额和销量不等于利润。"
        or any(analysis.intent.value == "profit" for _, analysis in entries)
    )


def render_analysis(
    result: AnalysisResult,
    orders: pd.DataFrame,
    metrics: BusinessMetrics,
    capabilities: DataCapabilities,
    *,
    answer: str | None = None,
    plan: IntentPlan | None = None,
    fallback_used: bool = False,
    agent_fallback: bool = False,
    agent_response: AgentResponse | None = None,
    user_goal: str | None = None,
    message_key: str = "analysis",
) -> None:
    display_answer = result.conclusion if fallback_used or not answer else answer
    st.markdown(
        f'<div class="assistant-response"><span class="ai-label">✦</span>'
        f'<div class="assistant-answer">{_answer_html(display_answer)}</div></div>',
        unsafe_allow_html=True,
    )
    if fallback_used:
        st.markdown(
            '<div class="fallback-note">AI 理解服务暂时不可用，已切换至基础分析模式。</div>',
            unsafe_allow_html=True,
        )
    elif agent_fallback:
        st.markdown(
            '<div class="fallback-note">Agent 分析暂时不可用，已切换至标准分析模式。</div>',
            unsafe_allow_html=True,
        )
    if agent_response and (agent_response.steps or agent_response.skills_used):
        labels = {
            "analyze_product": "商品表现", "analyze_user": "用户结构",
            "analyze_repurchase": "复购表现", "analyze_profit": "利润表现",
            "analyze_channel": "渠道表现", "analyze_trend": "销售趋势",
            "analyze_strategy": "经营策略", "create_visualization": "可视化",
            "generate_report": "经营报告",
        }
        completed = tuple(
            labels.get(step.tool_name, step.tool_name)
            for step in agent_response.steps if step.supported
        )
        skill_labels = {
            "business-diagnosis": "经营诊断",
        }
        used_skills = tuple(
            skill_labels.get(name, name) for name in agent_response.skills_used
        )
        if completed or used_skills:
            summary = f"✓ 已完成 {len(completed)} 项分析" if completed else "分析过程"
            with st.expander(summary, expanded=False):
                skill_trace = "".join(
                    f"◇ 使用{escape(name)} Skill<br>" for name in used_skills
                )
                st.markdown(
                    '<div class="trace-list">'
                    + skill_trace
                    + "<br>".join(f"✓ {escape(item)}" for item in completed)
                    + "</div>",
                    unsafe_allow_html=True,
                )

    entries = _analysis_entries(result, agent_response)
    if any(item.intent.value == "trend" and item.supported for _, item in entries):
        render_kpi_summary(comparison_kpis(orders, capabilities)[:3])

    charts: dict[str, list] = {}
    if agent_response and agent_response.visualizations:
        result_map = dict(agent_response.analysis_results)
        for spec in agent_response.visualizations:
            chart_result = result_map.get(spec.analysis_type)
            if chart_result is None:
                continue
            chart = build_visualization(
                chart_result.intent, orders, chart_result, capabilities,
                metric=spec.metric, action="visualize", visualization=True,
            )
            if chart is not None:
                charts.setdefault(spec.analysis_type, []).append(chart)
    elif plan is not None and plan.action in {"visualize", "compare"}:
        chart = build_visualization(
            result.intent, orders, result, capabilities, metric=plan.metric,
            action=plan.action, visualization=plan.visualization,
        )
        if chart is not None:
            charts.setdefault(result.intent.value, []).append(chart)

    detailed = len(entries) > 1 or result.intent.value == "trend" or (
        plan is not None and plan.action in {"explain", "compare"}
    )
    labels = {
        "trend": "销售趋势", "product": "商品表现", "channel": "渠道表现",
        "user": "用户表现", "repurchase": "复购表现", "profit": "利润表现",
        "strategy": "经营策略",
    }
    chart_index = 0
    for name, analysis in entries:
        if detailed:
            section_items = []
            if len(entries) > 1 and analysis.conclusion not in display_answer:
                section_items.append(analysis.conclusion)
            section_items.extend(_novel_items(analysis.evidence, display_answer, 3))
            section_items.extend(_novel_items(analysis.reasons, display_answer, 2))
            _render_analysis_lines(labels.get(name, "分析结果"), _unique_items(section_items))
        for chart in charts.get(name, []):
            with st.container(border=True, key=f"viz_{message_key}_{chart_index}"):
                st.markdown(
                    f'<div class="chart-title">{escape(_chart_title(analysis))}</div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(chart, use_container_width=True)
            chart_index += 1

    if _should_show_suggestions(result, plan, entries, user_goal):
        suggestions = _unique_items(*(item.recommendations for _, item in entries))[:3]
        rows = "".join(
            f'<div class="suggestion-row"><span class="suggestion-index">{index}</span>'
            f'<span>{escape(item)}</span></div>'
            for index, item in enumerate(suggestions, 1)
        )
        if rows:
            st.markdown(
                '<div class="suggestions"><div class="analysis-section-title">建议</div>'
                + rows + "</div>",
                unsafe_allow_html=True,
            )

    limitations = _visible_limitations(entries)
    if limitations:
        st.markdown(
            '<div class="limit-box"><strong>数据说明</strong><br>'
            + "<br>".join(escape(item) for item in limitations)
            + "</div>",
            unsafe_allow_html=True,
        )


def _chart_title(result: AnalysisResult) -> str:
    return {
        "trend": "销售额趋势（最近 14 天）",
        "product": "商品销售表现",
        "profit": "商品利润与毛利率",
        "channel": "渠道销售表现",
        "repurchase": "人群复购表现",
        "user": "用户结构与消费贡献",
    }.get(result.intent.value, "数据分析")


def render_empty_state() -> None:
    st.markdown(
        '<div class="empty-state"><div class="empty-orb">✦</div>'
        '<h2>从经营问题开始</h2><p>我会根据当前数据库自动计算指标并选择合适的可视化。</p></div>',
        unsafe_allow_html=True,
    )


def render_data_start_state() -> None:
    st.markdown(
        '<div class="data-start-state"><div class="empty-orb">✦</div>'
        '<h1>从数据开始</h1><p>创建一个数据库后，就可以直接向数据提问。</p></div>',
        unsafe_allow_html=True,
    )
