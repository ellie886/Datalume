"""Limited, deterministic summaries for user-confirmed partial data mappings.

This module does not use the order-analysis engine or infer missing fields.  It
expects canonical column names (date, revenue, channel) selected by the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class PartialSummary:
    row_count: int
    total_revenue: float | None
    daily_revenue: pd.DataFrame | None
    channel_revenue: pd.DataFrame | None
    issues: tuple[str, ...]


def _parse_date(value: object) -> pd.Timestamp | None:
    """Accept real date cells and unambiguous year-first dates only."""
    if pd.isna(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        parsed = pd.Timestamp(value)
        return parsed.tz_localize(None).normalize() if parsed.tzinfo else parsed.normalize()
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^(\d{4})年(\d{1,2})月(\d{1,2})日", r"\1-\2-\3", text)
    if not re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?", text):
        return None
    try:
        return pd.Timestamp(text).normalize()
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_revenue(value: object) -> float | None:
    """Do not infer currency, separators, percentages or units."""
    if pd.isna(value) or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
                return None
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return amount if math.isfinite(amount) else None


def build_partial_summary(frame: pd.DataFrame) -> PartialSummary:
    """Compute only supported complete aggregates; never discard invalid rows.

    Revenue, date, and channel are validated independently.  A missing or bad
    date blocks daily results but does not block an otherwise valid total or
    channel summary.  A missing or bad revenue blocks all monetary summaries.
    """
    if not frame.columns.is_unique:
        return PartialSummary(len(frame), None, None, None, ("存在重名字段，请先确认字段映射。",))
    count = len(frame)
    if not count:
        return PartialSummary(0, None, None, None, ("没有可供分析的数据。",))

    issues: list[str] = []
    amounts: pd.Series | None = None
    if "revenue" not in frame:
        issues.append("未提供销售额字段，无法汇总销售额。")
    else:
        parsed_amounts = frame["revenue"].map(_parse_revenue)
        bad_amounts = int(parsed_amounts.isna().sum())
        if bad_amounts:
            issues.append(f"销售额有 {bad_amounts} 行为空或无法识别，请先核对；暂不计算销售额。")
        else:
            amounts = parsed_amounts.astype(float)

    dates: pd.Series | None = None
    if "date" not in frame:
        issues.append("未提供日期字段，无法展示销售趋势。")
    else:
        parsed_dates = frame["date"].map(_parse_date)
        bad_dates = int(parsed_dates.isna().sum())
        if bad_dates:
            issues.append(f"日期有 {bad_dates} 行为空或无法识别；请使用年-月-日格式核对后再看趋势。")
        else:
            dates = pd.to_datetime(parsed_dates)

    channels: pd.Series | None = None
    if "channel" not in frame:
        issues.append("未提供渠道字段，无法比较渠道销售额。")
    else:
        parsed_channels = frame["channel"].astype("string").str.strip()
        bad_channels = int((parsed_channels.isna() | parsed_channels.eq("")).sum())
        if bad_channels:
            issues.append(f"渠道有 {bad_channels} 行为空；请先核对后再做渠道比较。")
        else:
            channels = parsed_channels

    total: float | None = None
    daily: pd.DataFrame | None = None
    by_channel: pd.DataFrame | None = None
    if amounts is not None:
        total = float(amounts.sum())
        if not math.isfinite(total):
            issues.append("销售额汇总超出可表示范围，请检查数据。")
            amounts = None
            total = None
    if amounts is not None and dates is not None:
        daily = pd.DataFrame({"日期": dates, "销售额": amounts}).groupby("日期", as_index=False, sort=True)["销售额"].sum()
        if not daily["销售额"].map(math.isfinite).all():
            daily = None
            issues.append("按日汇总超出可表示范围，请检查数据。")
    if amounts is not None and channels is not None:
        by_channel = pd.DataFrame({"渠道": channels, "销售额": amounts}).groupby("渠道", as_index=False, sort=False)["销售额"].sum().sort_values("销售额", ascending=False, ignore_index=True)
        if not by_channel["销售额"].map(math.isfinite).all():
            by_channel = None
            issues.append("渠道汇总超出可表示范围，请检查数据。")
    return PartialSummary(count, total, daily, by_channel, tuple(issues))


def render_partial_analysis(frame: pd.DataFrame) -> None:
    """Render this deliberately limited mode, without invoking the AI agent."""
    result = build_partial_summary(frame)
    st.subheader("基础数据分析")
    st.caption("仅按已确认的字段进行汇总；金额单位与原表一致。")
    st.caption(f"数据行数：{result.row_count}，按确认后的数据汇总。")
    if result.total_revenue is not None:
        st.metric("销售额合计", f"{result.total_revenue:,.2f}")
    if result.daily_revenue is not None:
        st.markdown("**销售额趋势**")
        st.line_chart(result.daily_revenue.set_index("日期"))
        st.dataframe(result.daily_revenue, hide_index=True, use_container_width=True)
    if result.channel_revenue is not None:
        st.markdown("**渠道销售额**")
        st.bar_chart(result.channel_revenue.set_index("渠道"))
        st.dataframe(result.channel_revenue, hide_index=True, use_container_width=True)
    for issue in result.issues:
        st.info(issue)
    st.caption("此模式暂不提供利润、订单量、客单价或复购分析，也不会调用 AI Agent。")
