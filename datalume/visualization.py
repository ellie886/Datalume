"""按分析意图自动选择 Datalume 可视化。"""

from __future__ import annotations

from typing import Literal

import altair as alt
import pandas as pd

from .analysis_engine import AnalysisResult
from .data_processor import DataCapabilities
from .router import QuestionType


VisualizationKind = Literal["line", "bar", "profit", "channel", "repurchase", "user", "none"]

PALETTE = ("#E99BC0", "#E9AB78", "#D4C183", "#82B5A1", "#7DA8E7", "#B6A0DA", "#CC9CB5")


def visualization_kind(intent: QuestionType) -> VisualizationKind:
    mapping: dict[QuestionType, VisualizationKind] = {
        QuestionType.TREND: "line",
        QuestionType.PRODUCT: "bar",
        QuestionType.PROFIT: "profit",
        QuestionType.CHANNEL: "channel",
        QuestionType.REPURCHASE: "repurchase",
        QuestionType.USER: "user",
        QuestionType.STRATEGY: "none",
    }
    return mapping[intent]


def style_chart(chart: alt.Chart, *, height: int = 232) -> alt.Chart:
    """Apply the dark presentation theme without changing data or encodings."""

    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#FFFFFF",
            gridOpacity=0.08,
            gridDash=[2, 4],
            domain=False,
            tickColor="#59545F",
            tickSize=3,
            labelColor="#B8B2C2",
            titleColor="#B8B2C2",
            labelFontSize=11,
            labelPadding=8,
            titleFontSize=11,
            titlePadding=12,
        )
        .configure_legend(
            labelColor="#C6C0CF", labelFontSize=11, title=None,
            orient="top-right", direction="horizontal", symbolType="circle", symbolSize=65,
        )
    )


def _trend_chart(orders: pd.DataFrame) -> alt.Chart:
    daily = orders.groupby(orders["date"].dt.normalize())["sales"].sum()
    latest = pd.Timestamp(orders["date"].max()).normalize()
    dates = pd.date_range(latest - pd.offsets.Day(13), latest, freq="D")
    daily = daily.reindex(dates, fill_value=0.0).rename_axis("date").reset_index()
    recent_start = latest - pd.offsets.Day(6)
    daily["周期"] = daily["date"].map(
        lambda value: "最近一周" if pd.Timestamp(value).normalize() >= recent_start else "上一周"
    )
    base = alt.Chart(daily).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%-m/%-d", labelAngle=0, tickCount=7, labelOverlap=True)),
        y=alt.Y("sales:Q", title=None, scale=alt.Scale(zero=True)),
        color=alt.Color(
            "周期:N",
            scale=alt.Scale(domain=["上一周", "最近一周"], range=[PALETTE[0], PALETTE[4]]),
        ),
        strokeDash=alt.StrokeDash(
            "周期:N",
            scale=alt.Scale(domain=["上一周", "最近一周"], range=[[5, 3], [1, 0]]),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="日期"),
            alt.Tooltip("sales:Q", title="销售额（¥）", format=",.2f"),
        ],
    )
    return style_chart(base.mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2))


def _product_chart(result: AnalysisResult, metric: str | None = None) -> alt.Chart | None:
    use_quantity = metric == "quantity" or (
        metric is None and "销售件数" in result.conclusion and result.metrics.get("product_quantity")
    )
    values = result.metrics.get("product_quantity" if use_quantity else "product_revenue", {})
    if not values:
        return None
    frame = pd.DataFrame({"商品": list(values), "数值": list(values.values())})
    label = "销量" if use_quantity else "销售额"
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[4]).encode(
        x=alt.X("数值:Q", title=None),
        y=alt.Y(
            "商品:N",
            sort=alt.SortField(field="数值", order="descending"),
            title=None,
        ),
        tooltip=["商品:N", alt.Tooltip("数值:Q", title=label, format=",.2f")],
    )
    return style_chart(chart)


def _profit_chart(result: AnalysisResult) -> alt.Chart | None:
    values = result.metrics.get("product_profit", {})
    if not values:
        return None
    frame = pd.DataFrame.from_dict(values, orient="index").reset_index(names="商品")
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[1]).encode(
        x=alt.X("商品:N", title=None, sort="-y"),
        y=alt.Y("profit:Q", title="利润（¥）"),
        tooltip=[
            "商品:N",
            alt.Tooltip("revenue:Q", title="销售额（¥）", format=",.2f"),
            alt.Tooltip("cost:Q", title="成本（¥）", format=",.2f"),
            alt.Tooltip("profit:Q", title="利润（¥）", format=",.2f"),
            alt.Tooltip("margin:Q", title="毛利率", format=".1%"),
        ],
    )
    line = alt.Chart(frame).mark_line(point=True, color=PALETTE[5]).encode(
        x=alt.X("商品:N", sort="-y"),
        y=alt.Y("margin:Q", title="毛利率", axis=alt.Axis(format="%", orient="right")),
    )
    return style_chart(alt.layer(bars, line).resolve_scale(y="independent"))


def _channel_chart(result: AnalysisResult, metric: str | None = None) -> alt.Chart | None:
    values = result.metrics.get("channel_summary", {})
    if not values:
        return None
    frame = pd.DataFrame.from_dict(values, orient="index").reset_index(names="渠道")
    field = {
        "users": "users", "orders": "orders", "avg_order_value": "aov",
        "repurchase_rate": "repeat_rate", "revenue": "sales",
    }.get(metric, "sales")
    field_title = {
        "users": "用户数", "orders": "订单量", "aov": "客单价（¥）",
        "repeat_rate": "复购率", "sales": "销售额（¥）",
    }[field]
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[1]).encode(
        x=alt.X("渠道:N", title=None, sort="-y"),
        y=alt.Y(f"{field}:Q", title=field_title),
        tooltip=[
            "渠道:N",
            alt.Tooltip("users:Q", title="用户数"),
            alt.Tooltip("orders:Q", title="订单量"),
            alt.Tooltip("sales:Q", title="销售额（¥）", format=",.2f"),
            alt.Tooltip("aov:Q", title="客单价（¥）", format=",.2f"),
        ],
    )
    return style_chart(chart)


def _repurchase_chart(result: AnalysisResult) -> alt.Chart | None:
    groups = result.metrics.get("group_repeat_rates", {}).get("grade", {})
    if groups:
        frame = pd.DataFrame({"年级": list(groups), "复购占比": list(groups.values())})
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[5]).encode(
            x=alt.X("年级:N", title=None),
            y=alt.Y("复购占比:Q", title=None, axis=alt.Axis(format="%")),
            tooltip=["年级:N", alt.Tooltip("复购占比:Q", format=".1%")],
        )
        return style_chart(chart)
    users = result.metrics.get("user_rfm", [])[:8]
    if not users:
        return None
    frame = pd.DataFrame(users)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[5]).encode(
        x=alt.X("user_id:N", title=None),
        y=alt.Y("orders:Q", title=None),
        tooltip=["user_id:N", "orders:Q", "sales:Q", "recency_days:Q"],
    )
    return style_chart(chart)


def _user_chart(orders: pd.DataFrame) -> alt.Chart:
    frame = orders.groupby("grade", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("grade:N", title=None, sort="-y"),
        y=alt.Y("sales:Q", title=None),
        color=alt.Color("grade:N", scale=alt.Scale(range=list(PALETTE)), legend=None),
        tooltip=[alt.Tooltip("grade:N", title="年级"), alt.Tooltip("sales:Q", title="销售额", format=",.2f")],
    )
    return style_chart(chart)


def build_visualization(
    intent: QuestionType,
    orders: pd.DataFrame,
    result: AnalysisResult,
    capabilities: DataCapabilities,
    *,
    metric: str | None = None,
    action: str | None = None,
    visualization: bool | None = None,
) -> alt.Chart | None:
    """由意图自动选择图表；不向用户暴露手动图表选择。"""

    if not result.supported:
        return None
    if visualization is False and action not in {"visualize", "compare"}:
        return None
    kind = visualization_kind(intent)
    if kind == "line":
        return _trend_chart(orders)
    if kind == "bar":
        return _product_chart(result, metric)
    if kind == "profit":
        return _profit_chart(result)
    if kind == "channel":
        return _channel_chart(result, metric)
    if kind == "repurchase":
        return _repurchase_chart(result)
    if kind == "user":
        return _user_chart(orders)
    return None
