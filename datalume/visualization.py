"""按分析意图自动选择 Datalume 可视化。"""

from __future__ import annotations

from typing import Literal

import altair as alt
import pandas as pd

from .analysis_engine import AnalysisResult
from .data_processor import DataCapabilities
from .router import QuestionType


VisualizationKind = Literal["line", "bar", "profit", "channel", "repurchase", "user", "none"]

PALETTE = ("#D96C75", "#D99A5B", "#D7BE63", "#62A77B", "#5B83B4", "#8870B2", "#C77A9B")


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


def _style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(height=205)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#F1F1EF",
            gridOpacity=0.7,
            domainColor="#E1E1DE",
            tickColor="#E1E1DE",
            labelColor="#666666",
            titleColor="#666666",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(labelColor="#666666", labelFontSize=11, title=None, orient="top-right")
    )


def _trend_chart(orders: pd.DataFrame) -> alt.Chart:
    daily = orders.groupby("date", as_index=False)["sales"].sum().sort_values("date").tail(14)
    latest = pd.Timestamp(daily["date"].max()).normalize()
    recent_start = latest - pd.offsets.Day(6)
    daily["周期"] = daily["date"].map(
        lambda value: "最近一周" if pd.Timestamp(value).normalize() >= recent_start else "上一周"
    )
    base = alt.Chart(daily).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%-m/%-d", labelAngle=0)),
        y=alt.Y("sales:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "周期:N",
            scale=alt.Scale(domain=["上一周", "最近一周"], range=[PALETTE[0], PALETTE[3]]),
        ),
        tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("sales:Q", title="销售额", format=",.2f")],
    )
    return _style(base.mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2))


def _product_chart(result: AnalysisResult, metric: str | None = None) -> alt.Chart | None:
    use_quantity = metric == "quantity" or (
        metric is None and "销售件数" in result.conclusion and result.metrics.get("product_quantity")
    )
    values = result.metrics.get("product_quantity" if use_quantity else "product_revenue", {})
    if not values:
        return None
    frame = pd.DataFrame({"商品": list(values), "数值": list(values.values())}).sort_values("数值")
    label = "销量" if use_quantity else "销售额"
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[4]).encode(
        x=alt.X("数值:Q", title=None),
        y=alt.Y("商品:N", sort=None, title=None),
        tooltip=["商品:N", alt.Tooltip("数值:Q", title=label, format=",.2f")],
    )
    return _style(chart)


def _profit_chart(result: AnalysisResult) -> alt.Chart | None:
    values = result.metrics.get("product_profit", {})
    if not values:
        return None
    frame = pd.DataFrame.from_dict(values, orient="index").reset_index(names="商品")
    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[3]).encode(
        x=alt.X("商品:N", title=None, sort="-y"),
        y=alt.Y("profit:Q", title=None),
        tooltip=["商品:N", "revenue:Q", "cost:Q", "profit:Q", alt.Tooltip("margin:Q", format=".1%")],
    )
    line = alt.Chart(frame).mark_line(point=True, color=PALETTE[5]).encode(
        x=alt.X("商品:N", sort="-y"),
        y=alt.Y("margin:Q", title=None, axis=alt.Axis(format="%", orient="right")),
    )
    return _style(alt.layer(bars, line).resolve_scale(y="independent"))


def _channel_chart(result: AnalysisResult, metric: str | None = None) -> alt.Chart | None:
    values = result.metrics.get("channel_summary", {})
    if not values:
        return None
    frame = pd.DataFrame.from_dict(values, orient="index").reset_index(names="渠道")
    field = {
        "users": "users", "orders": "orders", "avg_order_value": "aov",
        "repurchase_rate": "repeat_rate", "revenue": "sales",
    }.get(metric, "sales")
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[1]).encode(
        x=alt.X("渠道:N", title=None, sort="-y"),
        y=alt.Y(f"{field}:Q", title=None),
        tooltip=["渠道:N", "users:Q", "orders:Q", "sales:Q", "aov:Q"],
    )
    return _style(chart)


def _repurchase_chart(result: AnalysisResult) -> alt.Chart | None:
    groups = result.metrics.get("group_repeat_rates", {}).get("grade", {})
    if groups:
        frame = pd.DataFrame({"年级": list(groups), "复购占比": list(groups.values())})
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[5]).encode(
            x=alt.X("年级:N", title=None),
            y=alt.Y("复购占比:Q", title=None, axis=alt.Axis(format="%")),
            tooltip=["年级:N", alt.Tooltip("复购占比:Q", format=".1%")],
        )
        return _style(chart)
    users = result.metrics.get("user_rfm", [])[:8]
    if not users:
        return None
    frame = pd.DataFrame(users)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=PALETTE[5]).encode(
        x=alt.X("user_id:N", title=None),
        y=alt.Y("orders:Q", title=None),
        tooltip=["user_id:N", "orders:Q", "sales:Q", "recency_days:Q"],
    )
    return _style(chart)


def _user_chart(orders: pd.DataFrame) -> alt.Chart:
    frame = orders.groupby("grade", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("grade:N", title=None, sort="-y"),
        y=alt.Y("sales:Q", title=None),
        color=alt.Color("grade:N", scale=alt.Scale(range=list(PALETTE)), legend=None),
        tooltip=[alt.Tooltip("grade:N", title="年级"), alt.Tooltip("sales:Q", title="销售额", format=",.2f")],
    )
    return _style(chart)


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
