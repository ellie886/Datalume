"""Datalume 真实数据分析引擎。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data_processor import BusinessMetrics, DataCapabilities
from .router import QuestionType


def money(value: float) -> str:
    return f"¥{value:,.2f}"


def percent_change(current: float, previous: float) -> float | None:
    return None if previous == 0 else (current - previous) / previous


def change_text(value: float | None) -> str:
    if value is None:
        return "上一周期为 0，无法计算百分比"
    direction = "上升" if value > 0 else "下降" if value < 0 else "持平"
    return f"{direction} {abs(value):.1%}"


@dataclass(frozen=True)
class AnalysisResult:
    intent: QuestionType
    supported: bool
    conclusion: str
    evidence: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        sections = [f"### 结论\n\n{self.conclusion}"]
        for title, items in (("数据证据", self.evidence), ("原因分析", self.reasons), ("可执行建议", self.recommendations), ("数据限制", self.limitations)):
            if items:
                sections.append(f"### {title}\n\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(sections)


def order_count(orders: pd.DataFrame, caps: DataCapabilities) -> int:
    return int(orders[caps.order_id_field].nunique()) if caps.repurchase_reliable and caps.order_id_field else len(orders)


def user_order_counts(orders: pd.DataFrame, caps: DataCapabilities) -> pd.Series:
    return orders.groupby("user_id")[caps.order_id_field].nunique() if caps.repurchase_reliable and caps.order_id_field else orders.groupby("user_id").size()


def product_analysis(question: str, orders: pd.DataFrame | None, metrics: BusinessMetrics) -> AnalysisResult:
    revenue = metrics.product_sales.astype(float)
    quantity = orders.groupby("product")["quantity"].sum().sort_values(ascending=False) if orders is not None else pd.Series(dtype=float)
    asks_volume = any(keyword in question for keyword in ("销量", "件数", "卖得最多"))
    if asks_volume and not quantity.empty:
        top = str(quantity.index[0]); conclusion = f"按销售件数计算，销量最高的商品是 **{top}**，共售出 **{int(quantity.iloc[0])} 件**。"
    else:
        top = metrics.top_product; conclusion = f"按销售额计算，排名第一的是 **{top}**，销售额为 **{money(float(revenue.iloc[0]))}**。"
    evidence = []
    for index, product in enumerate(revenue.index[:5], 1):
        sales = float(revenue.loc[product]); share = sales / metrics.total_sales if metrics.total_sales else 0
        qty = f"、销量 {int(quantity.get(product, 0))} 件" if not quantity.empty else ""
        evidence.append(f"第 {index} 名 {product}：销售额 {money(sales)}{qty}、贡献率 {share:.1%}。")
    recommendation = f"推广方案：围绕 {top} 做小规模套餐或社群测试，并观察曝光、转化、客单价和贡献率。" if "推广" in question else f"保持 {top} 的库存与曝光，同时观察销量、销售额、贡献率和利润。"
    return AnalysisResult(QuestionType.PRODUCT, True, conclusion, tuple(evidence), ("排名来自当前数据的 price × quantity 汇总。",), (recommendation,), ("销售额和销量不等于利润。",), {"top_product": top, "product_revenue": revenue.to_dict(), "product_quantity": quantity.to_dict()})


def user_analysis(orders: pd.DataFrame | None, metrics: BusinessMetrics) -> AnalysisResult:
    evidence = [f"当前共有 {metrics.unique_users} 位用户。", f"主要性别人群为 {metrics.top_gender}性。", f"销售贡献最高的年级为 {metrics.top_grade}，贡献 {money(float(metrics.grade_sales.iloc[0]))}。"]
    top_users = []
    if orders is not None:
        sales = orders.groupby("user_id")["sales"].sum().sort_values(ascending=False)
        top_users = [(str(user), float(value)) for user, value in sales.head(5).items()]
        evidence.append("高消费用户：" + "；".join(f"{user} {money(value)}" for user, value in top_users))
    return AnalysisResult(QuestionType.USER, True, f"当前高价值人群首先集中在 **{metrics.top_grade}**。", tuple(evidence), ("高价值年级按销售贡献、高价值用户按累计消费识别。",), (f"优先验证 {metrics.top_grade} 用户的复购和活动响应。",), ("画像只描述已成交用户。",), {"top_users": top_users})


def repurchase_analysis(orders: pd.DataFrame | None, metrics: BusinessMetrics, caps: DataCapabilities | None) -> AnalysisResult:
    if orders is None or caps is None:
        return AnalysisResult(QuestionType.REPURCHASE, False, "当前调用没有订单明细，无法计算用户复购。", limitations=("至少需要 user_id、date 和订单明细。",))
    counts = user_order_counts(orders, caps).sort_values(ascending=False)
    repeat_users = counts[counts > 1]; repeat_rate = len(repeat_users) / len(counts) if len(counts) else 0
    user_sales = orders.groupby("user_id")["sales"].sum(); first = orders.groupby("user_id")["date"].min(); last = orders.groupby("user_id")["date"].max(); latest = pd.Timestamp(orders["date"].max())
    rfm = [{"user_id": str(user), "orders": int(freq), "sales": float(user_sales.loc[user]), "first_purchase": pd.Timestamp(first.loc[user]).strftime("%Y-%m-%d"), "last_purchase": pd.Timestamp(last.loc[user]).strftime("%Y-%m-%d"), "recency_days": int((latest - pd.Timestamp(last.loc[user])).days)} for user, freq in counts.head(8).items()]
    evidence = [f"共 {len(counts)} 位用户，其中 {len(repeat_users)} 位购买超过 1 次；重复购买用户占比为 {repeat_rate:.1%}。"]
    if rfm:
        evidence.append("购买次数领先用户：" + "；".join(f"{item['user_id']} {item['orders']} 次 / {money(item['sales'])}" for item in rfm[:5]))
    group_rates = {}
    for dimension in ("grade", "gender"):
        dimension_by_user = orders.groupby("user_id")[dimension].first()
        summary = pd.DataFrame({"orders": counts, dimension: dimension_by_user}).groupby(dimension)["orders"].agg(users="size", repeat_users=lambda values: int((values > 1).sum()))
        summary["repeat_rate"] = summary["repeat_users"] / summary["users"]
        group_rates[dimension] = {str(group): float(row["repeat_rate"]) for group, row in summary.iterrows()}
        evidence.append(f"{dimension} 维度重复购买占比最高的是 {summary['repeat_rate'].idxmax()}：{summary['repeat_rate'].max():.1%}。")
    if caps.repurchase_reliable:
        conclusion = f"当前整体复购用户率为 **{repeat_rate:.1%}**。"; limitations = ()
    else:
        conclusion = f"按“每行代表一笔订单”估算，重复购买用户占比为 **{repeat_rate:.1%}**；这不是可靠的正式复购率。"; limitations = ("缺少 order_id，无法区分订单明细和多次购买。",)
    return AnalysisResult(QuestionType.REPURCHASE, True, conclusion, tuple(evidence), ("频次、最近购买和消费金额共同形成基础 RFM 证据。",), ("针对高频且最近活跃的用户做复购活动。",), limitations, {"repeat_user_rate": repeat_rate, "repeat_users": len(repeat_users), "user_rfm": rfm, "group_repeat_rates": group_rates, "reliable": caps.repurchase_reliable})


def profit_analysis(question: str, orders: pd.DataFrame | None, metrics: BusinessMetrics, caps: DataCapabilities | None) -> AnalysisResult:
    if orders is None or caps is None or not caps.profit_supported:
        return AnalysisResult(QuestionType.PROFIT, False, "当前数据无法判断利润，因为缺少成本数据或可用利润字段。", (f"当前只能确认销售额最高的商品是 {metrics.top_product}。",), recommendations=("上传 unit_cost、cost/total_cost 或 profit 字段后再比较。",), limitations=("禁止把销售额最高直接等同于利润最高。",))
    work = orders[["product", "sales", "quantity"]].copy(); assumptions = []
    if caps.profit_field:
        work["profit"] = pd.to_numeric(orders[caps.profit_field]); work["cost"] = work["sales"] - work["profit"]
    elif caps.unit_cost_field:
        work["cost"] = pd.to_numeric(orders[caps.unit_cost_field]) * work["quantity"]; work["profit"] = work["sales"] - work["cost"]
    else:
        work["cost"] = pd.to_numeric(orders[caps.total_cost_field]); work["profit"] = work["sales"] - work["cost"]
        if caps.total_cost_field and caps.total_cost_field.lower() == "cost": assumptions.append("cost 按每行总成本处理；单位成本请使用 unit_cost。")
    summary = work.groupby("product").agg(revenue=("sales", "sum"), cost=("cost", "sum"), profit=("profit", "sum")); summary["margin"] = summary["profit"] / summary["revenue"].replace(0, pd.NA); summary = summary.sort_values("profit", ascending=False)
    top_profit = str(summary.index[0]); quantity = orders.groupby("product")["quantity"].sum().sort_values(ascending=False); volume = any(key in question for key in ("销量", "件数", "卖得最多")); comparison = str(quantity.index[0]) if volume else metrics.top_product; label = "销量" if volume else "销售额"
    conclusion = f"{label}最高的 {comparison} 同时也是利润最高的商品。" if comparison == top_profit else f"{label}最高的 {comparison} 并不是利润最高；利润最高的是 **{top_profit}**。"
    evidence = []
    for product, row in summary.iterrows():
        margin = "无法计算" if pd.isna(row.margin) else f"{float(row.margin):.1%}"
        evidence.append(f"{product}：收入 {money(float(row.revenue))}、成本 {money(float(row.cost))}、利润 {money(float(row.profit))}、毛利率 {margin}。")
    return AnalysisResult(QuestionType.PROFIT, True, conclusion, tuple(evidence), ("利润按收入减成本并按商品汇总。",), (f"结合销量和毛利评估 {top_profit}。",), tuple(assumptions), {"comparison_product": comparison, "top_profit_product": top_profit, "product_profit": summary.to_dict("index")})


def channel_analysis(orders: pd.DataFrame | None, caps: DataCapabilities | None) -> AnalysisResult:
    if orders is None or caps is None or not caps.channel_supported:
        return AnalysisResult(QuestionType.CHANNEL, False, "当前数据无法回答哪个获客渠道效果最好，因为缺少渠道字段。", recommendations=("上传 channel、source 或 acquisition_channel 字段后再比较。",), limitations=("没有渠道字段时不会用通用建议替代渠道结论。",))
    channel = caps.channel_field; grouped = orders.groupby(channel); sales = grouped["sales"].sum(); users = grouped["user_id"].nunique(); counts = grouped[caps.order_id_field].nunique() if caps.repurchase_reliable and caps.order_id_field else grouped.size(); summary = pd.DataFrame({"sales": sales, "orders": counts, "users": users}); summary["aov"] = summary["sales"] / summary["orders"]
    user_counts = orders.groupby([channel, "user_id"])[caps.order_id_field].nunique() if caps.repurchase_reliable and caps.order_id_field else orders.groupby([channel, "user_id"]).size(); summary["repeat_rate"] = user_counts.groupby(level=0).apply(lambda values: float((values > 1).mean()))
    limitations = [] if caps.repurchase_reliable else ["渠道复购按每行一单估算。"]
    if caps.channel_cost_supported and caps.acquisition_cost_field:
        costs = pd.to_numeric(orders[caps.acquisition_cost_field]).groupby(orders[channel]).sum(); summary["acquisition_cost"] = costs; summary["cac"] = costs / summary["users"].replace(0, pd.NA); summary["roi"] = (summary["sales"] - costs) / costs.replace(0, pd.NA); limitations.append("ROI 未扣除商品成本时不等同于利润 ROI。")
    else: limitations.append("缺少获客成本字段，无法计算 CAC 或 ROI。")
    best = str(summary["sales"].idxmax()); evidence = []
    for name, row in summary.sort_values("sales", ascending=False).iterrows():
        detail = f"{name}：用户 {int(row.users)}、订单 {int(row.orders)}、销售额 {money(float(row.sales))}、客单价 {money(float(row.aov))}、重复购买占比 {float(row.repeat_rate):.1%}"
        if "cac" in summary: detail += f"、CAC {'无法计算' if pd.isna(row.cac) else money(float(row.cac))}、ROI {'无法计算' if pd.isna(row.roi) else f'{float(row.roi):.1%}'}"
        evidence.append(detail + "。")
    return AnalysisResult(QuestionType.CHANNEL, True, f"按当前销售额指标，表现最高的渠道是 **{best}**。", tuple(evidence), ("渠道效果综合展示用户、订单、销售额、客单价和可用成本指标。",), ("继续对比渠道转化、获客成本和后续复购。",), tuple(limitations), {"best_channel": best, "channel_summary": summary.to_dict("index")})


def trend_analysis(orders: pd.DataFrame | None, metrics: BusinessMetrics, caps: DataCapabilities | None) -> AnalysisResult:
    if orders is None or caps is None:
        return AnalysisResult(QuestionType.TREND, False, "当前调用没有订单明细，无法拆解销售变化。")
    latest = pd.Timestamp(orders["date"].max()).normalize(); recent_start = latest - pd.offsets.Day(6); previous_end = recent_start - pd.offsets.Day(1); previous_start = previous_end - pd.offsets.Day(6); earliest = pd.Timestamp(orders["date"].min()).normalize()
    if earliest > previous_start:
        days = (latest - earliest).days + 1
        return AnalysisResult(QuestionType.TREND, False, "当前数据周期不足，无法比较最近一周与上一完整周，因此不能判断最近一周销售额为何下降。", (f"数据范围只有 {days} 天。",), recommendations=("至少提供连续 14 天数据。",), limitations=("周期不足时不会推断下降原因。",))
    recent = orders[orders["date"].dt.normalize().between(recent_start, latest)]; previous = orders[orders["date"].dt.normalize().between(previous_start, previous_end)]
    rs, ps = float(recent["sales"].sum()), float(previous["sales"].sum()); ro, po = order_count(recent, caps), order_count(previous, caps); ra, pa = rs / ro if ro else 0, ps / po if po else 0; ru, pu = int(recent["user_id"].nunique()), int(previous["user_id"].nunique()); sc, oc, ac, uc = percent_change(rs, ps), percent_change(ro, po), percent_change(ra, pa), percent_change(ru, pu)
    rp = recent.groupby("product")["sales"].sum().sort_values(ascending=False); pp = previous.groupby("product")["sales"].sum().sort_values(ascending=False); rg = recent.groupby("grade")["sales"].sum().sort_values(ascending=False); pg = previous.groupby("grade")["sales"].sum().sort_values(ascending=False)
    evidence = [f"最近一周：销售额 {money(rs)}、订单 {ro}、客单价 {money(ra)}、用户 {ru}。", f"上一周：销售额 {money(ps)}、订单 {po}、客单价 {money(pa)}、用户 {pu}。", f"销售额{change_text(sc)}；订单量{change_text(oc)}；客单价{change_text(ac)}；用户数{change_text(uc)}。", f"主力商品由 {pp.index[0]} 变为 {rp.index[0]}；主要年级由 {pg.index[0]} 变为 {rg.index[0]}。"]
    if caps.channel_supported and caps.channel_field:
        rc = recent.groupby(caps.channel_field)["sales"].sum().sort_values(ascending=False); pc = previous.groupby(caps.channel_field)["sales"].sum().sort_values(ascending=False); evidence.append(f"主渠道由 {pc.index[0]} 变为 {rc.index[0]}。")
    limitations = []
    if not caps.repurchase_reliable: limitations.append("缺少 order_id，订单量按数据行估算。")
    if not caps.channel_supported: limitations.append("缺少渠道字段，无法拆解渠道结构变化。")
    return AnalysisResult(QuestionType.TREND, True, f"最近一周销售额相较上一周 **{change_text(sc)}**。应先看订单量与客单价拆解，不能直接归因。", tuple(evidence), ("销售额变化可拆解为订单量变化 × 客单价变化；结构变化只说明同期关联。",), ("订单下降时检查流量/转化，客单价下降时检查商品组合和优惠门槛。",), tuple(limitations), {"recent_sales": rs, "previous_sales": ps, "sales_change": sc, "recent_orders": ro, "previous_orders": po, "recent_aov": ra, "previous_aov": pa, "recent_users": ru, "previous_users": pu})


def strategy_analysis(metrics: BusinessMetrics, caps: DataCapabilities | None) -> AnalysisResult:
    recommendations = [f"围绕 {metrics.top_product} 设计组合套餐并观察客单价和贡献率。", f"在 {metrics.top_grade} 用户中小规模测试活动。"]
    limitations = []
    if caps and not caps.profit_supported: limitations.append("缺少成本/利润字段，策略未经过盈利能力校验。")
    if caps and not caps.channel_supported: limitations.append("缺少渠道字段，无法分配渠道预算。")
    return AnalysisResult(QuestionType.STRATEGY, True, f"当前应以主力商品 {metrics.top_product} 和高价值年级 {metrics.top_grade} 为切入点。", (f"总销售额 {money(metrics.total_sales)}，订单 {metrics.order_count}，客单价 {money(metrics.avg_order_value)}。",), ("建议来自当前真实指标。",), tuple(recommendations), tuple(limitations))


def run_analysis(intent: QuestionType, question: str, orders: pd.DataFrame | None, metrics: BusinessMetrics, caps: DataCapabilities | None) -> AnalysisResult:
    return {
        QuestionType.PRODUCT: lambda: product_analysis(question, orders, metrics),
        QuestionType.USER: lambda: user_analysis(orders, metrics),
        QuestionType.REPURCHASE: lambda: repurchase_analysis(orders, metrics, caps),
        QuestionType.PROFIT: lambda: profit_analysis(question, orders, metrics, caps),
        QuestionType.CHANNEL: lambda: channel_analysis(orders, caps),
        QuestionType.TREND: lambda: trend_analysis(orders, metrics, caps),
        QuestionType.STRATEGY: lambda: strategy_analysis(metrics, caps),
    }[intent]()
