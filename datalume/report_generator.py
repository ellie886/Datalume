"""基于当前订单数据生成可下载的结构化经营诊断报告。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_processor import (
    BusinessMetrics,
    DataCapabilities,
    DataOverview,
    detect_data_capabilities,
)


def _money(value: float) -> str:
    return f"¥{value:,.2f}"


@dataclass(frozen=True)
class ProductPerformance:
    product: str
    sales: float
    quantity: int
    contribution: float


@dataclass(frozen=True)
class ActionRecommendation:
    action: str
    reason: str
    metrics_to_watch: str


@dataclass(frozen=True)
class DiagnosticReport:
    total_sales: float
    order_count: int
    avg_order_value: float
    user_count: int
    product_count: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    top_product: str
    top_product_ratio: float
    product_performance: tuple[ProductPerformance, ...]
    product_dependency_assessment: str
    main_gender: str
    main_grade: str
    high_value_grade: str
    repurchase_assessment: str
    can_measure_repurchase: bool
    peak_date: pd.Timestamp
    peak_sales: float
    low_date: pd.Timestamp
    low_sales: float
    volatility_assessment: str
    key_issues: tuple[str, ...]
    recommendations: tuple[ActionRecommendation, ...]
    confidence: str
    limitations: tuple[str, ...]

    def to_markdown(self) -> str:
        products = "\n".join(
            f"| {item.product} | {_money(item.sales)} | {item.quantity} | {item.contribution:.1%} |"
            for item in self.product_performance
        )
        issues = "\n".join(f"- {item}" for item in self.key_issues)
        actions = "\n\n".join(
            f"{index}. **建议：** {item.action}\n   - **原因：** {item.reason}\n"
            f"   - **建议观察：** {item.metrics_to_watch}"
            for index, item in enumerate(self.recommendations, 1)
        )
        limits = "\n".join(f"- {item}" for item in self.limitations)
        return f"""# Datalume AI 经营诊断报告

## 一、经营概览

- 总销售额：**{_money(self.total_sales)}**
- 订单数量：**{self.order_count}**
- 平均客单价：**{_money(self.avg_order_value)}**
- 用户数量：**{self.user_count}**
- 商品数量：**{self.product_count}**
- 日期范围：**{self.start_date:%Y-%m-%d} 至 {self.end_date:%Y-%m-%d}**

## 二、商品诊断

热销商品为 **{self.top_product}**，主力商品销售贡献率为 **{self.top_product_ratio:.1%}**。

| 商品 | 销售额 | 销量 | 贡献率 |
|---|---:|---:|---:|
{products}

{self.product_dependency_assessment}

## 三、用户诊断

- 主要性别人群：**{self.main_gender}性**
- 主要年级：**{self.main_grade}**
- 高价值年级：**{self.high_value_grade}**
- 复购判断：{self.repurchase_assessment}

## 四、销售趋势诊断

- 销售高峰：**{self.peak_date:%Y-%m-%d}**，{_money(self.peak_sales)}
- 销售低谷：**{self.low_date:%Y-%m-%d}**，{_money(self.low_sales)}
- 波动判断：{self.volatility_assessment}

## 五、核心经营问题

{issues}

## 六、行动建议

{actions}

## 七、报告可信度 / 数据限制

- 可信度：**{self.confidence}**
{limits}
"""


def generate_diagnostic_report(
    orders: pd.DataFrame,
    metrics: BusinessMetrics,
    overview: DataOverview,
    capabilities: DataCapabilities | None = None,
) -> DiagnosticReport:
    """只依据当前数据生成报告，不补造不可获得的指标。"""

    caps = capabilities or detect_data_capabilities(orders)
    product_quantity = orders.groupby("product")["quantity"].sum()
    performance = tuple(
        ProductPerformance(
            str(product),
            float(sales),
            int(product_quantity.get(product, 0)),
            float(sales / metrics.total_sales) if metrics.total_sales else 0.0,
        )
        for product, sales in metrics.product_sales.items()
    )
    ratio = metrics.top_product_ratio
    if ratio >= 0.6:
        dependency = f"{metrics.top_product} 贡献 {ratio:.1%}，存在较高的单一商品依赖风险。"
    elif ratio >= 0.4:
        dependency = f"{metrics.top_product} 贡献 {ratio:.1%}，主力商品特征明显，需持续观察集中度。"
    else:
        dependency = f"{metrics.top_product} 贡献 {ratio:.1%}，当前未观察到明显的单一商品依赖。"

    daily = metrics.daily_sales.astype(float)
    peak_date, low_date = pd.Timestamp(daily.idxmax()), pd.Timestamp(daily.idxmin())
    peak_sales, low_sales = float(daily.max()), float(daily.min())
    mean = float(daily.mean())
    coefficient = float(daily.std(ddof=0) / mean) if mean else 0.0
    if len(daily) < 7:
        volatility = f"仅有 {len(daily)} 个销售日期，样本不足，波动结论可信度有限。"
    elif coefficient >= 0.5:
        volatility = f"日销售额变异系数约 {coefficient:.1%}，波动较大。"
    elif coefficient >= 0.25:
        volatility = f"日销售额变异系数约 {coefficient:.1%}，存在一定波动。"
    else:
        volatility = f"日销售额变异系数约 {coefficient:.1%}，整体相对平稳。"

    if caps.repurchase_reliable and caps.order_id_field:
        per_user = orders.groupby("user_id")[caps.order_id_field].nunique()
        rate = float((per_user > 1).mean()) if len(per_user) else 0.0
        repurchase = f"可按订单号判断；购买超过一次的用户占比为 {rate:.1%}。"
    else:
        repurchase = "当前缺少完整订单号，无法准确区分订单粒度并判断正式复购率。"

    days = int((overview.end_date.normalize() - overview.start_date.normalize()).days) + 1
    issues: list[str] = []
    if ratio >= 0.4:
        issues.append(f"主力商品贡献 {ratio:.1%}，商品结构集中度值得关注。")
    if overview.row_count < 30:
        issues.append(f"当前只有 {overview.row_count} 条订单记录，样本较少。")
    if metrics.avg_order_value < float(orders["price"].median()) * 1.25:
        issues.append("客单价与单件商品价格接近，组合销售仍有验证空间。")
    if metrics.grade_sales.iloc[0] / metrics.total_sales >= 0.45:
        issues.append(f"{metrics.top_grade} 销售贡献较集中，用户结构存在集中风险。")
    if not caps.repurchase_reliable:
        issues.append("缺少可靠订单标识，无法准确衡量复购。")
    if not caps.profit_supported:
        issues.append("缺少成本或利润字段，无法确认销售增长是否带来利润增长。")
    issues = issues[:4] or ["当前未识别到明显异常，建议继续扩大样本并观察趋势。"]

    recs = [
        ActionRecommendation(
            f"围绕 {metrics.top_product} 设计组合套餐并做小范围测试。",
            f"该商品当前贡献总销售额 {ratio:.1%}，具备主力商品特征。",
            "套餐转化率、客单价、主力商品贡献率",
        ),
        ActionRecommendation(
            f"针对 {metrics.top_grade} 用户设计分层活动。",
            f"该年级当前销售贡献最高，为 {_money(float(metrics.grade_sales.iloc[0]))}。",
            "活动转化率、用户数、年级销售贡献",
        ),
        ActionRecommendation(
            "按周复盘订单量与客单价的变化。",
            "销售额变化需要拆解为订单量与客单价，避免只看总额。",
            "销售额、订单量、客单价、用户数",
        ),
    ]
    if caps.repurchase_reliable:
        recs.append(ActionRecommendation("对高频且最近活跃用户做复购触达。", "当前数据可识别订单频次与最近购买时间。", "复购用户率、购买频次、触达转化率"))
    else:
        recs.append(ActionRecommendation("后续数据增加唯一 order_id。", "没有订单标识时无法可靠区分一次订单中的多行明细。", "字段完整率、可识别订单数、复购用户率"))
    if not caps.profit_supported:
        recs.append(ActionRecommendation("补充单位成本或订单利润字段。", "销售额不能替代利润，当前无法判断商品盈利能力。", "毛利、毛利率、商品利润贡献"))

    limitations = []
    if overview.row_count < 30:
        limitations.append(f"样本只有 {overview.row_count} 条订单记录，细分结论易受单笔订单影响。")
    if days < 14:
        limitations.append(f"数据周期只有 {days} 天，无法稳定比较两个完整周周期。")
    limitations.extend(caps.limitations)
    confidence = "较低" if overview.row_count < 30 or days < 14 else "中等"
    if overview.row_count >= 200 and days >= 28 and caps.repurchase_reliable:
        confidence = "较高"

    return DiagnosticReport(
        metrics.total_sales, metrics.order_count, metrics.avg_order_value,
        overview.user_count, overview.product_count, overview.start_date,
        overview.end_date, metrics.top_product, ratio, performance, dependency,
        metrics.top_gender, str(metrics.grade_count.index[0]), metrics.top_grade,
        repurchase, caps.repurchase_reliable, peak_date, peak_sales, low_date,
        low_sales, volatility, tuple(issues), tuple(recs[:5]), confidence,
        tuple(dict.fromkeys(limitations)),
    )
