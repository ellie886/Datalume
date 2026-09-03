"""Datalume AI 编排：LLM 理解 → Python 计算 → LLM 自然回答。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from .analysis_engine import AnalysisResult, run_analysis
from .data_processor import BusinessMetrics, DataCapabilities, detect_data_capabilities
from .deepseek_client import DeepSeekClient, DeepSeekError, IntentPlan, analysis_payload
from .router import QuestionType, classify_question, get_question_type_label, has_explicit_intent


@dataclass(frozen=True)
class AIAnalysis:
    question_type: QuestionType
    category_label: str
    answer: str
    result: AnalysisResult
    plan: IntentPlan
    fallback_used: bool = False


ChatHistory = Sequence[Mapping[str, str]]
DEICTIC = ("它", "这个", "那个", "上述", "刚才", "前面", "那应该", "那销售额", "再画", "画一下")
BARE = ("为什么", "原因呢", "怎么做", "具体呢", "然后呢", "怎么办", "帮我画一下", "画一下")
VISUAL_WORDS = ("画", "图", "图表", "可视化", "曲线", "柱状")


def _follow_up(question: str) -> bool:
    normalized = question.strip().lower().rstrip("？?!！。 ")
    return normalized in BARE or any(marker in normalized for marker in DEICTIC)


def _previous_question(history: ChatHistory) -> str | None:
    for message in reversed(history[-8:]):
        content = str(message.get("content", "")).strip()
        if message.get("role") == "user" and content and not _follow_up(content):
            return content
    return None


def _context_from_history(history: ChatHistory | None) -> dict[str, Any]:
    previous = _previous_question(history or ())
    if not previous:
        return {}
    intent = classify_question(previous).value
    return {
        "previous_intent": intent,
        "previous_metric": _infer_metric(previous, {"product": "revenue", "trend": "revenue"}.get(intent)),
    }


def _infer_metric(question: str, previous: str | None = None) -> str | None:
    normalized = question.lower()
    rules = (
        ("毛利率", "margin"), ("利润", "profit"), ("销量", "quantity"),
        ("件数", "quantity"), ("销售额", "revenue"), ("营收", "revenue"),
        ("客单价", "avg_order_value"), ("订单", "orders"),
        ("用户数", "users"), ("复购", "repurchase_rate"),
    )
    for keyword, metric in rules:
        if keyword in normalized:
            return metric
    return previous


def _fallback_plan(question: str, conversation_context: Mapping[str, Any] | None) -> IntentPlan:
    context = conversation_context or {}
    previous_intent = context.get("previous_intent")
    generic_visual = any(word in question for word in VISUAL_WORDS) and not has_explicit_intent(question)
    follow_up = bool(previous_intent and (_follow_up(question) or generic_visual or not has_explicit_intent(question)))
    intent = str(previous_intent) if follow_up else classify_question(question).value
    previous_metric = str(context.get("previous_metric")) if context.get("previous_metric") else None
    metric = _infer_metric(question, previous_metric if follow_up else None)
    if metric is None:
        metric = {"product": "revenue", "trend": "revenue", "repurchase": "repurchase_rate"}.get(intent)
    if any(word in question for word in VISUAL_WORDS):
        action = "visualize"
    elif any(word in question for word in ("为什么", "原因")):
        action = "explain"
    elif any(word in question for word in ("建议", "怎么", "推广", "提升")):
        action = "recommend"
    elif any(word in question for word in ("对比", "比较", "是否")):
        action = "compare"
    else:
        action = "analyze"
    return IntentPlan(
        intent=intent,
        metric=metric,
        action=action,
        visualization=action in {"visualize", "compare"} or intent != "strategy",
        follow_up=follow_up,
        time_range=context.get("previous_time_range") if follow_up else None,
        entity=str(context.get("previous_entity")) if follow_up and context.get("previous_entity") else None,
        reasoning_request=action == "explain",
    )


def _merge_follow_up_context(plan: IntentPlan, context: Mapping[str, Any] | None) -> IntentPlan:
    if not plan.follow_up or not context:
        return plan
    return IntentPlan(
        intent=plan.intent or str(context.get("previous_intent", "strategy")),
        metric=plan.metric or context.get("previous_metric"),
        action=plan.action,
        visualization=plan.visualization,
        follow_up=True,
        time_range=plan.time_range if plan.time_range is not None else context.get("previous_time_range"),
        entity=plan.entity or context.get("previous_entity"),
        reasoning_request=plan.reasoning_request,
    )


def _analysis_question(question: str, plan: IntentPlan) -> str:
    """把结构化 metric 转为现有分析函数可稳定理解的最小查询。"""

    intent_metric = {
        ("product", "quantity"): "商品销量",
        ("product", "revenue"): "商品销售额",
        ("profit", "profit"): "商品利润",
        ("profit", "margin"): "商品毛利率",
        ("channel", "revenue"): "渠道销售额",
        ("repurchase", "repurchase_rate"): "用户复购率",
        ("trend", "revenue"): "最近一周销售额趋势",
    }.get((plan.intent, plan.metric))
    if intent_metric is None:
        return question
    suffix = {
        "explain": " 为什么",
        "recommend": " 推广",
    }.get(plan.action, "")
    return intent_metric + suffix


def _context_for_llm(context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    return {key: value for key, value in context.items() if key != "previous_analysis_result"}


def _derive_entity(plan: IntentPlan, result: AnalysisResult) -> str | None:
    if plan.entity:
        return plan.entity
    for key in ("top_product", "top_profit_product", "best_channel"):
        if result.metrics.get(key):
            return str(result.metrics[key])
    top_users = result.metrics.get("top_users")
    if top_users:
        return str(top_users[0][0])
    return None


def _update_conversation_context(
    context: MutableMapping[str, Any] | None,
    plan: IntentPlan,
    result: AnalysisResult,
) -> None:
    if context is None:
        return
    context.update(
        {
            "previous_intent": plan.intent,
            "previous_metric": plan.metric,
            "previous_entity": _derive_entity(plan, result),
            "previous_time_range": plan.time_range,
            "previous_analysis_result": result,
            "previous_analysis_summary": analysis_payload(result),
            "previous_visualization_type": {
                "trend": "line", "product": "bar", "profit": "profit",
                "channel": "channel", "repurchase": "repurchase", "user": "user",
                "strategy": "none",
            }.get(plan.intent, "none"),
        }
    )


def analyze_question(
    question: str,
    metrics: BusinessMetrics,
    history: ChatHistory | None = None,
    orders: pd.DataFrame | None = None,
    capabilities: DataCapabilities | None = None,
    *,
    llm_client: DeepSeekClient | None = None,
    conversation_context: MutableMapping[str, Any] | None = None,
) -> AIAnalysis:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("经营问题不能为空。")
    caps = capabilities or (detect_data_capabilities(orders) if orders is not None else None)
    working_context: MutableMapping[str, Any] = conversation_context if conversation_context is not None else _context_from_history(history)
    llm_context = _context_for_llm(working_context)
    fallback_used = False

    if llm_client is not None:
        try:
            plan = llm_client.parse_intent(
                cleaned,
                conversation_context=llm_context,
                capabilities=caps,
            )
            plan = _merge_follow_up_context(plan, working_context)
        except DeepSeekError:
            plan = _fallback_plan(cleaned, working_context)
            fallback_used = True
    else:
        plan = _fallback_plan(cleaned, working_context)

    intent = QuestionType(plan.intent)
    result = run_analysis(intent, _analysis_question(cleaned, plan), orders, metrics, caps)
    answer = result.to_markdown()
    if llm_client is not None and not fallback_used:
        try:
            answer = llm_client.generate_answer(
                cleaned,
                plan=plan,
                result=result,
                conversation_context=llm_context,
                capabilities=caps,
            )
        except DeepSeekError:
            fallback_used = True

    _update_conversation_context(conversation_context, plan, result)
    return AIAnalysis(intent, get_question_type_label(intent), answer, result, plan, fallback_used)


def generate_ai_answer(
    question: str,
    metrics: BusinessMetrics,
    history: ChatHistory | None = None,
    orders: pd.DataFrame | None = None,
    capabilities: DataCapabilities | None = None,
) -> str:
    return analyze_question(question, metrics, history, orders, capabilities).answer


def generate_operation_summary(
    *, total_sales: float, order_count: int, top_product: str,
    top_gender: str, top_grade: str, avg_order_value: float | None = None,
    top_product_ratio: float | None = None,
) -> str:
    average = avg_order_value if avg_order_value is not None else total_sales / order_count if order_count else 0
    ratio = f"\n- 主力商品贡献率：**{top_product_ratio:.1%}**" if top_product_ratio is not None else ""
    return f"""### 综合运营分析

- 销售额：**¥{total_sales:,.2f}**
- 订单量：**{order_count} 单**
- 平均客单价：**¥{average:,.2f}**
- 主力商品：**{top_product}**
- 核心用户：**{top_gender}性用户**
- 高价值年级：**{top_grade}**{ratio}

### 运营建议

1. 保持 {top_product} 库存稳定并验证组合销售。
2. 针对 {top_grade} 用户设计可衡量的活动。
3. 持续复盘销售额、订单、客单价与当前数据支持的扩展指标。"""
