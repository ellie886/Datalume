"""Datalume Agent 白名单工具注册表。所有业务数字仍由现有分析引擎计算。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

import pandas as pd

from .agent_models import AgentToolCall, AgentToolResult, VisualizationSpec
from .analysis_engine import AnalysisResult, run_analysis
from .data_processor import BusinessMetrics, DataCapabilities, DataOverview
from .report_generator import generate_diagnostic_report
from .router import QuestionType
from .visualization import build_visualization, visualization_kind


ANALYSIS_TO_INTENT = {
    "analyze_product": QuestionType.PRODUCT,
    "analyze_user": QuestionType.USER,
    "analyze_repurchase": QuestionType.REPURCHASE,
    "analyze_profit": QuestionType.PROFIT,
    "analyze_channel": QuestionType.CHANNEL,
    "analyze_trend": QuestionType.TREND,
    "analyze_strategy": QuestionType.STRATEGY,
}

TOOL_LABELS = {
    "analyze_product": "商品表现",
    "analyze_user": "用户结构",
    "analyze_repurchase": "复购表现",
    "analyze_profit": "利润表现",
    "analyze_channel": "渠道表现",
    "analyze_trend": "销售趋势",
    "analyze_strategy": "经营策略",
    "create_visualization": "数据可视化",
    "generate_report": "经营诊断报告",
}


def _compact(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _compact(value.item(), depth=depth)
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_compact(item, depth=depth + 1) for item in list(value)[:20]]
    return str(value)


@dataclass
class AgentToolContext:
    user_goal: str
    database_id: str
    orders: pd.DataFrame
    metrics: BusinessMetrics
    capabilities: DataCapabilities
    overview: DataOverview
    analysis_results: dict[str, AnalysisResult] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]
    executor: Callable[[AgentToolCall, AgentToolContext], AgentToolResult]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def _object_schema(properties: Mapping[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _enum(values: tuple[str, ...], description: str) -> dict[str, Any]:
    return {"type": ["string", "null"], "enum": [*values, None], "description": description}


def _validate_arguments(call: AgentToolCall, schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(call.arguments, Mapping):
        raise ValueError("工具参数必须是 JSON 对象。")
    arguments = dict(call.arguments)
    properties = schema.get("properties", {})
    unexpected = sorted(set(arguments) - set(properties))
    if unexpected:
        raise ValueError(f"不支持的参数：{'、'.join(unexpected)}")
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        raise ValueError(f"缺少参数：{'、'.join(missing)}")
    for name, value in arguments.items():
        definition = properties[name]
        enum_values = definition.get("enum")
        if enum_values is not None and value not in enum_values:
            raise ValueError(f"参数 {name} 不在允许范围内。")
        if name == "top_n":
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10:
                raise ValueError("top_n 必须是 1 到 10 的整数。")
    return arguments


def _analysis_question(tool_name: str, arguments: Mapping[str, Any], user_goal: str) -> str:
    metric = arguments.get("metric")
    base = {
        "analyze_product": "商品销量" if metric == "quantity" else "商品销售额",
        "analyze_user": "用户画像和高价值用户",
        "analyze_repurchase": "用户复购率与 RFM",
        "analyze_profit": "商品毛利率" if metric == "margin" else "商品利润",
        "analyze_channel": "渠道表现与获客效果",
        "analyze_trend": "最近一周销售额趋势",
        "analyze_strategy": "经营策略建议",
    }[tool_name]
    entity = arguments.get("entity") or arguments.get("channel")
    return f"{base} {entity or ''}；用户目标：{user_goal}".strip()


def _analysis_executor(call: AgentToolCall, context: AgentToolContext) -> AgentToolResult:
    definition = TOOL_DEFINITIONS[call.name]
    arguments = _validate_arguments(call, definition.parameters)
    intent = ANALYSIS_TO_INTENT[call.name]
    result = run_analysis(
        intent,
        _analysis_question(call.name, arguments, context.user_goal),
        context.orders,
        context.metrics,
        context.capabilities,
    )
    context.analysis_results[intent.value] = result
    return AgentToolResult(
        call.call_id,
        call.name,
        result.supported,
        result.conclusion,
        metrics=_compact(result.metrics),
        evidence=tuple(result.evidence[:5]),
        reasons=tuple(result.reasons[:4]),
        recommendations=tuple(result.recommendations[:4]),
        limitations=tuple(result.limitations[:5]),
        metadata={
            "analysis_type": intent.value,
            "metric": arguments.get("metric"),
            "entity": arguments.get("entity") or arguments.get("channel"),
        },
        analysis_result=result,
    )


def _visualization_executor(call: AgentToolCall, context: AgentToolContext) -> AgentToolResult:
    definition = TOOL_DEFINITIONS[call.name]
    arguments = _validate_arguments(call, definition.parameters)
    analysis_type = str(arguments["analysis_type"])
    result = context.analysis_results.get(analysis_type)
    if result is None:
        return AgentToolResult(
            call.call_id, call.name, False,
            f"尚未执行 {analysis_type} 分析，无法生成对应图表。",
            limitations=("必须先调用对应分析工具，再创建可视化。",),
            error="missing_analysis_result",
        )
    intent = QuestionType(analysis_type)
    metric = arguments.get("metric")
    chart = build_visualization(
        intent, context.orders, result, context.capabilities,
        metric=metric, action="visualize", visualization=True,
    )
    if chart is None:
        return AgentToolResult(
            call.call_id, call.name, False, "当前分析结果无法生成有效图表。",
            limitations=tuple(result.limitations), error="visualization_unavailable",
        )
    spec = VisualizationSpec(
        analysis_type=analysis_type,
        metric=str(metric) if metric else None,
        chart_type=str(arguments.get("chart_type") or visualization_kind(intent)),
        source_tool=f"analyze_{analysis_type}",
    )
    return AgentToolResult(
        call.call_id, call.name, True,
        f"已准备{TOOL_LABELS.get(spec.source_tool or '', analysis_type)}图表。",
        metadata={
            "analysis_type": analysis_type,
            "metric": spec.metric,
            "chart_type": spec.chart_type,
        },
        visualization=spec,
    )


def _report_executor(call: AgentToolCall, context: AgentToolContext) -> AgentToolResult:
    definition = TOOL_DEFINITIONS[call.name]
    _validate_arguments(call, definition.parameters)
    report = generate_diagnostic_report(
        context.orders, context.metrics, context.overview, context.capabilities
    )
    return AgentToolResult(
        call.call_id, call.name, True, "已基于当前数据库生成经营诊断报告。",
        metrics={
            "total_sales": report.total_sales,
            "order_count": report.order_count,
            "avg_order_value": report.avg_order_value,
            "user_count": report.user_count,
            "product_count": report.product_count,
            "confidence": report.confidence,
        },
        evidence=(
            f"主力商品：{report.top_product}，贡献率 {report.top_product_ratio:.1%}。",
            f"销售高峰：{report.peak_date:%Y-%m-%d}。",
        ),
        limitations=tuple(report.limitations[:5]),
        metadata={"report_generated": True, "confidence": report.confidence},
        report=report,
    )


PRODUCT_SCHEMA = _object_schema({
    "metric": _enum(("quantity", "revenue"), "商品排名指标"),
    "entity": {"type": ["string", "null"], "description": "可选商品名"},
    "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
})
USER_SCHEMA = _object_schema({
    "metric": _enum(("users", "revenue", "frequency", "recency"), "用户分析指标"),
    "entity": {"type": ["string", "null"], "description": "可选用户或人群"},
    "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
})
REPURCHASE_SCHEMA = _object_schema({
    "metric": _enum(("repurchase_rate", "frequency", "recency"), "复购指标"),
    "entity": {"type": ["string", "null"], "description": "可选用户或人群"},
})
PROFIT_SCHEMA = _object_schema({
    "metric": _enum(("profit", "margin"), "利润指标"),
    "entity": {"type": ["string", "null"], "description": "可选商品名"},
})
CHANNEL_SCHEMA = _object_schema({
    "metric": _enum(("revenue", "orders", "users", "avg_order_value", "repurchase_rate"), "渠道指标"),
    "channel": {"type": ["string", "null"], "description": "可选渠道名"},
})
TREND_SCHEMA = _object_schema({
    "metric": _enum(("revenue", "orders", "avg_order_value", "users"), "趋势指标"),
    "time_range": {"type": ["string", "null"], "description": "例如 recent_week"},
})
STRATEGY_SCHEMA = _object_schema({
    "metric": _enum(("revenue", "orders", "avg_order_value", "users"), "策略关注指标"),
})
VISUALIZATION_SCHEMA = _object_schema({
    "analysis_type": {"type": "string", "enum": ["product", "user", "repurchase", "profit", "channel", "trend"]},
    "metric": _enum(("quantity", "revenue", "profit", "margin", "orders", "users", "repurchase_rate", "avg_order_value"), "图表指标"),
    "chart_type": _enum(("line", "bar", "profit", "channel", "repurchase", "user"), "图表类型"),
}, required=("analysis_type",))
REPORT_SCHEMA = _object_schema({})


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "analyze_product": ToolDefinition("analyze_product", "计算商品销量、销售额、贡献率和排名。", PRODUCT_SCHEMA, _analysis_executor),
    "analyze_user": ToolDefinition("analyze_user", "计算用户结构、消费贡献和高价值用户。", USER_SCHEMA, _analysis_executor),
    "analyze_repurchase": ToolDefinition("analyze_repurchase", "计算购买次数、复购用户、人群复购差异和基础 RFM。", REPURCHASE_SCHEMA, _analysis_executor),
    "analyze_profit": ToolDefinition("analyze_profit", "在存在成本或利润字段时计算商品利润和毛利率。", PROFIT_SCHEMA, _analysis_executor),
    "analyze_channel": ToolDefinition("analyze_channel", "在存在渠道字段时计算渠道用户、订单、销售额、客单价和复购表现。", CHANNEL_SCHEMA, _analysis_executor),
    "analyze_trend": ToolDefinition("analyze_trend", "比较最近周期与上一可比周期并拆解销售额、订单量和客单价变化。", TREND_SCHEMA, _analysis_executor),
    "analyze_strategy": ToolDefinition("analyze_strategy", "基于当前已计算经营指标给出策略建议。", STRATEGY_SCHEMA, _analysis_executor),
    "create_visualization": ToolDefinition("create_visualization", "为已完成的分析选择现有安全图表；不会执行模型生成代码。", VISUALIZATION_SCHEMA, _visualization_executor),
    "generate_report": ToolDefinition("generate_report", "仅在用户明确要求时生成确定性的经营诊断报告。", REPORT_SCHEMA, _report_executor),
}


class AgentToolRegistry:
    def __init__(self, definitions: Mapping[str, ToolDefinition] | None = None) -> None:
        self._definitions = dict(definitions or TOOL_DEFINITIONS)

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.openai_schema() for definition in self._definitions.values()]

    def execute(self, call: AgentToolCall, context: AgentToolContext) -> AgentToolResult:
        definition = self._definitions.get(call.name)
        if definition is None:
            return AgentToolResult(
                call.call_id, call.name, False,
                f"工具 {call.name} 不在 Datalume 白名单中，未执行。",
                error="unknown_tool",
                limitations=("Agent 只能调用预定义的数据分析工具。",),
            )
        try:
            return definition.executor(call, context)
        except (KeyError, TypeError, ValueError) as error:
            return AgentToolResult(
                call.call_id, call.name, False, "工具参数校验失败，未执行。",
                error=str(error), limitations=(str(error),),
            )
        except Exception:
            return AgentToolResult(
                call.call_id, call.name, False, "工具执行失败，已安全停止该方向。",
                error="tool_execution_failed",
            )
