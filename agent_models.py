"""Datalume Agent 的公开状态、工具调用与响应模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentModelTurn:
    """DeepSeek 单轮响应；只保留可公开内容和结构化 tool calls。"""

    content: str | None
    tool_calls: tuple[AgentToolCall, ...]
    assistant_message: Mapping[str, Any]


@dataclass(frozen=True)
class VisualizationSpec:
    analysis_type: str
    metric: str | None = None
    chart_type: str | None = None
    source_tool: str | None = None


@dataclass(frozen=True)
class AgentToolResult:
    tool_call_id: str
    tool_name: str
    supported: bool
    summary: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    analysis_result: Any | None = field(default=None, repr=False, compare=False)
    visualization: VisualizationSpec | None = field(default=None, repr=False, compare=False)
    report: Any | None = field(default=None, repr=False, compare=False)

    def observation(self) -> dict[str, Any]:
        """返回可发送给模型的紧凑结果，不含 DataFrame、Chart 或完整报告。"""

        return {
            "tool_name": self.tool_name,
            "supported": self.supported,
            "summary": self.summary,
            "metrics": dict(self.metrics),
            "evidence": list(self.evidence),
            "reasons": list(self.reasons),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
            "error": self.error,
        }


@dataclass(frozen=True)
class AgentStep:
    step_index: int
    tool_name: str
    tool_arguments: Mapping[str, Any]
    tool_result_summary: str
    supported: bool
    latency_ms: int
    error: str | None = None


@dataclass
class AgentState:
    user_goal: str
    current_database_id: str
    conversation_context: Mapping[str, Any]
    skills_used: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    tool_results: list[AgentToolResult] = field(default_factory=list)
    final_answer: str | None = None
    finished: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    steps: tuple[AgentStep, ...]
    visualizations: tuple[VisualizationSpec, ...]
    fallback_used: bool
    fallback_level: str
    stop_reason: str
    primary_result: Any
    plan: Any
    analysis_results: tuple[tuple[str, Any], ...] = ()
    report: Any | None = field(default=None, repr=False, compare=False)
    skills_used: tuple[str, ...] = ()

    def public_trace(self) -> dict[str, Any]:
        return {
            "tools": [step.tool_name for step in self.steps],
            "stop_reason": self.stop_reason,
            "fallback_level": self.fallback_level,
            "skills_used": list(self.skills_used),
        }
