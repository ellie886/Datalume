"""Datalume 单 Agent 多步 Tool Calling 运行时与三级回退。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping, MutableMapping

from dotenv import load_dotenv

from .agent_models import AgentResponse, AgentState, AgentStep, AgentToolResult
from .agent_tools import AgentToolContext, AgentToolRegistry
from .ai_agent import analyze_question
from .analysis_engine import AnalysisResult
from .data_processor import BusinessMetrics, DataCapabilities, DataOverview
from .deepseek_client import DeepSeekClient, DeepSeekError, IntentPlan, analysis_payload, capability_payload
from .router import QuestionType
from .skill_loader import (
    LoadedSkill,
    SkillLoader,
    SkillLoadError,
    deterministic_skill_selection,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_AGENT_STEPS = 4


class AgentRuntimeError(RuntimeError):
    """Agent 无法安全完成当前循环。"""


@dataclass(frozen=True)
class AgentConfig:
    enabled: bool = True
    max_steps: int = DEFAULT_MAX_AGENT_STEPS

    @classmethod
    def from_env(cls) -> "AgentConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        enabled = os.getenv("DATALUME_AGENT_ENABLED", "true").strip().lower() not in {
            "0", "false", "no", "off"
        }
        try:
            max_steps = int(os.getenv("DATALUME_AGENT_MAX_STEPS", str(DEFAULT_MAX_AGENT_STEPS)))
        except ValueError:
            max_steps = DEFAULT_MAX_AGENT_STEPS
        return cls(enabled=enabled, max_steps=min(8, max(1, max_steps)))


def _compact_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    allowed = (
        "previous_intent", "previous_metric", "previous_entity",
        "previous_time_range", "previous_analysis_summary",
        "previous_visualization_type", "last_user_goal",
        "last_tools_used", "last_entities", "last_metrics", "last_summary",
        "last_skills_used",
    )
    compact: dict[str, Any] = {}
    for key in allowed:
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            compact[key] = dict(list(value.items())[:20])
        elif isinstance(value, (list, tuple)):
            compact[key] = list(value[:20])
        else:
            compact[key] = value
    return compact


def _system_prompt(active_skill: LoadedSkill | None = None) -> str:
    base = """你是 Datalume Data Agent。你负责选择工具、读取 observation、决定是否继续，并最终用中文回答。
所有数字必须来自工具 observation，禁止自行计算、猜测、补造或声称读取了完整 CSV。
跨维度问题可以连续调用多个工具；下一步应根据用户目标和已有 observation 决定，不能使用固定 pipeline。
每个 analyze_* 工具一次调用就会返回该维度的全部可用核心指标；同一分析工具在一次任务中最多调用一次，不要为了更换 metric 重复调用。
当 observation 已覆盖用户明确要求的维度时，必须立即输出 final answer，不再探索用户未要求的利润、复购或其他方向。
一次任务通常只需要 1–3 个分析工具；不要在同一轮并行请求超过完成用户目标所必需的工具。
数据能力不足时保留限制，不得把销售额等同利润，也不得用通用建议代替缺失的渠道结论。
用户明确要求图表时，在对应分析工具之后调用 create_visualization；不得生成 Python 或 Altair 代码。
用户明确要求经营报告时调用 generate_report。信息足够后直接给 final answer，不再调用工具。
最终回答先直接回答，再给关键数据证据；多维分析按维度组织，必要时给建议与数据限制。
不要输出隐藏思维链，只返回公开结论。"""
    if active_skill is None:
        return base
    return (
        base
        + f"\n\n# 已加载 Agent Skill：{active_skill.metadata.name}\n"
        + "以下 Skill 是本轮任务的专业方法与约束；它不替代你根据 Observation 自主决定下一步。\n\n"
        + active_skill.instructions
    )


def _initial_messages(
    question: str,
    database_id: str,
    context: Mapping[str, Any] | None,
    capabilities: DataCapabilities,
    *,
    active_skill: LoadedSkill | None = None,
    available_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    payload = {
        "user_goal": question,
        "current_database_id": database_id,
        "conversation_context": _compact_context(context),
        "data_capabilities": capability_payload(capabilities),
        "available_fields": list(available_fields),
        "active_skill": active_skill.metadata.name if active_skill else None,
        "instruction": "请先选择必要的白名单工具；不要直接编造业务数字。",
    }
    return [
        {"role": "system", "content": _system_prompt(active_skill)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]


def _resolve_skill(
    question: str,
    *,
    llm_client: DeepSeekClient,
    conversation_context: Mapping[str, Any] | None,
    capabilities: DataCapabilities,
    loader: SkillLoader,
) -> LoadedSkill | None:
    """metadata-first 选择；选择或加载失败时安全回到原 Agent v1。"""

    try:
        metadata = loader.list_metadata()
    except SkillLoadError as error:
        LOGGER.warning("skill_metadata_unavailable reason=%s", type(error).__name__)
        return None
    if not metadata:
        return None
    selector = getattr(llm_client, "select_skill", None)
    try:
        if not callable(selector):
            raise AttributeError("skill selector unavailable")
        selected = selector(
            question,
            skills=[item.as_prompt_payload() for item in metadata],
            conversation_context=conversation_context,
            capabilities=capabilities,
        )
    except (DeepSeekError, AttributeError, TypeError, ValueError, RuntimeError):
        selected = deterministic_skill_selection(question, metadata)
        LOGGER.info("skill_selection_fallback selected=%s", selected or "none")
    if selected is None:
        return None
    try:
        return loader.load(selected)
    except SkillLoadError as error:
        LOGGER.warning("skill_load_failed skill=%s reason=%s", selected, type(error).__name__)
        return None


def _tool_message(result: AgentToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(result.observation(), ensure_ascii=False, default=str),
    }


def _best_effort_answer(results: list[AgentToolResult]) -> str:
    informative = [result for result in results if result.summary]
    if not informative:
        return "当前没有获得可用的数据分析结果，已停止本次 Agent 分析。"
    lines = [result.summary for result in informative[-4:]]
    limitations = []
    for result in informative:
        limitations.extend(result.limitations)
    answer = "\n\n".join(dict.fromkeys(lines))
    if limitations:
        answer += "\n\n数据限制：" + "；".join(dict.fromkeys(limitations))
    return answer


def _placeholder_result(results: list[AgentToolResult]) -> AnalysisResult:
    report_result = next((item for item in reversed(results) if item.report is not None), None)
    if report_result:
        return AnalysisResult(
            QuestionType.STRATEGY, True, report_result.summary,
            evidence=report_result.evidence, limitations=report_result.limitations,
            metrics=dict(report_result.metrics),
        )
    return AnalysisResult(
        QuestionType.STRATEGY, False, _best_effort_answer(results),
        limitations=tuple(
            dict.fromkeys(limit for result in results for limit in result.limitations)
        ),
    )


def _primary_analysis(results: list[AgentToolResult]) -> AnalysisResult:
    analyses = [item.analysis_result for item in results if item.analysis_result is not None]
    if analyses:
        return analyses[0]
    return _placeholder_result(results)


def _plan_for_result(result: AnalysisResult, results: list[AgentToolResult]) -> IntentPlan:
    source = next(
        (item for item in results if item.analysis_result is result),
        None,
    )
    metric = str(source.metadata.get("metric")) if source and source.metadata.get("metric") else None
    visualizations = [item.visualization for item in results if item.visualization is not None]
    return IntentPlan(
        intent=result.intent.value,
        metric=metric,
        action="visualize" if visualizations else "analyze",
        visualization=bool(visualizations),
        entity=str(source.metadata.get("entity")) if source and source.metadata.get("entity") else None,
    )


def _update_context(
    context: MutableMapping[str, Any] | None,
    question: str,
    response: AgentResponse,
) -> None:
    if context is None:
        return
    result = response.primary_result
    entities = []
    for _, analysis in response.analysis_results:
        for key in ("top_product", "top_profit_product", "best_channel"):
            if analysis.metrics.get(key):
                entities.append(str(analysis.metrics[key]))
    metrics = [
        str(step.tool_arguments["metric"])
        for step in response.steps
        if step.tool_arguments.get("metric")
    ]
    context.update({
        "previous_intent": result.intent.value,
        "previous_metric": response.plan.metric,
        "previous_entity": entities[0] if entities else None,
        "previous_time_range": None,
        "previous_analysis_result": result,
        "previous_analysis_summary": analysis_payload(result),
        "previous_visualization_type": response.visualizations[0].chart_type if response.visualizations else "none",
        "last_user_goal": question,
        "last_tools_used": [step.tool_name for step in response.steps],
        "last_entities": list(dict.fromkeys(entities))[:8],
        "last_metrics": list(dict.fromkeys(metrics))[:8],
        "last_summary": result.conclusion,
        "last_skills_used": list(response.skills_used),
    })


def _run_tool_loop(
    question: str,
    *,
    database_id: str,
    orders: Any,
    metrics: BusinessMetrics,
    capabilities: DataCapabilities,
    overview: DataOverview,
    llm_client: DeepSeekClient,
    conversation_context: MutableMapping[str, Any] | None,
    config: AgentConfig,
    registry: AgentToolRegistry,
    skill_loader: SkillLoader,
) -> AgentResponse:
    active_skill = _resolve_skill(
        question,
        llm_client=llm_client,
        conversation_context=conversation_context,
        capabilities=capabilities,
        loader=skill_loader,
    )
    state = AgentState(question, database_id, _compact_context(conversation_context))
    if active_skill is not None:
        state.skills_used.append(active_skill.metadata.name)
    tool_context = AgentToolContext(
        question, database_id, orders, metrics, capabilities, overview
    )
    messages = _initial_messages(
        question,
        database_id,
        conversation_context,
        capabilities,
        active_skill=active_skill,
        available_fields=tuple(str(column) for column in getattr(orders, "columns", ())),
    )
    started_total = time.monotonic()

    while len(state.steps) < config.max_steps:
        executed_tools = {result.tool_name for result in state.tool_results if result.error is None}
        available_tools = [
            schema for schema in registry.schemas()
            if schema["function"]["name"] not in executed_tools
        ]
        turn = llm_client.agent_completion(messages, tools=available_tools)
        messages.append(dict(turn.assistant_message))
        if turn.tool_calls:
            for call in turn.tool_calls:
                if len(state.steps) >= config.max_steps:
                    break
                started = time.monotonic()
                result = registry.execute(call, tool_context)
                latency = int((time.monotonic() - started) * 1000)
                state.tool_results.append(result)
                state.steps.append(AgentStep(
                    len(state.steps) + 1, call.name, dict(call.arguments),
                    result.summary, result.supported, latency, result.error,
                ))
                messages.append(_tool_message(result))
                LOGGER.info(
                    "agent_step database_id=%s step=%d tool=%s supported=%s latency_ms=%d",
                    database_id, len(state.steps), call.name, result.supported, latency,
                )
            continue

        if not state.tool_results:
            raise AgentRuntimeError("模型未调用任何数据工具就尝试直接回答。")
        state.finished = True
        state.stop_reason = "final_answer"
        supported = [result for result in state.tool_results if result.supported]
        state.final_answer = (turn.content or "").strip() if supported else _best_effort_answer(state.tool_results)
        if not state.final_answer:
            state.final_answer = _best_effort_answer(state.tool_results)
        break

    if not state.finished:
        state.finished = True
        state.stop_reason = "max_steps"
        state.final_answer = _best_effort_answer(state.tool_results)

    primary = _primary_analysis(state.tool_results)
    plan = _plan_for_result(primary, state.tool_results)
    visualizations = tuple(
        result.visualization for result in state.tool_results if result.visualization is not None
    )
    analysis_results = tuple(
        (str(result.metadata.get("analysis_type")), result.analysis_result)
        for result in state.tool_results
        if result.analysis_result is not None
    )
    report = next((item.report for item in reversed(state.tool_results) if item.report is not None), None)
    response = AgentResponse(
        answer=state.final_answer or _best_effort_answer(state.tool_results),
        steps=tuple(state.steps),
        visualizations=visualizations,
        fallback_used=False,
        fallback_level="agent",
        stop_reason=state.stop_reason or "unknown",
        primary_result=primary,
        plan=plan,
        analysis_results=analysis_results,
        report=report,
        skills_used=tuple(state.skills_used),
    )
    _update_context(conversation_context, question, response)
    LOGGER.info(
        "agent_complete database_id=%s steps=%d stop_reason=%s latency_ms=%d fallback_level=agent",
        database_id, len(state.steps), response.stop_reason,
        int((time.monotonic() - started_total) * 1000),
    )
    return response


def _legacy_response(
    question: str,
    *,
    orders: Any,
    metrics: BusinessMetrics,
    capabilities: DataCapabilities,
    llm_client: DeepSeekClient | None,
    conversation_context: MutableMapping[str, Any] | None,
    agent_failed: bool,
) -> AgentResponse:
    legacy = analyze_question(
        question, metrics, orders=orders, capabilities=capabilities,
        llm_client=llm_client, conversation_context=conversation_context,
    )
    level = "deterministic" if legacy.fallback_used else "single_step"
    return AgentResponse(
        answer=legacy.answer,
        steps=(),
        visualizations=(),
        fallback_used=agent_failed or legacy.fallback_used,
        fallback_level=level,
        stop_reason="agent_fallback" if agent_failed else "agent_disabled",
        primary_result=legacy.result,
        plan=legacy.plan,
        analysis_results=((legacy.result.intent.value, legacy.result),),
    )


def run_agent(
    question: str,
    *,
    database_id: str,
    orders: Any,
    metrics: BusinessMetrics,
    capabilities: DataCapabilities,
    overview: DataOverview,
    llm_client: DeepSeekClient | None,
    conversation_context: MutableMapping[str, Any] | None = None,
    config: AgentConfig | None = None,
    registry: AgentToolRegistry | None = None,
    skill_loader: SkillLoader | None = None,
) -> AgentResponse:
    """优先执行 Agent；失败时依次回退当前单步 LLM 与确定性分析。"""

    cleaned = question.strip()
    if not cleaned:
        raise ValueError("经营问题不能为空。")
    runtime_config = config or AgentConfig.from_env()
    if not runtime_config.enabled:
        return _legacy_response(
            cleaned, orders=orders, metrics=metrics, capabilities=capabilities,
            llm_client=llm_client, conversation_context=conversation_context,
            agent_failed=False,
        )
    if llm_client is None:
        return _legacy_response(
            cleaned, orders=orders, metrics=metrics, capabilities=capabilities,
            llm_client=None, conversation_context=conversation_context,
            agent_failed=True,
        )
    try:
        return _run_tool_loop(
            cleaned, database_id=database_id, orders=orders, metrics=metrics,
            capabilities=capabilities, overview=overview, llm_client=llm_client,
            conversation_context=conversation_context, config=runtime_config,
            registry=registry or AgentToolRegistry(),
            skill_loader=skill_loader or SkillLoader(),
        )
    except (AgentRuntimeError, DeepSeekError, AttributeError, TypeError, ValueError) as error:
        LOGGER.warning(
            "agent_fallback database_id=%s reason=%s", database_id, type(error).__name__
        )
        return _legacy_response(
            cleaned, orders=orders, metrics=metrics, capabilities=capabilities,
            llm_client=llm_client, conversation_context=conversation_context,
            agent_failed=True,
        )
