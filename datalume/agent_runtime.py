"""Datalume 单 Agent 多步 Tool Calling 运行时与三级回退。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, MutableMapping

from dotenv import load_dotenv

from .agent_models import (
    AgentRequestDecision,
    AgentResponse,
    AgentState,
    AgentStep,
    AgentToolResult,
)
from .agent_tools import AgentToolContext, AgentToolRegistry
from .ai_agent import analyze_question
from .analysis_engine import AnalysisResult
from .data_processor import BusinessMetrics, DataCapabilities, DataOverview
from .deepseek_client import DeepSeekClient, DeepSeekError, IntentPlan, analysis_payload, capability_payload
from .router import QuestionType, has_explicit_intent
from .skill_loader import (
    LoadedSkill,
    SkillLoader,
    SkillLoadError,
    deterministic_skill_selection,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_AGENT_STEPS = 4
CLARIFICATION_SKILL_NAME = "clarify-analysis-request"
METRICS_SKILL_NAME = "business-metrics"
MAX_CLARIFICATION_ATTEMPTS = 3
CLARIFICATION_PROMPTS = {
    "object": "你希望分析哪个对象？",
    "metric": "你希望使用哪个指标判断？",
    "time_range": "你希望分析哪个时间范围？",
    "comparison": "你希望与哪个对象或时间段比较？",
    "referent": "你提到的对象具体指什么？",
    "scope": "你希望分析哪一部分数据？",
    "other": "请补充一下你希望分析的对象或口径？",
}
_UNSAFE_CLARIFICATION_OPTION = re.compile(
    r"[\n\r。；]|[%￥¥$]|\d[\d,.]*\s*(?:元|件|单|人|次|%)|"
    r"(?:是|为|达到|下降|上涨|增长|减少|增加|贡献)"
)
_METRIC_CLARIFICATION_REPLY = re.compile(
    r"(?:就?按|就?用|看)?"
    r"(?:销售额|营收|销量|件数|利润额|利润|毛利率|毛利|订单量|用户数|客单价|复购率|频次)"
    r"(?:(?:来)?(?:看|算|比较|判断|衡量)|就好|即可|吧)?"
)


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
        "last_skills_used", "pending_clarification",
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
        metadata = tuple(
            item for item in loader.list_metadata()
            if item.name != CLARIFICATION_SKILL_NAME
        )
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


def _resolve_request(
    question: str,
    *,
    llm_client: DeepSeekClient,
    conversation_context: Mapping[str, Any] | None,
    capabilities: DataCapabilities,
    loader: SkillLoader,
    available_fields: tuple[str, ...],
    available_tools: tuple[Mapping[str, str], ...],
) -> tuple[AgentRequestDecision, LoadedSkill | None]:
    """优先执行智能澄清与 Skill 选择；旧 client 自动保持原选择流程。"""

    planner = getattr(llm_client, "plan_request", None)
    try:
        metadata = loader.list_metadata()
        clarification_skill = loader.load(CLARIFICATION_SKILL_NAME)
        business_metadata = tuple(
            item for item in metadata if item.name != CLARIFICATION_SKILL_NAME
        )
        metrics_instructions = ""
        if any(item.name == METRICS_SKILL_NAME for item in business_metadata):
            try:
                metrics_instructions = loader.load(METRICS_SKILL_NAME).instructions
            except SkillLoadError:
                LOGGER.warning("metrics_policy_unavailable")
        if not callable(planner):
            raise AttributeError("request planner unavailable")
        decision = planner(
            question,
            skills=[item.as_prompt_payload() for item in business_metadata],
            clarification_instructions=clarification_skill.instructions,
            conversation_context=conversation_context,
            capabilities=capabilities,
            available_fields=available_fields,
            available_tools=available_tools,
            metrics_instructions=metrics_instructions,
        )
        if not isinstance(decision, AgentRequestDecision):
            raise TypeError("invalid request decision")
        if decision.action not in {"continue", "clarify"}:
            raise ValueError("invalid request action")
        if decision.action == "clarify":
            return decision, None
        active_skill = loader.load(decision.skill) if decision.skill else None
        return decision, active_skill
    except (DeepSeekError, SkillLoadError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        LOGGER.info("request_planning_fallback reason=%s", type(error).__name__)
        pending = (
            conversation_context.get("pending_clarification")
            if conversation_context else None
        )
        metric_reply = _METRIC_CLARIFICATION_REPLY.fullmatch(
            re.sub(r"[\s。！？?!]", "", question)
        )
        use_pending = isinstance(pending, Mapping) and (
            not has_explicit_intent(question) or metric_reply is not None
        )
        fallback_decision = AgentRequestDecision(
            action="continue",
            resolved_question=question,
            uses_pending=use_pending,
        )
        skill_question = _resolved_question(
            question, fallback_decision, conversation_context
        )
        active_skill = _resolve_skill(
            skill_question,
            llm_client=llm_client,
            conversation_context=conversation_context,
            capabilities=capabilities,
            loader=loader,
        )
        return AgentRequestDecision(
            action="continue",
            resolved_question=question,
            skill=active_skill.metadata.name if active_skill else None,
            uses_pending=use_pending,
        ), active_skill


def _safe_clarification_options(options: tuple[str, ...]) -> tuple[str, ...]:
    """只允许短口径选项，避免模型绕过数据工具输出业务结论。"""

    safe: list[str] = []
    for item in options:
        option = str(item).strip()
        if not option or len(option) > 40 or _UNSAFE_CLARIFICATION_OPTION.search(option):
            continue
        safe.append(option)
    return tuple(dict.fromkeys(safe))[:3]


def _pending_details(context: Mapping[str, Any] | None) -> tuple[str | None, list[str], int]:
    pending = context.get("pending_clarification") if context else None
    if not isinstance(pending, Mapping):
        return None, [], 0
    original = str(pending.get("original_question") or "").strip() or None
    raw_supplements = pending.get("supplements") or []
    if not isinstance(raw_supplements, (list, tuple)):
        raw_supplements = []
    supplements = [
        str(item).strip() for item in raw_supplements
        if str(item).strip()
    ][:MAX_CLARIFICATION_ATTEMPTS]
    try:
        attempts = max(0, int(pending.get("attempts") or 0))
    except (TypeError, ValueError):
        attempts = 0
    return original, list(dict.fromkeys(supplements)), attempts


def _resolved_question(
    question: str,
    decision: AgentRequestDecision,
    conversation_context: Mapping[str, Any] | None,
) -> str:
    """只拼接用户原话；绝不采用模型改写后的分析目标。"""

    original, supplements, _ = _pending_details(conversation_context)
    if not original or not decision.uses_pending:
        return question
    additions = list(supplements)
    if question != original and question not in additions:
        additions.append(question)
    if not additions:
        return original
    return original + "\n用户补充：" + "；".join(additions[:MAX_CLARIFICATION_ATTEMPTS])


def _clarification_response(
    question: str,
    decision: AgentRequestDecision,
    conversation_context: MutableMapping[str, Any] | None,
) -> AgentResponse:
    prompt = CLARIFICATION_PROMPTS.get(
        decision.ambiguity_type, CLARIFICATION_PROMPTS["other"]
    )
    options = _safe_clarification_options(decision.options)
    clarification = prompt
    if options:
        clarification += "\n\n" + "\n".join(
            f"{index}. {option}" for index, option in enumerate(options, 1)
        )
    if conversation_context is not None:
        original_question, supplements, previous_attempts = _pending_details(
            conversation_context
        )
        if original_question is None:
            original_question = question
        elif question != original_question and question not in supplements:
            supplements.append(question)
        attempts = min(previous_attempts + 1, MAX_CLARIFICATION_ATTEMPTS)
        if previous_attempts >= MAX_CLARIFICATION_ATTEMPTS:
            clarification = (
                "当前信息仍不足，暂时无法安全执行分析。"
                "请重新完整描述要分析的对象、指标和时间范围。"
            )
            conversation_context.pop("pending_clarification", None)
            options = ()
        else:
            conversation_context["pending_clarification"] = {
                "original_question": original_question,
                "supplements": supplements[:MAX_CLARIFICATION_ATTEMPTS],
                "clarification_question": prompt,
                "options": list(options),
                "reason": decision.reason,
                "attempts": attempts,
            }
    return AgentResponse(
        answer=clarification,
        steps=(),
        visualizations=(),
        fallback_used=False,
        fallback_level="agent",
        stop_reason=(
            "clarification_limit"
            if conversation_context is not None
            and "pending_clarification" not in conversation_context
            and clarification.startswith("当前信息仍不足")
            else "needs_clarification"
        ),
        primary_result=None,
        plan=None,
        analysis_results=(),
        skills_used=(CLARIFICATION_SKILL_NAME,),
        needs_clarification=True,
        clarification_options=options,
    )


def _tool_summaries(registry: AgentToolRegistry) -> tuple[Mapping[str, str], ...]:
    """给请求规划器提供最小白名单能力摘要，不暴露执行实现。"""

    summaries: list[Mapping[str, str]] = []
    for schema in registry.schemas():
        function = schema.get("function", {})
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        summaries.append({
            "name": name,
            "description": str(function.get("description") or "").strip(),
        })
    return tuple(summaries)


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
    context.pop("pending_clarification", None)
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
    active_skill: LoadedSkill | None,
) -> AgentResponse:
    available_fields = tuple(str(column) for column in getattr(orders, "columns", ()))
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
        available_fields=available_fields,
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
    runtime_registry = registry or AgentToolRegistry()
    loader = skill_loader or SkillLoader()
    available_fields = tuple(str(column) for column in getattr(orders, "columns", ()))
    resolved_question = cleaned
    try:
        request_decision, active_skill = _resolve_request(
            cleaned,
            llm_client=llm_client,
            conversation_context=conversation_context,
            capabilities=capabilities,
            loader=loader,
            available_fields=available_fields,
            available_tools=_tool_summaries(runtime_registry),
        )
        if request_decision.action == "clarify":
            return _clarification_response(
                cleaned, request_decision, conversation_context
            )
        resolved_question = _resolved_question(
            cleaned, request_decision, conversation_context
        )
        if (
            conversation_context is not None
            and not request_decision.uses_pending
        ):
            conversation_context.pop("pending_clarification", None)
        return _run_tool_loop(
            resolved_question, database_id=database_id, orders=orders, metrics=metrics,
            capabilities=capabilities, overview=overview, llm_client=llm_client,
            conversation_context=conversation_context, config=runtime_config,
            registry=runtime_registry,
            active_skill=active_skill,
        )
    except (AgentRuntimeError, DeepSeekError, AttributeError, TypeError, ValueError) as error:
        LOGGER.warning(
            "agent_fallback database_id=%s reason=%s", database_id, type(error).__name__
        )
        if conversation_context is not None:
            conversation_context.pop("pending_clarification", None)
        return _legacy_response(
            resolved_question, orders=orders, metrics=metrics, capabilities=capabilities,
            llm_client=llm_client, conversation_context=conversation_context,
            agent_failed=True,
        )
