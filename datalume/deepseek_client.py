"""DeepSeek OpenAI-compatible 客户端：结构化理解与自然语言生成。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from .agent_models import AgentModelTurn, AgentToolCall
from .analysis_engine import AnalysisResult
from .data_processor import DataCapabilities

try:
    from openai import OpenAI
except ImportError:  # 依赖未安装时仍允许基础分析模式启动
    OpenAI = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT_SECONDS = 20.0
VALID_INTENTS = {"product", "user", "repurchase", "profit", "channel", "trend", "strategy"}
VALID_ACTIONS = {"analyze", "visualize", "explain", "compare", "recommend"}
VALID_METRICS = {
    "quantity", "revenue", "profit", "margin", "orders", "users",
    "repurchase_rate", "avg_order_value", "frequency", "recency", None,
}


class DeepSeekError(RuntimeError):
    """DeepSeek 不可用或返回无法使用的内容。"""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        raw_timeout = os.getenv("DEEPSEEK_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
        try:
            timeout = max(1.0, float(raw_timeout))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            model=os.getenv("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "").strip() or DEFAULT_BASE_URL,
            timeout_seconds=timeout,
        )


@dataclass(frozen=True)
class IntentPlan:
    intent: str
    metric: str | None = None
    action: str = "analyze"
    visualization: bool = False
    follow_up: bool = False
    time_range: Any | None = None
    entity: str | None = None
    reasoning_request: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IntentPlan":
        intent = str(payload.get("intent", "")).strip().lower()
        metric_value = payload.get("metric")
        metric = str(metric_value).strip().lower() if metric_value not in (None, "", "null") else None
        action = str(payload.get("action", "analyze")).strip().lower()
        if intent not in VALID_INTENTS:
            raise DeepSeekError("LLM 返回了不支持的 intent。")
        if metric not in VALID_METRICS:
            metric = None
        if action not in VALID_ACTIONS:
            action = "analyze"
        entity_value = payload.get("entity")
        entity = str(entity_value).strip() if entity_value not in (None, "", "null") else None
        return cls(
            intent=intent,
            metric=metric,
            action=action,
            visualization=bool(payload.get("visualization", False)),
            follow_up=bool(payload.get("follow_up", False)),
            time_range=payload.get("time_range"),
            entity=entity,
            reasoning_request=bool(payload.get("reasoning_request", False)),
        )


def _compact(value: Any, *, depth: int = 0) -> Any:
    """把分析结果收敛为可序列化的小对象，绝不发送完整 CSV。"""

    if depth > 4:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _compact(item, depth=depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact(item, depth=depth + 1) for item in list(value)[:20]]
    return str(value)


def analysis_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "supported": result.supported,
        "conclusion": result.conclusion,
        "evidence": list(result.evidence),
        "reasons": list(result.reasons),
        "recommendations": list(result.recommendations),
        "limitations": list(result.limitations),
        "metrics": _compact(result.metrics),
    }


def capability_payload(capabilities: DataCapabilities | None) -> dict[str, Any]:
    if capabilities is None:
        return {}
    return {
        "repurchase_reliable": capabilities.repurchase_reliable,
        "profit_supported": capabilities.profit_supported,
        "channel_supported": capabilities.channel_supported,
        "channel_cost_supported": capabilities.channel_cost_supported,
        "limitations": list(capabilities.limitations),
    }


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig | None = None, *, sdk_client: Any | None = None) -> None:
        self.config = config or DeepSeekConfig.from_env()
        if sdk_client is not None:
            self._client = sdk_client
        elif self.config.api_key and OpenAI is not None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=1,
            )
        else:
            self._client = None

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(DeepSeekConfig.from_env())

    @property
    def available(self) -> bool:
        return self._client is not None

    def _completion(
        self,
        messages: list[dict[str, Any]],
        *,
        json_output: bool = False,
        max_tokens: int = 700,
    ) -> str:
        if self._client is None:
            raise DeepSeekError("DeepSeek API Key 缺失或 OpenAI SDK 不可用。")
        started = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
            if json_output:
                kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content or not str(content).strip():
                raise DeepSeekError("DeepSeek 返回空内容。")
            return str(content).strip()
        except DeepSeekError:
            raise
        except Exception as error:
            raise DeepSeekError("DeepSeek 请求失败。") from error
        finally:
            LOGGER.info("deepseek_request latency_ms=%d", int((time.monotonic() - started) * 1000))

    def agent_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_tokens: int = 900,
    ) -> AgentModelTurn:
        """执行一轮 OpenAI-compatible Tool Calling，并规范化 tool_calls。"""

        if self._client is None:
            raise DeepSeekError("DeepSeek API Key 缺失或 OpenAI SDK 不可用。")
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=False,
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            message = response.choices[0].message
            content = str(message.content).strip() if message.content else None
            calls: list[AgentToolCall] = []
            serialized_calls: list[dict[str, Any]] = []
            for raw_call in list(getattr(message, "tool_calls", None) or []):
                call_id = str(getattr(raw_call, "id", "")).strip()
                function = getattr(raw_call, "function", None)
                name = str(getattr(function, "name", "")).strip()
                raw_arguments = getattr(function, "arguments", "{}")
                if not call_id or not name:
                    raise DeepSeekError("DeepSeek 返回了不完整的 tool call。")
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except (json.JSONDecodeError, TypeError) as error:
                    raise DeepSeekError("DeepSeek tool call 参数无法解析。") from error
                if not isinstance(arguments, Mapping):
                    raise DeepSeekError("DeepSeek tool call 参数不是 JSON 对象。")
                calls.append(AgentToolCall(call_id, name, dict(arguments)))
                serialized_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                })
            if not calls and not content:
                raise DeepSeekError("DeepSeek Agent 返回空内容。")
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if serialized_calls:
                assistant_message["tool_calls"] = serialized_calls
            return AgentModelTurn(content, tuple(calls), assistant_message)
        except DeepSeekError:
            raise
        except Exception as error:
            raise DeepSeekError("DeepSeek Tool Calling 请求失败。") from error
        finally:
            LOGGER.info(
                "deepseek_agent_request latency_ms=%d",
                int((time.monotonic() - started) * 1000),
            )

    def select_skill(
        self,
        question: str,
        *,
        skills: Sequence[Mapping[str, str]],
        conversation_context: Mapping[str, Any] | None,
        capabilities: DataCapabilities | None,
    ) -> str | None:
        """只使用 Skill metadata 选择专业方法，不加载或发送完整 SKILL.md。"""

        if not skills:
            return None
        allowed = {
            str(item.get("name", "")).strip()
            for item in skills if str(item.get("name", "")).strip()
        }
        system = """你是 Datalume Agent 的 Skill 选择器。只输出 JSON，不回答业务问题。
根据 user_goal 和 available_skills 的 name/description 判断是否需要一个 Skill。
只有复杂诊断、原因定位或异常调查才选专业 Skill；能由单个分析工具回答的指标查询、排名或画图应返回 null。
不得仅因为问题出现“销售”就选择 Skill。
输出格式：{"skill": "skill-name"} 或 {"skill": null}。"""
        payload = {
            "user_goal": question,
            "available_skills": list(skills),
            "conversation_context": _compact(conversation_context or {}),
            "data_capabilities": capability_payload(capabilities),
        }
        content = self._completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            json_output=True,
            max_tokens=180,
        )
        try:
            selected = json.loads(content)
        except json.JSONDecodeError as error:
            raise DeepSeekError("DeepSeek Skill selection JSON 无法解析。") from error
        if not isinstance(selected, Mapping):
            raise DeepSeekError("DeepSeek Skill selection JSON 不是对象。")
        name = selected.get("skill")
        if name in (None, "", "null"):
            return None
        normalized = str(name).strip()
        if normalized not in allowed:
            raise DeepSeekError("DeepSeek 选择了未知 Skill。")
        return normalized

    def parse_intent(
        self,
        question: str,
        *,
        conversation_context: Mapping[str, Any] | None,
        capabilities: DataCapabilities | None,
    ) -> IntentPlan:
        system = """你是 Datalume 的经营问题理解器。只输出 JSON 对象，不回答业务问题。
允许的 intent：product,user,repurchase,profit,channel,trend,strategy。
允许的 action：analyze,visualize,explain,compare,recommend。
metric 优先使用 quantity,revenue,profit,margin,orders,users,repurchase_rate,avg_order_value,frequency,recency 或 null。
结合 conversation_context 判断自然追问；当前问题优先。visualization 表示用户明确或隐含需要图表。
输出字段必须完整：intent,metric,action,visualization,follow_up,time_range,entity,reasoning_request。
示例 JSON：{"intent":"product","metric":"quantity","action":"visualize","visualization":true,"follow_up":true,"time_range":null,"entity":null,"reasoning_request":false}。"""
        user_payload = {
            "question": question,
            "conversation_context": _compact(conversation_context or {}),
            "data_capabilities": capability_payload(capabilities),
        }
        content = self._completion(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            json_output=True,
            max_tokens=500,
        )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise DeepSeekError("DeepSeek intent JSON 无法解析。") from error
        if not isinstance(payload, Mapping):
            raise DeepSeekError("DeepSeek intent JSON 不是对象。")
        return IntentPlan.from_mapping(payload)

    def generate_answer(
        self,
        question: str,
        *,
        plan: IntentPlan,
        result: AnalysisResult,
        conversation_context: Mapping[str, Any] | None,
        capabilities: DataCapabilities | None,
    ) -> str:
        system = """你是 Datalume 数据分析助手。用简洁、自然的中文直接回答。
所有核心数字只能引用 analysis_result；不得自行计算、猜测或补造原因。
简单问题用一两句话；用户要求原因或建议时才展开。数据不足时明确说明限制。
不要固定输出“结论/证据/原因/建议/限制”五段式，不要声称看到了未提供的完整 CSV。
如果 action=visualize，简短说明图表展示什么即可。"""
        user_payload = {
            "question": question,
            "intent_plan": _compact(plan.__dict__),
            "conversation_context": _compact(conversation_context or {}),
            "analysis_result": analysis_payload(result),
            "data_capabilities": capability_payload(capabilities),
        }
        return self._completion(
            [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            max_tokens=700,
        )
