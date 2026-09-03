"""Datalume 多数据库会话状态管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping
from uuid import uuid4

from .data_processor import fingerprint_csv


@dataclass
class DatabaseProject:
    database_id: str
    name: str
    csv_content: bytes
    data_source_id: str
    pinned: bool = False
    chat_messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_context: dict[str, Any] = field(default_factory=dict)
    agent_history: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_report: Any | None = None
    diagnostic_report_source_id: str | None = None


def initialize_database_state(state: MutableMapping[str, Any]) -> None:
    """首次进入时初始化真正的空数据库集合。"""

    if "databases" not in state:
        state["databases"] = {}
    if state.get("active_database_id") not in state["databases"]:
        state["active_database_id"] = None


def create_database(
    state: MutableMapping[str, Any],
    name: str,
    csv_content: bytes,
    *,
    pinned: bool = False,
) -> DatabaseProject:
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("数据库名称不能为空。")
    database_id = uuid4().hex
    project = DatabaseProject(
        database_id=database_id,
        name=cleaned_name,
        csv_content=csv_content,
        data_source_id=fingerprint_csv(csv_content),
        pinned=pinned,
    )
    state["databases"][database_id] = project
    state["active_database_id"] = database_id
    return project


def get_active_database(state: MutableMapping[str, Any]) -> DatabaseProject | None:
    database_id = state.get("active_database_id")
    return state["databases"].get(database_id)


def switch_database(state: MutableMapping[str, Any], database_id: str) -> None:
    if database_id not in state["databases"]:
        raise KeyError(f"数据库不存在：{database_id}")
    state["active_database_id"] = database_id


def rename_database(
    state: MutableMapping[str, Any], database_id: str, name: str
) -> DatabaseProject:
    """只修改显示名称；稳定的 database_id 和项目私有状态保持不变。"""

    if database_id not in state["databases"]:
        raise KeyError(f"数据库不存在：{database_id}")
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("数据库名称不能为空。")
    duplicate = any(
        project.database_id != database_id
        and project.name.strip().casefold() == cleaned_name.casefold()
        for project in state["databases"].values()
    )
    if duplicate:
        raise ValueError("已存在同名数据库。")
    project = state["databases"][database_id]
    project.name = cleaned_name
    return project


def delete_database(
    state: MutableMapping[str, Any], database_id: str
) -> DatabaseProject | None:
    """删除一个会话数据库，并在需要时按置顶优先选择新的 active。"""

    if database_id not in state["databases"]:
        raise KeyError(f"数据库不存在：{database_id}")
    was_active = state.get("active_database_id") == database_id
    project = state["databases"].pop(database_id)

    # 主动释放只属于被删除项目的会话数据，避免外部旧引用继续持有内容。
    project.csv_content = b""
    project.data_source_id = ""
    project.pinned = False
    project.chat_messages.clear()
    project.conversation_context.clear()
    project.agent_history.clear()
    project.diagnostic_report = None
    project.diagnostic_report_source_id = None

    if was_active:
        remaining = search_databases(state)
        state["active_database_id"] = (
            remaining[0].database_id if remaining else None
        )
    return get_active_database(state)


def toggle_database_pin(
    state: MutableMapping[str, Any], database_id: str
) -> bool:
    project = state["databases"][database_id]
    project.pinned = not project.pinned
    return project.pinned


def search_databases(
    state: MutableMapping[str, Any], query: str = ""
) -> list[DatabaseProject]:
    normalized = query.strip().lower()
    projects = list(state["databases"].values())
    if normalized:
        projects = [project for project in projects if normalized in project.name.lower()]
    return sorted(projects, key=lambda project: (not project.pinned, project.name))


def replace_database_data(project: DatabaseProject, csv_content: bytes) -> bool:
    """替换当前项目数据时清空只属于旧数据的对话和报告。"""

    new_source_id = fingerprint_csv(csv_content)
    if new_source_id == project.data_source_id:
        return False
    project.csv_content = csv_content
    project.data_source_id = new_source_id
    project.chat_messages.clear()
    project.conversation_context.clear()
    project.agent_history.clear()
    project.diagnostic_report = None
    project.diagnostic_report_source_id = None
    return True
