"""Datalume Agent Skill 的轻量发现、加载与确定性回退选择。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SKILLS_ROOT = PROJECT_ROOT / "skills"
VALID_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillLoadError(RuntimeError):
    """Skill metadata 或正文无法安全读取。"""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str

    def as_prompt_payload(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class LoadedSkill:
    metadata: SkillMetadata
    instructions: str
    path: Path


def _front_matter(path: Path, *, include_instructions: bool = False) -> tuple[list[str], str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise SkillLoadError(f"Skill 缺少 metadata：{path.name}")
            metadata_lines: list[str] = []
            for line in handle:
                if line.strip() == "---":
                    instructions = handle.read().strip() if include_instructions else ""
                    return metadata_lines, instructions
                metadata_lines.append(line.rstrip("\n"))
    except OSError as error:
        raise SkillLoadError(f"无法读取 Skill：{path.name}") from error
    raise SkillLoadError(f"Skill metadata 未闭合：{path.name}")


def _parse_metadata(lines: Iterable[str], path: Path) -> SkillMetadata:
    values: dict[str, str] = {}
    current_key: str | None = None
    folded: list[str] = []

    def flush() -> None:
        nonlocal current_key, folded
        if current_key is not None:
            values[current_key] = " ".join(part.strip() for part in folded if part.strip())
        current_key, folded = None, []

    for raw_line in lines:
        if raw_line.startswith((" ", "\t")) and current_key:
            folded.append(raw_line)
            continue
        flush()
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        key, value = key.strip(), raw_value.strip()
        if value in {">", "|"}:
            current_key = key
            continue
        values[key] = value.strip('"\'')
    flush()

    name = values.get("name", "").strip()
    description = values.get("description", "").strip()
    if not VALID_SKILL_NAME.fullmatch(name) or not description:
        raise SkillLoadError(f"Skill metadata 无效：{path.name}")
    return SkillMetadata(name, description)


class SkillLoader:
    """先暴露 metadata；只有选中后才读取并返回完整 instructions。"""

    def __init__(self, skills_root: str | Path = DEFAULT_SKILLS_ROOT) -> None:
        self.skills_root = Path(skills_root)

    def _skill_files(self) -> tuple[Path, ...]:
        if not self.skills_root.exists():
            return ()
        return tuple(sorted(self.skills_root.glob("*/SKILL.md")))

    def list_metadata(self) -> tuple[SkillMetadata, ...]:
        metadata = []
        seen: set[str] = set()
        for path in self._skill_files():
            lines, _ = _front_matter(path)
            item = _parse_metadata(lines, path)
            if item.name in seen:
                raise SkillLoadError(f"Skill 名称重复：{item.name}")
            seen.add(item.name)
            metadata.append(item)
        return tuple(metadata)

    def load(self, name: str) -> LoadedSkill:
        if not VALID_SKILL_NAME.fullmatch(name):
            raise SkillLoadError("Skill 名称无效。")
        path = self.skills_root / name / "SKILL.md"
        lines, instructions = _front_matter(path, include_instructions=True)
        metadata = _parse_metadata(lines, path)
        if metadata.name != name or not instructions:
            raise SkillLoadError(f"Skill 正文无效：{name}")
        return LoadedSkill(metadata, instructions, path)


def deterministic_skill_selection(
    question: str, available: Iterable[SkillMetadata]
) -> str | None:
    """仅在 LLM Skill selection 不可用时启用的保守组合规则。"""

    names = {item.name for item in available}
    if "business-diagnosis" not in names:
        return None
    normalized = re.sub(r"\s+", "", question.lower())
    diagnostic_signals = (
        "为什么", "什么原因", "诊断", "哪里出了问题", "出了什么问题",
        "经营情况怎么样", "业绩变差", "表现变差", "突然少", "异常",
    )
    change_signals = (
        "下降", "下滑", "减少", "变少", "变差", "变化", "波动",
        "经营问题", "问题", "业绩", "订单", "销售", "营收",
    )
    simple_query_signals = ("是多少", "最高", "排名", "排行", "画图", "画出来")
    diagnostic = any(signal in normalized for signal in diagnostic_signals)
    changing = any(signal in normalized for signal in change_signals)
    simple_only = any(signal in normalized for signal in simple_query_signals) and not diagnostic
    return "business-diagnosis" if diagnostic and changing and not simple_only else None
