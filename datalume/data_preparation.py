"""Inspect and prepare uploaded tables without guessing missing business facts.

The caller chooses a worksheet, header row and column mappings. All row numbers
shown to users are one-based physical file/sheet row numbers. This module does
not contact a model or persist the original upload.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Mapping

import pandas as pd

from .data_processor import DataValidationError, OPTIONAL_FIELD_ALIASES, REQUIRED_COLUMNS, _excel_workbook

# Deliberately exact aliases only: similar names can have different meanings.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "日期", "交易日期", "订单日期", "下单日期", "销售日期"),
    "revenue": ("sales", "revenue", "销售额", "营业额", "销售金额", "营业收入"),
    "channel": ("channel", "source", "渠道", "销售渠道", "来源"),
    "user_id": ("user_id", "用户id", "用户编号", "客户id", "客户编号"),
    "product": ("product", "商品", "商品名称", "产品", "产品名称"),
    "price": ("price", "单价", "商品单价"),
    "quantity": ("quantity", "数量", "销量", "销售数量"),
    "gender": ("gender", "性别"),
    "grade": ("grade", "年级"),
    "order_id": ("order_id", "订单id", "订单编号", "订单号"),
    "unit_cost": ("unit_cost", "单位成本", "成本单价"),
    "total_cost": ("total_cost", "总成本"),
    "profit": ("profit", "利润", "毛利"),
}
# Carry forward every optional alias recognised by the existing order engine;
# auto-selection is still suppressed when more than one matching column exists.
for _field, _aliases in OPTIONAL_FIELD_ALIASES.items():
    FIELD_ALIASES[_field] = tuple(dict.fromkeys((*FIELD_ALIASES.get(_field, ()), *_aliases)))
ORDER_FIELDS = ("user_id", "product", "price", "quantity", "date", "gender", "grade")
NUMERIC_FIELDS = frozenset({"revenue", "price", "quantity"})
_NUMBER = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_AMBIGUOUS_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:\s.*)?$")
_TIMEZONE_DATE = re.compile(r"(?:[zZ]|[+-]\d\d:?\d\d)$")
# Downstream legacy code recognises these names automatically. Do not leave an
# unselected original column under one of those names: doing so would turn an
# unconfirmed field into an apparently confirmed business capability.
_RESERVED_NAMES = {
    name.casefold()
    for name in (
        *FIELD_ALIASES, *REQUIRED_COLUMNS, "sales", *OPTIONAL_FIELD_ALIASES,
        *(alias for aliases in FIELD_ALIASES.values() for alias in aliases),
        *(alias for aliases in OPTIONAL_FIELD_ALIASES.values() for alias in aliases),
    )
}


@dataclass(frozen=True)
class TablePreview:
    columns: tuple[str, ...]
    frame: pd.DataFrame  # integer column positions, original values as strings
    header_row: int
    filename: str
    sheet_name: str | None = None
    row_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class QualityPreview:
    original_rows: int
    empty_rows: tuple[int, ...]
    duplicate_rows: tuple[int, ...]
    invalid_numeric: dict[str, tuple[int, ...]]
    invalid_dates: tuple[int, ...]
    ambiguous_dates: tuple[int, ...]


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    profile: str  # 'orders' or 'revenue'
    csv_content: bytes
    quality: QualityPreview


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def read_table(content: bytes, *, filename: str, sheet_name: str | None = None,
               header_row: int = 0) -> TablePreview:
    """Read one sheet/CSV with a user-selectable zero-based header row.

    Excel numeric cells are read as stored; display formatting such as leading
    zero masks is not part of the cell value and cannot be reconstructed.
    """
    if not isinstance(header_row, int) or isinstance(header_row, bool) or header_row < 0:
        raise DataValidationError("表头行号无效。")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            text = content.decode("utf-8-sig")
            reader = csv.reader(StringIO(text, newline=""), strict=True)
            rows = []
            row_numbers = []
            while True:
                start_line = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                rows.append(row)
                row_numbers.append(start_line)
        except UnicodeDecodeError as error:
            raise DataValidationError("CSV 编码无法识别，请保存为 UTF-8 后重试。") from error
        except csv.Error as error:
            raise DataValidationError(f"CSV 格式无法解析：{error}") from error
        width = max((len(row) for row in rows), default=0)
        raw = pd.DataFrame([row + [""] * (width - len(row)) for row in rows])
        selected = None
    elif suffix in {".xls", ".xlsx"}:
        with _excel_workbook(content, filename=filename) as workbook:
            selected = sheet_name if sheet_name is not None else (workbook.sheet_names[0] if workbook.sheet_names else None)
            if selected not in workbook.sheet_names:
                raise DataValidationError("Excel 中找不到选定工作表。")
            try:
                raw = workbook.parse(sheet_name=selected, header=None, dtype=object, keep_default_na=False)
                row_numbers = list(range(1, len(raw) + 1))
            except Exception as error:
                raise DataValidationError("Excel 工作表无法读取，请检查文件后重试。") from error
    else:
        raise DataValidationError("目前支持 CSV、Excel .xlsx 和 .xls 文件。")
    if raw.empty or header_row >= len(raw) or raw.shape[1] == 0:
        raise DataValidationError("所选表头行不存在，或表格为空。")
    raw = raw.map(_text) if hasattr(raw, "map") else raw.applymap(_text)
    header = tuple(value.strip() or f"未命名列 {idx + 1}" for idx, value in enumerate(raw.iloc[header_row]))
    frame = raw.iloc[header_row + 1:].copy().reset_index(drop=True)
    frame.columns = range(len(header))
    return TablePreview(header, frame, header_row, filename, selected, tuple(row_numbers[header_row + 1:]))


def suggest_mappings(table: TablePreview) -> dict[str, int]:
    """Suggest only exact single matches; never resolve competing matches."""
    normalized = [name.strip().casefold() for name in table.columns]
    suggestions: dict[str, int] = {}
    for field, aliases in FIELD_ALIASES.items():
        normalized_aliases = {alias.casefold() for alias in aliases}
        found = [index for index, name in enumerate(normalized) if name in normalized_aliases]
        if len(found) == 1:
            suggestions[field] = found[0]
    # A source cannot be used as two different business fields automatically.
    counts = {index: list(suggestions.values()).count(index) for index in suggestions.values()}
    return {field: index for field, index in suggestions.items() if counts[index] == 1}


def _validate_mapping(table: TablePreview, mapping: Mapping[str, int]) -> None:
    unknown = set(mapping) - set(FIELD_ALIASES)
    if unknown:
        raise DataValidationError("存在不支持的目标字段。")
    positions = list(mapping.values())
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(table.columns) for index in positions):
        raise DataValidationError("字段映射包含无效的列。")
    if len(set(positions)) != len(positions):
        raise DataValidationError("同一列不能映射为多个字段，请检查字段映射。")


def _numeric(value: str) -> float | None:
    value = value.strip()
    if not _NUMBER.fullmatch(value):
        return None
    result = float(value.replace(",", ""))
    return result if math.isfinite(result) else None


def _is_ambiguous_date(value: str) -> bool:
    match = _AMBIGUOUS_DATE.match(value.strip())
    return bool(match and int(match[1]) <= 12 and int(match[2]) <= 12 and int(match[1]) != int(match[2]))


def inspect_quality(table: TablePreview, mapping: Mapping[str, int]) -> QualityPreview:
    _validate_mapping(table, mapping)
    frame = table.frame
    # An otherwise-empty required subset is not enough: inspect every original column.
    empty = frame.apply(lambda column: column.str.strip().eq("")).all(axis=1)
    nonempty = frame.loc[~empty]
    duplicates = nonempty.duplicated(keep="first")
    def physical(indices) -> tuple[int, ...]:
        return tuple(table.row_numbers[int(i)] if table.row_numbers else int(i) + table.header_row + 2 for i in indices)
    invalid_numeric: dict[str, tuple[int, ...]] = {}
    for field in NUMERIC_FIELDS.intersection(mapping):
        values = nonempty[mapping[field]]
        bad = values.map(lambda item: _numeric(item) is None)
        invalid_numeric[field] = physical(values.index[bad])
    invalid_dates: tuple[int, ...] = ()
    ambiguous_dates: tuple[int, ...] = ()
    if "date" in mapping:
        values = nonempty[mapping["date"]].str.strip()
        ambiguous = values.map(_is_ambiguous_date)
        timezone = values.str.contains(_TIMEZONE_DATE, regex=True)
        parsed = pd.to_datetime(values.mask(timezone), errors="coerce", format="mixed")
        invalid_dates = physical(values.index[(parsed.isna() | timezone) & ~ambiguous])
        ambiguous_dates = physical(values.index[ambiguous])
    return QualityPreview(len(frame), physical(frame.index[empty]), physical(nonempty.index[duplicates]), invalid_numeric, invalid_dates, ambiguous_dates)


def prepare_table(table: TablePreview, mapping: Mapping[str, int], *, remove_duplicates: bool = False) -> PreparedData:
    """Apply confirmed mappings and safe normalization. Reject invalid values.

    Unknown columns are preserved (with unique names). Empty rows are removed;
    exact duplicates are preserved unless the caller explicitly opts in.
    """
    _validate_mapping(table, mapping)
    order_profile = all(field in mapping for field in ORDER_FIELDS)
    revenue_profile = all(field in mapping for field in ("date", "revenue"))
    if not order_profile and not revenue_profile:
        raise DataValidationError("请选择日期和销售额，或映射完整订单字段后再导入。")
    profile = "orders" if order_profile else "revenue"
    needed = set(ORDER_FIELDS if order_profile else ("date", "revenue"))
    if not order_profile and "channel" in mapping:
        needed.add("channel")
    quality = inspect_quality(table, mapping)
    frame = table.frame.copy()
    empty = frame.apply(lambda column: column.str.strip().eq("")).all(axis=1)
    frame = frame.loc[~empty].copy()
    if remove_duplicates:
        frame = frame.loc[~frame.duplicated(keep="first")].copy()
    if frame.empty:
        raise DataValidationError("所选表格没有可分析的数据。")
    for field in needed:
        values = frame[mapping[field]].str.strip()
        if values.eq("").any():
            raise DataValidationError(f"{field} 存在空值，请检查预览后修改原表或映射。")
    for field in NUMERIC_FIELDS.intersection(needed):
        converted = frame[mapping[field]].map(_numeric)
        if converted.isna().any():
            raise DataValidationError(f"{field} 存在无法识别的数字，请检查预览。")
        if field == "price" and (converted < 0).any():
            raise DataValidationError("单价不能为负数。")
        if field == "quantity" and ((converted <= 0) | (converted % 1 != 0)).any():
            raise DataValidationError("数量必须是大于 0 的整数。")
        frame[mapping[field]] = converted
    dates = frame[mapping["date"]].str.strip()
    if dates.map(_is_ambiguous_date).any():
        raise DataValidationError("日期存在月日顺序不明确的值，请先在原表统一为年-月-日。")
    if dates.str.contains(_TIMEZONE_DATE, regex=True).any():
        raise DataValidationError("日期包含时区，暂不支持自动转换，请先统一到业务时区。")
    parsed = pd.to_datetime(dates, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise DataValidationError("日期存在无法识别的值，请检查预览。")
    frame[mapping["date"]] = parsed.map(lambda value: value.isoformat(sep=" "))
    # Use canonical names for selected columns, safe unique names for others.
    canonical = {index: field for field, index in mapping.items()}
    used = {name.casefold() for name in mapping}
    names: list[str] = []
    for index, name in enumerate(table.columns):
        if index in canonical:
            names.append(canonical[index])
            continue
        candidate = name if name.casefold() not in _RESERVED_NAMES else f"原始列_{index + 1}_{name}"
        base = candidate
        n = 2
        while candidate.casefold() in used or candidate.casefold() in _RESERVED_NAMES:
            candidate = f"{base}_{n}"
            n += 1
        used.add(candidate.casefold())
        names.append(candidate)
    frame.columns = names
    frame = frame.reset_index(drop=True)
    csv_content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return PreparedData(frame, profile, csv_content, quality)
