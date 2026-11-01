"""CSV/Excel 导入、清洗、指标计算与可选字段能力检测。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "orders.csv"
REQUIRED_COLUMNS = ("user_id", "product", "price", "quantity", "date", "gender", "grade")
TEXT_COLUMNS = ("user_id", "product", "gender", "grade")
OPTIONAL_FIELD_ALIASES = {
    "order_id": ("order_id", "order_no", "order_number", "订单id", "订单编号"),
    "unit_cost": ("unit_cost", "cost_per_unit", "unit_cost_price", "单位成本", "成本单价"),
    "total_cost": ("total_cost", "order_cost", "cost", "总成本", "成本"),
    "profit": ("profit", "gross_profit", "利润", "毛利"),
    "channel": ("channel", "source", "acquisition_channel", "marketing_channel", "渠道", "获客渠道", "来源"),
    "acquisition_cost": (
        "acquisition_cost", "cac_cost", "marketing_cost", "ad_spend", "channel_cost",
        "获客成本", "营销成本", "广告花费",
    ),
}


class DataValidationError(ValueError):
    def __init__(self, message: str, *, missing_fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing_fields = missing_fields


@dataclass(frozen=True)
class DataQualityReport:
    original_rows: int
    cleaned_rows: int
    empty_rows_removed: int
    duplicate_rows_removed: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DataOverview:
    row_count: int
    user_count: int
    product_count: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


@dataclass(frozen=True)
class DataCapabilities:
    order_id_field: str | None
    unit_cost_field: str | None
    total_cost_field: str | None
    profit_field: str | None
    channel_field: str | None
    acquisition_cost_field: str | None
    repurchase_reliable: bool
    profit_supported: bool
    channel_supported: bool
    channel_cost_supported: bool
    limitations: tuple[str, ...]

    def status_items(self) -> tuple[tuple[str, str, str], ...]:
        return (
            ("商品分析", "✓", "字段完整"),
            ("用户分析", "✓", "字段完整"),
            ("复购分析", "✓" if self.repurchase_reliable else "△", "可按订单号准确统计" if self.repurchase_reliable else "缺少订单号，仅能按每行一单估算"),
            ("趋势分析", "✓", "可按日期统计"),
            ("渠道分析", "✓" if self.channel_supported else "✗", f"字段：{self.channel_field}" if self.channel_supported else "缺少渠道字段"),
            ("利润分析", "✓" if self.profit_supported else "✗", "存在可用成本/利润字段" if self.profit_supported else "缺少可用成本或利润字段"),
        )


@dataclass(frozen=True)
class BusinessMetrics:
    total_sales: float
    order_count: int
    avg_order_value: float
    total_quantity: int
    unique_users: int
    top_product: str
    top_gender: str
    top_grade: str
    top_product_ratio: float
    product_sales: pd.Series
    daily_sales: pd.Series
    gender_count: pd.Series
    grade_count: pd.Series
    grade_sales: pd.Series

    def ai_context(self) -> dict[str, Any]:
        return {
            "total_sales": self.total_sales, "order_count": self.order_count,
            "avg_order_value": self.avg_order_value, "total_quantity": self.total_quantity,
            "unique_users": self.unique_users, "top_product": self.top_product,
            "top_gender": self.top_gender, "top_grade": self.top_grade,
            "top_product_ratio": self.top_product_ratio,
            "product_sales": self.product_sales.to_dict(), "grade_sales": self.grade_sales.to_dict(),
        }


def read_default_order_bytes(path: str | Path = DEFAULT_DATA_PATH) -> bytes:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"订单数据文件不存在：{data_path}")
    return data_path.read_bytes()


def fingerprint_csv(content: bytes) -> str:
    return sha256(content).hexdigest()


def _resolve_optional_field(orders: pd.DataFrame, canonical: str) -> str | None:
    columns = {str(column).strip().lower(): str(column) for column in orders.columns}
    return next((columns[alias.lower()] for alias in OPTIONAL_FIELD_ALIASES[canonical] if alias.lower() in columns), None)


def _field_complete(orders: pd.DataFrame, field: str | None) -> bool:
    if field is None:
        return False
    values = orders[field]
    return bool((values.notna() & values.astype("string").str.strip().ne("")).all())


def _numeric_usable(orders: pd.DataFrame, field: str | None) -> bool:
    if not _field_complete(orders, field):
        return False
    values = pd.to_numeric(orders[field], errors="coerce")
    return bool(values.notna().all() and values.map(lambda value: abs(float(value)) != float("inf")).all())


def detect_data_capabilities(orders: pd.DataFrame) -> DataCapabilities:
    order_id = _resolve_optional_field(orders, "order_id")
    unit_cost = _resolve_optional_field(orders, "unit_cost")
    total_cost = _resolve_optional_field(orders, "total_cost")
    profit = _resolve_optional_field(orders, "profit")
    channel = _resolve_optional_field(orders, "channel")
    acquisition_cost = _resolve_optional_field(orders, "acquisition_cost")
    reliable = _field_complete(orders, order_id)
    usable_unit = _numeric_usable(orders, unit_cost)
    usable_total = _numeric_usable(orders, total_cost)
    usable_profit = _numeric_usable(orders, profit)
    channel_ok = _field_complete(orders, channel)
    channel_cost_ok = channel_ok and _numeric_usable(orders, acquisition_cost)
    limitations = []
    if not reliable:
        limitations.append("缺少完整 order_id，复购只能按每行代表一笔订单估算。")
    if not (usable_unit or usable_total or usable_profit):
        limitations.append("缺少可用成本或利润字段，无法判断商品利润。")
    if not channel_ok:
        limitations.append("缺少 channel/source/acquisition_channel 等渠道字段。")
    elif not channel_cost_ok:
        limitations.append("存在渠道字段，但缺少可用获客成本字段，无法计算 CAC/ROI。")
    return DataCapabilities(
        order_id if reliable else None, unit_cost if usable_unit else None,
        total_cost if usable_total else None, profit if usable_profit else None,
        channel if channel_ok else None, acquisition_cost if channel_cost_ok else None,
        reliable, bool(usable_unit or usable_total or usable_profit), channel_ok,
        channel_cost_ok, tuple(limitations),
    )


def _read_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    readable = BytesIO(source) if isinstance(source, bytes) else source
    if hasattr(readable, "read"):
        payload = readable.read()
        readable = StringIO(payload) if isinstance(payload, str) else BytesIO(payload)
    try:
        # IDs are labels, even when all their characters happen to be digits.
        # Inspect original header spelling so case/whitespace aliases also retain
        # leading zeros and long identifiers when an Excel sheet becomes CSV.
        header = pd.read_csv(readable, nrows=0)
        identifiers = {"user_id", *OPTIONAL_FIELD_ALIASES["order_id"]}
        text_ids = {
            column: "string" for column in header.columns
            if str(column).strip().lower() in identifiers
        }
        if hasattr(readable, "seek"):
            readable.seek(0)
        return pd.read_csv(readable, dtype=text_ids)
    except EmptyDataError as error:
        raise DataValidationError("CSV 文件为空或不包含表头。") from error
    except UnicodeDecodeError as error:
        raise DataValidationError("CSV 编码无法识别，请保存为 UTF-8 后重试。") from error
    except ParserError as error:
        raise DataValidationError(f"CSV 格式无法解析：{error}") from error


def _rows(mask: pd.Series) -> str:
    return "、".join(str(int(index) + 2) for index in mask.index[mask])


def process_order_csv(source: str | Path | bytes | BinaryIO) -> tuple[pd.DataFrame, DataQualityReport]:
    orders = _read_csv(source)
    orders.columns = [str(column).strip() for column in orders.columns]
    missing = tuple(column for column in REQUIRED_COLUMNS if column not in orders.columns)
    if missing:
        raise DataValidationError(f"当前文件缺少字段：{'、'.join(missing)}", missing_fields=missing)
    if orders.empty:
        raise DataValidationError("CSV 中没有订单数据。")
    original_rows = len(orders)
    required = orders[list(REQUIRED_COLUMNS)].astype("string")
    empty_mask = (required.isna() | required.apply(lambda column: column.str.strip().eq(""))).all(axis=1)
    empty_removed = int(empty_mask.sum())
    orders = orders.loc[~empty_mask].copy()
    duplicate_mask = orders.duplicated(keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    orders = orders.loc[~duplicate_mask].copy()
    if orders.empty:
        raise DataValidationError("清理空行和重复行后没有可分析的数据。")
    blank_masks = {column: orders[column].isna() | orders[column].astype("string").str.strip().eq("") for column in REQUIRED_COLUMNS}
    blank_fields = [column for column, mask in blank_masks.items() if mask.any()]
    if blank_fields:
        combined = pd.concat([blank_masks[column] for column in blank_fields], axis=1).any(axis=1)
        raise DataValidationError(f"核心字段存在空数据：{'、'.join(blank_fields)}（CSV 行：{_rows(combined)}）")
    price = pd.to_numeric(orders["price"], errors="coerce")
    quantity = pd.to_numeric(orders["quantity"], errors="coerce")
    invalid_price = price.isna() | price.isin([float("inf"), float("-inf")])
    invalid_quantity = quantity.isna() | quantity.isin([float("inf"), float("-inf")])
    if invalid_price.any():
        raise DataValidationError(f"price 存在无法转换为数字的值（CSV 行：{_rows(invalid_price)}）")
    if invalid_quantity.any():
        raise DataValidationError(f"quantity 存在无法转换为数字的值（CSV 行：{_rows(invalid_quantity)}）")
    dates = pd.to_datetime(orders["date"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise DataValidationError(f"date 存在无法解析的日期（CSV 行：{_rows(dates.isna())}）")
    if (price < 0).any() or (quantity <= 0).any() or (quantity % 1 != 0).any():
        raise DataValidationError("price 不能为负数，quantity 必须为大于 0 的整数。")
    for column in TEXT_COLUMNS:
        orders[column] = orders[column].astype("string").str.strip()
    orders["price"], orders["quantity"], orders["date"] = price.astype(float), quantity.astype(int), dates
    orders["sales"] = orders["price"] * orders["quantity"]
    orders = orders.sort_values("date").reset_index(drop=True)
    warnings = []
    if empty_removed:
        warnings.append(f"已安全移除 {empty_removed} 行完全空白数据。")
    if duplicates_removed:
        warnings.append(f"检测到并移除 {duplicates_removed} 行完全重复数据。")
    return orders, DataQualityReport(original_rows, len(orders), empty_removed, duplicates_removed, tuple(warnings))


def _excel_workbook(content: bytes, *, filename: str) -> pd.ExcelFile:
    """Open a workbook in memory; the caller owns and closes the handle."""

    suffix = Path(filename).suffix.lower()
    engines = {".xlsx": "openpyxl", ".xls": "xlrd"}
    if suffix not in engines:
        raise DataValidationError("请选择 Excel 文件（.xlsx 或 .xls）。")
    try:
        return pd.ExcelFile(BytesIO(content), engine=engines[suffix])
    except ImportError as error:
        raise DataValidationError(
            f"Excel 读取组件 {engines[suffix]} 不可用，请安装 requirements.txt 中的依赖。"
        ) from error
    except Exception as error:
        raise DataValidationError(
            "Excel 文件无法读取，请确认文件未损坏、未加密，并重新保存后上传。"
        ) from error


def excel_sheet_names(content: bytes, *, filename: str) -> tuple[str, ...]:
    """Inspect sheet names without combining tables or writing uploaded files."""

    with _excel_workbook(content, filename=filename) as workbook:
        names = tuple(workbook.sheet_names)
    if not names:
        raise DataValidationError("Excel 中没有可读取的工作表。")
    return names


def prepare_order_upload(
    content: bytes, *, filename: str, sheet_name: str | None = None
) -> bytes:
    """Validate CSV or one Excel sheet and return the existing CSV input contract.

    Keep raw worksheet rows so downstream quality reports still count duplicate
    and blank rows. Workbook formatting/formulas are not executed or persisted.
    """

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        process_order_csv(content)
        return content
    if suffix not in {".xlsx", ".xls"}:
        raise DataValidationError("目前支持 CSV、Excel .xlsx 和 .xls 文件。")
    selected = sheet_name
    try:
        with _excel_workbook(content, filename=filename) as workbook:
            if not workbook.sheet_names:
                raise DataValidationError("Excel 中没有可读取的工作表。")
            if selected is None:
                selected = workbook.sheet_names[0]
            if selected not in workbook.sheet_names:
                raise DataValidationError(f"Excel 中找不到工作表“{selected}”。")
            # object keeps text identifiers as stored, rather than coercing them
            # into numbers. Native date cells remain real date/time values.
            frame = workbook.parse(sheet_name=selected, dtype=object)
    except DataValidationError:
        raise
    except Exception as error:
        raise DataValidationError(
            "Excel 工作表无法读取，请确认文件未损坏、未加密，并重新保存后上传。"
        ) from error
    normalized = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    try:
        process_order_csv(normalized)
    except DataValidationError as error:
        message = str(error).replace("CSV", "表格")
        raise DataValidationError(
            f"工作表“{selected}”：{message}", missing_fields=error.missing_fields
        ) from error
    return normalized


def load_order_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return process_order_csv(path)[0]


def summarize_orders(orders: pd.DataFrame) -> DataOverview:
    return DataOverview(len(orders), int(orders["user_id"].nunique()), int(orders["product"].nunique()), pd.Timestamp(orders["date"].min()), pd.Timestamp(orders["date"].max()))


def calculate_metrics(orders: pd.DataFrame) -> BusinessMetrics:
    if orders.empty or "sales" not in orders:
        raise DataValidationError("订单数据为空或尚未计算 sales 字段。")
    product_sales = orders.groupby("product")["sales"].sum().sort_values(ascending=False)
    daily_sales = orders.groupby("date")["sales"].sum().sort_index()
    gender_count = orders.groupby("gender")["user_id"].nunique().sort_values(ascending=False)
    grade_count = orders.groupby("grade")["user_id"].nunique().sort_values(ascending=False)
    grade_sales = orders.groupby("grade")["sales"].sum().sort_values(ascending=False)
    total_sales = float(orders["sales"].sum())
    capabilities = detect_data_capabilities(orders)
    order_count = int(orders[capabilities.order_id_field].nunique()) if capabilities.repurchase_reliable and capabilities.order_id_field else len(orders)
    return BusinessMetrics(
        total_sales, int(order_count), total_sales / order_count, int(orders["quantity"].sum()),
        int(orders["user_id"].nunique()), str(product_sales.index[0]), str(gender_count.index[0]),
        str(grade_sales.index[0]), float(product_sales.iloc[0] / total_sales) if total_sales else 0.0,
        product_sales, daily_sales, gender_count, grade_count, grade_sales,
    )
