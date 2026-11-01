"""Excel upload boundary and UI regressions; no external model requests."""

from __future__ import annotations

import base64
import csv
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
import zlib

import pandas as pd
import pytest
import streamlit as st
from openpyxl import Workbook
from streamlit.testing.v1 import AppTest

from datalume.analysis_engine import run_analysis
from datalume.database_manager import (
    create_database,
    initialize_database_state,
    replace_database_data,
)
from datalume.data_processor import (
    DataValidationError,
    calculate_metrics,
    detect_data_capabilities,
    excel_sheet_names,
    prepare_order_upload,
    process_order_csv,
    summarize_orders,
)
from datalume.report_generator import generate_diagnostic_report
from datalume.router import QuestionType


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
HEADERS = ["date", "user_id", "product", "price", "quantity", "gender", "grade"]
ROWS = [
    [datetime(2026, 1, 1), "001", "欧包", 10, 2, "女", "大一"],
    [datetime(2026, 1, 2), "002", "酸奶", 14, 3, "男", "大二"],
]


@pytest.fixture(autouse=True)
def disable_live_api(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")


def _xlsx(*sheets):
    """Build (name, headers, rows) sheets without filesystem artifacts."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, headers, rows in sheets:
        sheet = workbook.create_sheet(name)
        sheet.append(list(headers))
        for row in rows:
            sheet.append(list(row))
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def _csv(headers, rows):
    text = StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def _prepared(content, sheet_name=None):
    return prepare_order_upload(content, filename="订单.xlsx", sheet_name=sheet_name)


def _rich_source():
    headers = ["order_id", *HEADERS, "unit_cost", "channel", "acquisition_cost"]
    rows = []
    for day in range(1, 29):
        bread = bool(day % 2)
        rows.append([
            f"{day:05d}", datetime(2026, 8, day), f"{(day - 1) % 12 + 1:03d}",
            "欧包" if bread else "酸奶", 20 if bread else 12, 1, "女", "大一",
            8 if bread else 7, "社群" if day % 3 else "海报", 2 if day % 3 else 4,
        ])
    return headers, rows


def test_csv_upload_returns_original_bytes():
    original = _csv(HEADERS, ROWS)
    assert prepare_order_upload(original, filename="orders.csv") == original
    orders, _ = process_order_csv(original)
    assert calculate_metrics(orders).total_sales == 62


def test_sheet_names_and_native_excel_dates_are_loaded():
    content = _xlsx(("订单数据", HEADERS, ROWS))
    assert excel_sheet_names(content, filename="订单.XLSX") == ("订单数据",)
    orders, quality = process_order_csv(
        prepare_order_upload(content, filename="订单.XLSX")
    )
    assert quality.cleaned_rows == 2
    assert orders["date"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]
    assert calculate_metrics(orders).total_sales == 62
    assert orders["user_id"].tolist() == ["001", "002"]


def test_legacy_xls_reader_preserves_dates_and_text_identifiers():
    # Compressed BIFF8 fixture generated with xlwt; tests need only the xlrd reader.
    fixture = (
        "eNrtWE1IFGEYfr7ZHXfd1t3VNNBAREjLFNQORaBbYXlSLInaCGx1BlmMXZt2IwvKMk8VFHUpuhh26GJ16YcK6tYhMOoQBIHWsVNQ0EGd3u+dn9QKXChBmWf43nl/5313vt+dN5PFU2MPKqaxCK3wYc4sRME8naBW6AgxkN00Jevcg9RMDysKhUHqyAIVT4teB2Qfyv6ehoL7/pdEgU/UDmMQnZm0XrWM2MU1JIWsoYWowC3SRFDOVZUw7WO6luk99nzGdAdrLjNtId8pcQiT8c66bfYoPqhUsy0C+dxHHPOBNU0owys5is9eEZavip1GKnl0hRkq/WGMgzq0XU/rhpSjuIMQkCA0dHQ0tLVNoZQ6exzfzSrgmzOrX1R5+uXVC5D+x0J94C/6IHXiYv01xQ8MwzzCk2QUYZz2S4sibl/c3z2D6zQBwI3UWjKr06jIHdeNnpRG3KCR0XJ9WVVyqT6dVvBjuWQ6m8oO0ULQr6c13SBbv5HUpC1jaBzoAxobmyhf7YDaLSCOJySfQKcqDWRih2ZSnrx6IkEOW3PscMlxaA7JrYSXntiCpaeIp2SYqIYo88U8MWP022fufn3b0dsV72HNMG831qa0Qb4DmDgnIyg4whYfnE2pjiM2Mz3PT13PfAXTUpo1dK/pKrOZPSPsc4GtNfSELYx38dp5/EbiR7/sfVw5+jm+ifiJ9ulTpRPv42Oopk1So3h5jaBe1IubNySexJ27sBewj0zLf1vMgkrMrt20d94oZuX0JRQztST5doQrKfa7siQfST5X8pPkdyWVJNWVCkgqcKUASQG7BrGohjNKiHUlqJxU2F9wDUE7WnAN8ojwnHwE11DHNVs1WD6qLVuSzL7GlWT2sJ1d+UN2a/GvtrNb46PIjlY4e8TOrnD27Zw94mZXOHvUjZDZY64ks8v326qU4KFMRVvJL4TgwYMHDx48ePCwCiD44Ggdlv18NLQO4gH7u84stTnvM8mqxT5k6MrSn8rdSNPdwFBe42cdVOE8SywxxvleKHGAshsYQC/XMZD3+KWDvpj/e5YcGPt3Uyjf/HP51Pmf8/8Euyb9/g=="
    )
    content = zlib.decompress(base64.b64decode(fixture))
    assert excel_sheet_names(content, filename="legacy.xls") == ("订单",)
    normalized = prepare_order_upload(content, filename="legacy.xls", sheet_name="订单")
    orders, _ = process_order_csv(normalized)
    assert orders["user_id"].tolist() == ["001", "002"]
    assert orders["order_id"].tolist() == ["00001", "00002"]
    assert orders["date"].tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]
    assert calculate_metrics(orders).total_sales == 62


def test_selected_nonfirst_sheet_controls_the_import():
    content = _xlsx(
        ("说明", ["说明"], [["请导入第二张工作表"]]),
        ("实际订单", HEADERS, ROWS),
    )
    assert excel_sheet_names(content, filename="book.xlsx") == ("说明", "实际订单")
    with pytest.raises(DataValidationError) as error:
        _prepared(content, "说明")
    assert "price" in error.value.missing_fields
    orders, _ = process_order_csv(_prepared(content, "实际订单"))
    assert calculate_metrics(orders).total_sales == 62


def test_unknown_sheet_is_a_validation_error():
    content = _xlsx(("订单", HEADERS, ROWS))
    with pytest.raises(DataValidationError):
        _prepared(content, "不存在的表")


@pytest.mark.parametrize("filename", ["orders.txt", "orders.zip", "orders.xlsm"])
def test_unsupported_upload_extension_is_rejected(filename):
    with pytest.raises(DataValidationError):
        prepare_order_upload(_csv(HEADERS, ROWS), filename=filename)


@pytest.mark.parametrize("reader", [excel_sheet_names, prepare_order_upload])
@pytest.mark.parametrize("content", [b"", b"not an Excel workbook"])
def test_corrupt_excel_is_a_validation_error(reader, content):
    with pytest.raises(DataValidationError):
        reader(content, filename="broken.xlsx")


@pytest.mark.parametrize("order_header", ["order_id", " ORDER_NUMBER ", " 订单编号 "])
def test_text_identifier_precision_and_order_aliases_survive(order_header):
    headers = [order_header, *HEADERS]
    rows = [
        ["00001", datetime(2026, 1, 1), "001", "欧包", 10, 1, "女", "大一"],
        ["1234567890123456789012345", datetime(2026, 1, 2), "1234567890123456789012345", "欧包", 10, 1, "女", "大一"],
    ]
    content = _xlsx(("订单", headers, rows))
    orders, _ = process_order_csv(_prepared(content))
    field = detect_data_capabilities(orders).order_id_field
    assert field == order_header.strip()
    assert orders[field].tolist() == ["00001", "1234567890123456789012345"]
    assert orders["user_id"].tolist() == ["001", "1234567890123456789012345"]
    assert calculate_metrics(orders).order_count == 2


def test_csv_text_identifiers_and_padded_headers_are_preserved():
    headers = [" ORDER_ID ", *[" user_id " if field == "user_id" else field for field in HEADERS]]
    rows = [["00001", *ROWS[0]], ["1234567890123456789012345", *ROWS[1]]]
    orders, _ = process_order_csv(_csv(headers, rows))
    assert orders["user_id"].tolist() == ["001", "002"]
    assert orders["ORDER_ID"].tolist() == ["00001", "1234567890123456789012345"]


@pytest.mark.parametrize("filename", ["orders.csv", "orders.xlsx"])
def test_existing_na_identifier_validation_is_unchanged(filename):
    rows = [list(row) for row in ROWS]
    rows[0][HEADERS.index("user_id")] = "NA"
    content = _csv(HEADERS, rows) if filename.endswith(".csv") else _xlsx(("订单", HEADERS, rows))
    with pytest.raises(DataValidationError, match="user_id"):
        prepare_order_upload(content, filename=filename)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("price", "not-a-number", "price"),
        ("quantity", "not-a-number", "quantity"),
        ("date", "not-a-date", "date"),
        ("user_id", None, "空数据"),
        ("product", "  ", "空数据"),
        ("price", -1, "price"),
        ("quantity", 0, "quantity"),
        ("quantity", 1.5, "quantity"),
    ],
)
def test_excel_reuses_core_order_validation(column, value, message):
    rows = [list(row) for row in ROWS]
    rows[0][HEADERS.index(column)] = value
    with pytest.raises(DataValidationError, match=message):
        _prepared(_xlsx(("订单", HEADERS, rows)))


def test_missing_excel_fields_are_named():
    content = _xlsx(("订单", HEADERS[:-2], [row[:-2] for row in ROWS]))
    with pytest.raises(DataValidationError) as error:
        _prepared(content)
    assert error.value.missing_fields == ("gender", "grade")


def test_header_only_excel_sheet_is_rejected():
    with pytest.raises(DataValidationError):
        _prepared(_xlsx(("订单", HEADERS, [])))


def test_normalization_preserves_raw_rows_for_quality_reporting():
    headers = [*HEADERS, "note"]
    row = [*ROWS[0], "正常订单"]
    rows = [row, [*[None] * len(HEADERS), "核心字段完全空白"], row]
    normalized = _prepared(_xlsx(("订单", headers, rows)))
    orders, quality = process_order_csv(normalized)
    assert len(orders) == 1
    assert quality.original_rows == 3
    assert quality.empty_rows_removed == 1
    assert quality.duplicate_rows_removed == 1
    assert len(quality.warnings) == 2


def test_all_business_analysis_and_report_match_equivalent_csv():
    headers, rows = _rich_source()
    csv_orders, csv_quality = process_order_csv(_csv(headers, rows))
    excel_orders, excel_quality = process_order_csv(
        _prepared(_xlsx(("明细", headers, rows)))
    )
    pd.testing.assert_frame_equal(excel_orders, csv_orders)
    assert excel_quality == csv_quality
    csv_caps = detect_data_capabilities(csv_orders)
    excel_caps = detect_data_capabilities(excel_orders)
    assert excel_caps == csv_caps
    assert all((excel_caps.repurchase_reliable, excel_caps.profit_supported,
                excel_caps.channel_supported, excel_caps.channel_cost_supported))
    csv_metrics = calculate_metrics(csv_orders)
    excel_metrics = calculate_metrics(excel_orders)
    assert excel_metrics.total_sales == 448
    assert excel_metrics.order_count == 28
    for intent in QuestionType:
        expected = run_analysis(intent, "最近一周经营情况和商品利润怎么样？", csv_orders, csv_metrics, csv_caps)
        actual = run_analysis(intent, "最近一周经营情况和商品利润怎么样？", excel_orders, excel_metrics, excel_caps)
        assert actual.supported
        assert actual == expected
    expected_report = generate_diagnostic_report(
        csv_orders, csv_metrics, summarize_orders(csv_orders), csv_caps
    )
    actual_report = generate_diagnostic_report(
        excel_orders, excel_metrics, summarize_orders(excel_orders), excel_caps
    )
    assert actual_report == expected_report
    assert actual_report.to_markdown() == expected_report.to_markdown()


def test_only_selected_sheet_content_controls_the_source_fingerprint():
    first = _xlsx(("说明", ["notes"], [["old note"]]), ("订单", HEADERS, ROWS))
    second = _xlsx(("说明", ["notes"], [["new unrelated note"]]), ("订单", HEADERS, ROWS))
    assert _prepared(first, "订单") == _prepared(second, "订单")


def _seed_private_state(project, marker):
    project.chat_messages.append({"role": "user", "content": marker})
    project.conversation_context["marker"] = marker
    project.agent_history.append({"marker": marker})
    project.diagnostic_report = marker
    project.diagnostic_report_source_id = project.data_source_id


def test_excel_sheet_replacement_clears_only_current_database_and_keeps_identity():
    changed_rows = [list(row) for row in ROWS]
    changed_rows[0][HEADERS.index("price")] = 30
    workbook = _xlsx(("原始", HEADERS, ROWS), ("更新", HEADERS, changed_rows))
    state = {}
    initialize_database_state(state)
    other = create_database(state, "其他项目", _csv(HEADERS, ROWS))
    current = create_database(state, "当前项目", _prepared(workbook, "原始"), pinned=True)
    _seed_private_state(current, "当前旧内容")
    _seed_private_state(other, "其他内容")
    database_id = current.database_id
    source_id = current.data_source_id
    assert replace_database_data(current, _prepared(workbook, "更新"))
    assert current.database_id == database_id
    assert state["active_database_id"] == database_id
    assert current.name == "当前项目" and current.pinned
    assert current.data_source_id != source_id
    assert current.chat_messages == [] and current.conversation_context == {}
    assert current.agent_history == [] and current.diagnostic_report is None
    assert current.diagnostic_report_source_id is None
    assert other.chat_messages[0]["content"] == "其他内容"
    assert other.conversation_context == {"marker": "其他内容"}
    assert other.agent_history == [{"marker": "其他内容"}]
    assert other.diagnostic_report == "其他内容"
    orders, _ = process_order_csv(current.csv_content)
    assert calculate_metrics(orders).total_sales == 102


def test_reupload_same_excel_sheet_keeps_private_state():
    content = _xlsx(("订单", HEADERS, ROWS))
    state = {}
    initialize_database_state(state)
    current = create_database(state, "项目", _prepared(content))
    _seed_private_state(current, "保留")
    assert not replace_database_data(current, _prepared(content))
    assert current.chat_messages[0]["content"] == "保留"
    assert current.conversation_context == {"marker": "保留"}
    assert current.agent_history == [{"marker": "保留"}]
    assert current.diagnostic_report == "保留"
    assert current.diagnostic_report_source_id == current.data_source_id


class NamedUpload(BytesIO):
    def __init__(self, content, name="orders.xlsx"):
        super().__init__(content)
        self.name = name


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_app_creates_excel_database_from_selected_sheet(monkeypatch):
    content = _xlsx(("说明", ["notes"], [["not orders"]]), ("实际订单", HEADERS, ROWS))
    uploaded = NamedUpload(content)
    uploader_calls = []

    def fake_uploader(label, *args, **kwargs):
        uploader_calls.append((label, kwargs.get("type")))
        value = None if "替换" in label else uploaded
        # A real file_uploader registers its widget key; reproduce that contract
        # so the dialog's successful-create cleanup can delete the upload value.
        st.session_state[kwargs["key"]] = value
        return value

    monkeypatch.setattr(st, "file_uploader", fake_uploader)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    _button(app, "新建数据库").click().run()
    next(field for field in app.text_input if field.label == "数据库名称").input("Excel 导入")
    next(field for field in app.selectbox if field.label == "选择工作表").select("实际订单")
    app.run()
    _button(app, "创建数据库").click().run()
    assert not app.exception
    assert len(app.session_state["databases"]) == 1
    current = app.session_state["databases"][app.session_state["active_database_id"]]
    assert current.name == "Excel 导入"
    assert calculate_metrics(process_order_csv(current.csv_content)[0]).total_sales == 62
    assert any(label == "上传订单 CSV 或 Excel" and set(types) == {"csv", "xlsx", "xls"}
               for label, types in uploader_calls)
    assert len(app.chat_input) == 1


def test_app_confirms_excel_replacement_from_selected_sheet(monkeypatch):
    changed = [list(row) for row in ROWS]
    changed[0][HEADERS.index("price")] = 30
    uploaded = NamedUpload(_xlsx(("原始", HEADERS, ROWS), ("更新", HEADERS, changed)))
    monkeypatch.setattr(st, "file_uploader", lambda label, *args, **kwargs: uploaded if "替换" in label else None)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    current = create_database(app.session_state, "待更新", _csv(HEADERS, ROWS), pinned=True)
    _seed_private_state(current, "旧内容")
    database_id = current.database_id
    app.run()
    next(field for field in app.selectbox if field.label == "选择工作表").select("更新").run()
    assert current.chat_messages[0]["content"] == "旧内容"
    _button(app, "确认替换数据").click().run()
    assert not app.exception
    assert current.database_id == database_id and current.pinned
    assert current.chat_messages == [] and current.agent_history == []
    assert current.conversation_context == {} and current.diagnostic_report is None
    assert calculate_metrics(process_order_csv(current.csv_content)[0]).total_sales == 102
