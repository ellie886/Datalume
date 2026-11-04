"""Regression tests for confirmed mappings and limited data analysis."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from datalume.data_preparation import (
    inspect_quality,
    prepare_table,
    read_table,
    suggest_mappings,
)
from datalume.data_processor import DataValidationError, process_order_csv
from datalume.database_manager import create_database, initialize_database_state, replace_database_data
from datalume.partial_analysis import build_partial_summary


def _table(text, *, header_row=0):
    return read_table(text.encode("utf-8"), filename="表格.csv", header_row=header_row)


def test_chinese_headers_selected_header_row_and_identifier_preserved():
    table = _table("经营报表,,,,\n日期,销售额,渠道,用户编号,备注\n2026-01-01,23,门店,00123,首单\n", header_row=1)
    mapping = suggest_mappings(table)
    assert mapping == {"date": 0, "revenue": 1, "channel": 2, "user_id": 3}
    assert table.frame.iloc[0, 3] == "00123"
    result = prepare_table(table, mapping)
    assert result.profile == "revenue"
    assert result.frame.iloc[0]["user_id"] == "00123"
    exported = pd.read_csv(BytesIO(result.csv_content), dtype=str)
    assert exported.iloc[0]["user_id"] == "00123"
    assert exported.iloc[0]["备注"] == "首单"


def test_competing_aliases_not_guessed_and_mapping_collision_rejected():
    table = _table("日期,销售额,营业额\n2026-01-01,100,200\n")
    suggestions = suggest_mappings(table)
    assert suggestions == {"date": 0}
    with pytest.raises(DataValidationError, match="同一列"):
        prepare_table(table, {"date": 0, "revenue": 1, "channel": 1})
    with pytest.raises(DataValidationError, match="无效"):
        prepare_table(table, {"date": 0, "revenue": 99})


def test_duplicate_rows_require_opt_in_and_blank_rows_are_counted():
    table = _table("日期,销售额,渠道\n2026-01-01,100,门店\n,,\n2026-01-01,100,门店\n")
    mapping = suggest_mappings(table)
    quality = inspect_quality(table, mapping)
    assert quality.original_rows == 3
    assert quality.empty_rows == (3,)
    assert quality.duplicate_rows == (4,)
    kept = prepare_table(table, mapping)
    removed = prepare_table(table, mapping, remove_duplicates=True)
    assert len(kept.frame) == 2
    assert len(removed.frame) == 1
    assert build_partial_summary(kept.frame).total_revenue == 200
    assert build_partial_summary(removed.frame).total_revenue == 100


def test_nonempty_unmapped_column_is_not_an_empty_row():
    table = _table("日期,销售额,备注\n2026-01-01,100,\n,,这里有信息\n")
    mapping = suggest_mappings(table)
    assert inspect_quality(table, mapping).empty_rows == ()
    with pytest.raises(DataValidationError, match="空值"):
        prepare_table(table, mapping)


def test_quality_reports_physical_rows_and_rejects_ambiguous_date_and_bad_number():
    table = _table("报表,,\n日期,销售额,渠道\n03/04/2026,1x,门店\n2026-02-30,20,微信\n", header_row=1)
    mapping = suggest_mappings(table)
    quality = inspect_quality(table, mapping)
    assert quality.invalid_numeric["revenue"] == (3,)
    assert quality.ambiguous_dates == (3,)
    assert quality.invalid_dates == (4,)
    with pytest.raises(DataValidationError):
        prepare_table(table, mapping)


def test_confirmed_thousands_separator_and_negative_revenue():
    table = _table('日期,销售额,渠道\n2026-01-01,"1,200",门店\n2026-01-02,-20,微信\n')
    prepared = prepare_table(table, suggest_mappings(table))
    summary = build_partial_summary(prepared.frame)
    assert summary.total_revenue == 1180
    assert summary.daily_revenue["销售额"].tolist() == [1200, -20]
    assert dict(zip(summary.channel_revenue["渠道"], summary.channel_revenue["销售额"])) == {"门店": 1200, "微信": -20}


def test_order_profile_respects_duplicate_choice_and_keeps_user_ids():
    table = _table(
        "日期,用户编号,商品,单价,数量,性别,年级\n"
        "2026-01-01,00123,面包,5,2,女,大一\n"
        "2026-01-01,00123,面包,5,2,女,大一\n"
    )
    prepared = prepare_table(table, suggest_mappings(table), remove_duplicates=False)
    assert prepared.profile == "orders"
    kept, kept_report = process_order_csv(prepared.csv_content, remove_duplicates=False)
    removed, removed_report = process_order_csv(prepared.csv_content, remove_duplicates=True)
    assert kept["user_id"].tolist() == ["00123", "00123"]
    assert kept["sales"].sum() == 20
    assert removed["sales"].sum() == 10
    assert kept_report.duplicate_rows_removed == 0
    assert removed_report.duplicate_rows_removed == 1


def test_database_profile_and_dedup_are_part_of_source_identity():
    state = {}
    initialize_database_state(state)
    project = create_database(state, "测试", b"same data", profile="revenue", remove_duplicates=False)
    source_id = project.data_source_id
    project.chat_messages.append({"role": "user", "content": "old"})
    assert not replace_database_data(project, b"same data", profile="revenue", remove_duplicates=False)
    assert project.data_source_id == source_id and project.chat_messages
    assert replace_database_data(project, b"same data", profile="revenue", remove_duplicates=True)
    assert project.data_source_id != source_id and not project.chat_messages
    assert project.remove_duplicates is True and project.profile == "revenue"
    assert replace_database_data(project, b"same data", profile="orders", remove_duplicates=True)
    assert project.profile == "orders"


def test_partial_analysis_only_computes_valid_available_results():
    summary = build_partial_summary(pd.DataFrame({
        "date": ["not a date", "2026-01-02"],
        "revenue": [10, 20],
        "channel": ["门店", "微信"],
    }))
    assert summary.total_revenue == 30
    assert summary.daily_revenue is None
    assert summary.channel_revenue["销售额"].sum() == 30
    assert any("日期" in message for message in summary.issues)
    summary = build_partial_summary(pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "revenue": ["12", "not a number"],
        "channel": ["门店", "微信"],
    }))
    assert summary.total_revenue is None
    assert summary.daily_revenue is None
    assert summary.channel_revenue is None
    assert any("销售额" in message for message in summary.issues)


def test_partial_analysis_missing_channel_and_ambiguous_dates():
    summary = build_partial_summary(pd.DataFrame({"date": ["04/03/2026"], "revenue": [100]}))
    assert summary.total_revenue == 100
    assert summary.daily_revenue is None and summary.channel_revenue is None
    assert any("日期" in message for message in summary.issues)
    assert any("渠道" in message for message in summary.issues)


class _UploadedFile(BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.name = "input.csv"


def _mock_creation_upload(monkeypatch, data):
    uploaded = _UploadedFile(data.encode("utf-8"))

    def fake_uploader(label, *args, **kwargs):
        value = None if "替换" in label else uploaded
        st.session_state[kwargs["key"]] = value
        return value

    monkeypatch.setattr(st, "file_uploader", fake_uploader)


def _create_dialog():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    next(item for item in app.button if item.label == "新建数据库").click().run()
    next(item for item in app.text_input if item.label == "数据库名称").input("测试导入").run()
    return app


def _click_create(app):
    next(item for item in app.button if item.label == "创建数据库").click().run()


def _confirm_import(app):
    next(item for item in app.checkbox if item.label == "我已核对字段、清洗方式和金额口径").check().run()


def test_app_partial_import_requires_confirmation_and_displays_actual_aggregates(monkeypatch):
    _mock_creation_upload(monkeypatch, "日期,销售额,渠道\n2026-01-01,100,门店\n2026-01-02,200,微信\n")
    app = _create_dialog()
    assert not app.exception
    _click_create(app)
    assert not app.exception and len(app.session_state["databases"]) == 0
    _confirm_import(app)
    _click_create(app)
    assert not app.exception
    project = app.session_state["databases"][app.session_state["active_database_id"]]
    assert project.profile == "partial"
    assert any(metric.label == "销售额合计" and metric.value == "300.00" for metric in app.metric)
    assert any("销售额趋势" in item.value for item in app.markdown)
    assert any("渠道销售额" in item.value for item in app.markdown)
    rendered = [item.value for item in app.dataframe]
    assert any(list(frame.columns) == ["日期", "销售额"] and frame["销售额"].tolist() == [100.0, 200.0]
               for frame in rendered)
    assert any(list(frame.columns) == ["渠道", "销售额"]
               and dict(zip(frame["渠道"], frame["销售额"])) == {"门店": 100.0, "微信": 200.0}
               for frame in rendered)
    assert not app.chat_input


def test_app_standard_order_import_preserves_duplicate_rows_by_default(monkeypatch):
    _mock_creation_upload(
        monkeypatch,
        "日期,用户编号,商品,单价,数量,性别,年级\n"
        "2026-01-01,00123,面包,5,2,女,大一\n"
        "2026-01-01,00123,面包,5,2,女,大一\n",
    )
    app = _create_dialog()
    assert not app.exception
    _click_create(app)
    assert not app.exception and len(app.session_state["databases"]) == 0
    _confirm_import(app)
    _click_create(app)
    assert not app.exception
    project = app.session_state["databases"][app.session_state["active_database_id"]]
    assert project.profile == "orders" and project.remove_duplicates is False
    orders, _ = process_order_csv(project.csv_content, remove_duplicates=project.remove_duplicates)
    assert orders["user_id"].tolist() == ["00123", "00123"]
    assert orders["sales"].sum() == 20
    assert app.chat_input
