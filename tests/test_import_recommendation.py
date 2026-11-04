"""Safety and behavior tests for AI import recommendations; no live model calls."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from datalume.data_preparation import DataValidationError, prepare_table, read_table
from datalume.import_recommendation import (
    ImportRecommendationError,
    local_recommendation,
    recommend_import,
)


def _table(text, filename="input.csv"):
    return read_table(text.encode("utf-8"), filename=filename, header_row=0)


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _completion(self, messages, *, json_output=False, max_tokens=700):
        self.calls.append({"messages": messages, "json_output": json_output, "max_tokens": max_tokens})
        return self.result if isinstance(self.result, str) else json.dumps(self.result, ensure_ascii=False)


def _response(**overrides):
    result = {
        "description": "销售数据示例",
        "structure": "single_table",
        "header_row": 1,
        "mapping": {"date": 0, "revenue": 1},
        "concerns": ["请核对金额单位"],
    }
    result.update(overrides)
    return result


def test_valid_recommendation_is_only_a_suggestion_and_samples_are_bounded():
    header = ",".join(f"列{i}" for i in range(16))
    rows = [",".join([f"sample_row_{i:02d}", "X" * 100] + [f"v{i}_{j}" for j in range(2, 16)]) for i in range(1, 25)]
    table = _table(header + "\n" + "\n".join(rows) + "\n", filename="PRIVATE_FILENAME.csv")
    client = FakeClient(_response())
    suggestion = recommend_import(table, client)
    assert len(client.calls) == 1 and client.calls[0]["json_output"] is True
    payload = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
    assert "PRIVATE_FILENAME" not in payload
    assert "sample_row_01" in payload
    assert "sample_row_16" not in payload
    assert "列12" not in payload
    assert "X" * 81 not in payload
    assert suggestion.header_row == 0 and suggestion.mapping == {"date": 0, "revenue": 1}
    assert suggestion.source == "ai"
    # A recommendation is advisory and does not alter the table.
    assert len(table.frame) == 24 and table.frame.iloc[0, 1] == "X" * 100


@pytest.mark.parametrize("payload", [
    "not json",
    "[]",
    _response(header_row=0),
    _response(header_row=20),
    _response(header_row=True),
    _response(mapping={"date": 0, "revenue": 0}),
    _response(mapping={"date": 0, "revenue": 99}),
    _response(mapping={"date": 0, "revenue": True}),
    _response(mapping={"date": 0, "arbitrary_field": 1}),
    _response(concerns="不是列表"),
])
def test_rejects_malformed_or_conflicting_model_output(payload):
    table = _table("日期,销售额\n2026-01-01,100\n2026-01-02,200\n")
    with pytest.raises(ImportRecommendationError):
        recommend_import(table, FakeClient(payload))


def test_salary_sections_are_not_silently_treated_as_revenue():
    table = _table(
        "项目,2024,2023\n人均薪酬（万元）,35,32\n员工总数（人）,600,550\n\n"
        "人员结构,2024,2023\n研发人员,200,180\n"
    )
    suggestion = local_recommendation(table)
    assert "revenue" not in suggestion.mapping
    with pytest.raises(DataValidationError):
        prepare_table(table, suggestion.mapping)


class _UploadedFile(BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.name = "input.csv"


def _app_with_upload(monkeypatch, content):
    uploaded = _UploadedFile(content.encode("utf-8"))
    def fake_upload(label, *args, **kwargs):
        value = None if "替换" in label else uploaded
        st.session_state[kwargs["key"]] = value
        return value
    monkeypatch.setattr(st, "file_uploader", fake_upload)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    next(item for item in app.button if item.label == "新建数据库").click().run()
    next(item for item in app.text_input if item.label == "数据库名称").input("测试").run()
    return app


def test_external_ai_requires_explicit_consent_and_button(monkeypatch):
    from datalume.deepseek_client import DeepSeekClient
    calls = []
    def fake_completion(self, messages, **kwargs):
        calls.append(messages)
        return json.dumps(_response(), ensure_ascii=False)
    monkeypatch.setattr(DeepSeekClient, "_completion", fake_completion)
    app = _app_with_upload(monkeypatch, "日期,销售额\n2026-01-01,100\n2026-01-02,200\n")
    assert not app.exception
    assert not calls
    # The user can see what data is transmitted before choosing to send it.
    text = " ".join(str(item.value) for item in app.caption) + " " + " ".join(str(item.value) for item in app.info)
    assert "15" in text and ("发送" in text or "传" in text)
    button = next(item for item in app.button if item.label == "生成 AI 推荐方案")
    assert button.disabled or not calls
    consent = next(item for item in app.checkbox if item.label == "同意发送表格样本给 AI")
    consent.check().run()
    assert not calls  # Consent alone is not an external request.
    next(item for item in app.button if item.label == "生成 AI 推荐方案").click().run()
    assert not app.exception
    assert len(calls) == 1


def test_changing_mapping_invalidates_prior_confirmation(monkeypatch):
    app = _app_with_upload(monkeypatch, "日期,销售额,渠道\n2026-01-01,100,门店\n2026-01-02,200,微信\n")
    confirm = next(item for item in app.checkbox if item.label == "我已核对字段、清洗方式和金额口径")
    confirm.check().run()
    channel = next(item for item in app.selectbox if item.label == "渠道")
    channel.select(-1).run()
    confirm = next(item for item in app.checkbox if item.label == "我已核对字段、清洗方式和金额口径")
    assert confirm.value is False
    next(item for item in app.button if item.label == "创建数据库").click().run()
    assert not app.exception and len(app.session_state["databases"]) == 0


def test_apply_ai_suggestion_requires_a_fresh_import_confirmation(monkeypatch):
    from datalume.deepseek_client import DeepSeekClient
    calls = []
    def fake_completion(self, messages, **kwargs):
        calls.append(messages)
        return json.dumps(_response(mapping={"date": 0, "revenue": 2}), ensure_ascii=False)
    monkeypatch.setattr(DeepSeekClient, "_completion", fake_completion)
    app = _app_with_upload(monkeypatch, "日期,销售额,收入\n2026-01-01,100,80\n2026-01-02,200,150\n")
    next(item for item in app.checkbox if item.label == "我已核对字段、清洗方式和金额口径").check().run()
    next(item for item in app.checkbox if item.label == "同意发送表格样本给 AI").check().run()
    next(item for item in app.button if item.label == "生成 AI 推荐方案").click().run()
    assert len(calls) == 1 and not app.exception
    # Merely receiving the suggestion must not silently alter the accepted mapping.
    assert next(item for item in app.selectbox if item.label == "销售额").value != 2
    next(item for item in app.button if item.label == "应用 AI 推荐").click().run()
    assert not app.exception
    assert next(item for item in app.selectbox if item.label == "销售额").value == 2
    assert next(item for item in app.checkbox if item.label == "我已核对字段、清洗方式和金额口径").value is False
    next(item for item in app.button if item.label == "创建数据库").click().run()
    assert not app.exception and len(app.session_state["databases"]) == 0


@pytest.mark.parametrize("column_name", ["人均薪酬（万元）", "员工人数", "payroll"])
def test_ai_cannot_relabel_explicit_salary_or_headcount_as_revenue(column_name):
    table = _table(f"日期,{column_name}\n2026-01-01,100\n")
    with pytest.raises(ImportRecommendationError):
        recommend_import(table, FakeClient(_response()))
