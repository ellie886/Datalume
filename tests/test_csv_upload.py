import pytest

from app import sync_chat_history_with_data_source
from datalume.data_processor import DataValidationError, calculate_metrics, process_order_csv


BASE = """date,user_id,product,price,quantity,gender,grade
2026-01-01,u1,A,10,2,女,大一
2026-01-02,u2,B,14,3,男,大二
"""


def test_default_csv_loads(default_bundle):
    orders, quality, _, metrics, _ = default_bundle
    assert len(orders) == 14
    assert quality.cleaned_rows == 14
    assert metrics.total_sales == 464


def test_valid_uploaded_csv():
    orders, report = process_order_csv(BASE.encode())
    assert calculate_metrics(orders).total_sales == 62
    assert report.cleaned_rows == 2


def test_missing_fields_are_named():
    with pytest.raises(DataValidationError, match="gender、grade"):
        process_order_csv(b"date,user_id,product,price,quantity\n2026-01-01,u1,A,10,1\n")


@pytest.mark.parametrize(("column", "value", "expected"), [("price", "abc", "price"), ("quantity", "x", "quantity")])
def test_invalid_numeric_fields(column, value, expected):
    content = BASE.replace("10", value, 1) if column == "price" else BASE.replace(",2,女", f",{value},女", 1)
    with pytest.raises(DataValidationError, match=expected):
        process_order_csv(content.encode())


def test_invalid_date():
    with pytest.raises(DataValidationError, match="date"):
        process_order_csv(BASE.replace("2026-01-01", "bad-date").encode())


def test_partial_empty_data_rejected():
    with pytest.raises(DataValidationError, match="空数据"):
        process_order_csv(BASE.replace("u1,A", ",A").encode())


def test_duplicate_rows_cleaned():
    duplicate = BASE + "2026-01-02,u2,B,14,3,男,大二\n"
    orders, report = process_order_csv(duplicate.encode())
    assert len(orders) == 2
    assert report.duplicate_rows_removed == 1


def test_data_switch_recalculates_metrics(default_bundle):
    _, _, _, old_metrics, _ = default_bundle
    new_orders, _ = process_order_csv(BASE.encode())
    assert old_metrics.total_sales != calculate_metrics(new_orders).total_sales


def test_legacy_state_cleared_on_source_change():
    state = {"chat_data_source_id": "old", "chat_messages": [1], "diagnostic_report": object()}
    assert sync_chat_history_with_data_source(state, "new") is True
    assert state["chat_messages"] == []
    assert state["diagnostic_report"] is None
    assert sync_chat_history_with_data_source(state, "new") is False
