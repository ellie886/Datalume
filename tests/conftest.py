from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from datalume.data_processor import calculate_metrics, detect_data_capabilities, process_order_csv, summarize_orders


@pytest.fixture(autouse=True)
def disable_live_deepseek(monkeypatch):
    """自动化测试永不读取或消耗真实 DeepSeek API Key。"""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")


@pytest.fixture
def default_bundle():
    orders, quality = process_order_csv("data/orders.csv")
    return orders, quality, summarize_orders(orders), calculate_metrics(orders), detect_data_capabilities(orders)


def csv_bytes(rows: str) -> bytes:
    return rows.strip().encode("utf-8")


@pytest.fixture
def rich_csv() -> bytes:
    lines = ["order_id,date,user_id,product,price,quantity,gender,grade,unit_cost,channel,acquisition_cost"]
    for day in range(1, 29):
        user = f"u{(day - 1) % 12 + 1}"
        product = "欧包" if day % 2 else "酸奶"
        price = 20 if product == "欧包" else 12
        cost = 8 if product == "欧包" else 7
        channel = "社群" if day % 3 else "海报"
        acquisition = 2 if channel == "社群" else 4
        lines.append(f"o{day},2026-08-{day:02d},{user},{product},{price},1,女,大一,{cost},{channel},{acquisition}")
    return ("\n".join(lines) + "\n").encode("utf-8")
