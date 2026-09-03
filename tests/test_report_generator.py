from datalume.data_processor import calculate_metrics, process_order_csv, summarize_orders
from datalume.report_generator import generate_diagnostic_report


def test_default_report_generated(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert report.total_sales == 464
    assert report.top_product == "原味欧包"
    assert "经营概览" in report.to_markdown()


def test_report_metrics_change_with_upload(rich_csv):
    orders, _ = process_order_csv(rich_csv)
    metrics = calculate_metrics(orders)
    report = generate_diagnostic_report(orders, metrics, summarize_orders(orders))
    assert report.total_sales == metrics.total_sales
    assert report.total_sales != 464


def test_product_contribution_correct(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert report.top_product_ratio == 240 / 464
    assert report.product_performance[0].contribution == 240 / 464


def test_peak_and_low_dates_correct(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert report.peak_sales == metrics.daily_sales.max()
    assert report.low_sales == metrics.daily_sales.min()
    assert report.peak_date == metrics.daily_sales.idxmax()


def test_small_sample_warning(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert report.confidence == "较低"
    assert any("样本只有 14" in item for item in report.limitations)


def test_repurchase_not_invented(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert not report.can_measure_repurchase
    assert "无法准确" in report.repurchase_assessment


def test_download_markdown_has_content(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    content = generate_diagnostic_report(orders, metrics, overview, caps).to_markdown()
    assert content.startswith("# Datalume AI 经营诊断报告")
    assert "行动建议" in content
    assert len(content) > 500


def test_report_action_format(default_bundle):
    orders, _, overview, metrics, caps = default_bundle
    report = generate_diagnostic_report(orders, metrics, overview, caps)
    assert 3 <= len(report.recommendations) <= 5
    assert all(item.action and item.reason and item.metrics_to_watch for item in report.recommendations)
