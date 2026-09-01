from ai_agent import generate_operation_summary


def generate_auto_insights(metrics):
    """返回适合看板展示的简短、数据驱动洞察。"""

    return (
        f"主力商品 {metrics.top_product} 贡献总销售额 {metrics.top_product_ratio:.1%}。",
        f"当前销售贡献最高的年级为 {metrics.top_grade}。",
        f"平均客单价为 ¥{metrics.avg_order_value:,.2f}，建议与订单量一起持续观察。",
    )


def generate_insight(total_sales, order_count, top_product, top_gender, top_grade):
    return generate_operation_summary(total_sales=total_sales, order_count=order_count, top_product=top_product, top_gender=top_gender, top_grade=top_grade)
