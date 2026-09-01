from ai_agent import generate_operation_summary


def generate_strategy(total_sales, order_count, top_product, top_gender, top_grade, avg_order_value, top_product_ratio):
    return generate_operation_summary(total_sales=total_sales, order_count=order_count, top_product=top_product, top_gender=top_gender, top_grade=top_grade, avg_order_value=avg_order_value, top_product_ratio=top_product_ratio)
