"""Datalume 意图识别：当前问题优先，历史仅处理省略追问。"""

from __future__ import annotations

from enum import Enum


class QuestionType(str, Enum):
    PRODUCT = "product"
    USER = "user"
    REPURCHASE = "repurchase"
    PROFIT = "profit"
    CHANNEL = "channel"
    TREND = "trend"
    STRATEGY = "strategy"
    SALES_STRATEGY = "strategy"


QUESTION_TYPE_LABELS = {
    QuestionType.PRODUCT: "商品分析", QuestionType.USER: "用户分析",
    QuestionType.REPURCHASE: "复购分析", QuestionType.PROFIT: "利润分析",
    QuestionType.CHANNEL: "渠道分析", QuestionType.TREND: "趋势分析",
    QuestionType.STRATEGY: "销售策略",
}
KEYWORDS = {
    QuestionType.PROFIT: ("利润", "毛利", "成本", "盈利", "赚钱", "亏损", "毛利率", "profit", "margin"),
    QuestionType.CHANNEL: ("渠道", "来源", "获客", "投放", "广告", "cac", "roi", "roas", "channel", "source"),
    QuestionType.REPURCHASE: ("复购", "回购", "重复购买", "购买次数", "再次购买", "留存", "rfm"),
    QuestionType.TREND: ("最近一周", "近一周", "本周", "上周", "最近7天", "近7天", "环比", "同比", "趋势", "下降", "下滑", "上涨", "增长", "变化", "波动", "高峰", "低谷"),
    QuestionType.PRODUCT: ("商品", "产品", "单品", "品类", "销量", "销售额排名", "爆款", "库存", "热门", "卖得"),
    QuestionType.USER: ("用户", "客户", "学生", "人群", "画像", "性别", "年级", "会员", "高价值"),
    QuestionType.STRATEGY: ("策略", "建议", "活动", "优惠", "推广", "营销", "经营", "提升销售", "拉新", "转化", "社群", "怎么办", "怎么提升"),
}
PRIORITY = (QuestionType.PROFIT, QuestionType.CHANNEL, QuestionType.REPURCHASE, QuestionType.TREND, QuestionType.PRODUCT, QuestionType.USER, QuestionType.STRATEGY)


def matched_intents(question: str) -> tuple[QuestionType, ...]:
    normalized = (question or "").strip().lower()
    return tuple(intent for intent in PRIORITY if any(keyword in normalized for keyword in KEYWORDS[intent]))


def has_explicit_intent(question: str) -> bool:
    return bool(matched_intents(question))


def classify_question(question: str) -> QuestionType:
    return matched_intents(question)[0] if matched_intents(question) else QuestionType.STRATEGY


def get_question_type_label(question_type: QuestionType) -> str:
    return QUESTION_TYPE_LABELS[question_type]
