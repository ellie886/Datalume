---
name: business-diagnosis
description: >
  Diagnose changes or problems in business performance using
  available trend, order, product, channel and user data.
  Use when the user asks why business performance changed,
  what is driving a decline or anomaly, or where an operating
  problem may be coming from.
---

# Business Diagnosis Skill（经营诊断）

## Objective

识别经营表现变化，并通过趋势、订单、客单价、商品、渠道和用户等当前可用数据逐层定位主要驱动因素，形成有数据证据支持的诊断结论。

遵循：发现变化 → 拆解变化 → 提出假设 → 调用 Tool 验证 → 根据 Observation 接受、降低或排除假设 → 必要时继续调查 → 证据足够或数据不足时停止。

## 适用边界

适用于用户询问经营表现为何变化、下降或异常，以及要求定位经营问题来源的任务。仅问销售额、商品排名、渠道排名或画已有指标图时，不使用复杂诊断流程；能由一个 Tool 回答的问题应直接调用该 Tool。

## Step 1：确认现象

用户描述下降、变差或异常时，先调用 `analyze_trend` 验证现象是否真实存在，并确认当前周期、对比周期、变化方向和幅度。不得未经 Observation 验证就接受用户的判断。

## Step 2：第一层驱动拆解

对销售额变化优先使用：`Revenue ≈ Orders × Average Order Value`。

根据 `analyze_trend` 的 Observation 判断销售变化主要由订单量、客单价、两者共同驱动，还是数据不足。不要仅因销售额下降就随机分析全部维度。

## Step 3：选择最有价值的假设

- 订单量明显下降：根据数据能力和用户目标，优先从 `analyze_channel`、`analyze_product`、`analyze_user`、`analyze_repurchase` 中选择最能解释订单减少的一个方向。渠道、商品或复购并非必然原因，必须验证。
- 订单量基本稳定而客单价明显下降：优先用 `analyze_product` 检验高价商品占比下降、低价商品占比上升等商品结构假设。
- 用户明确指定商品、渠道、用户或复购维度时，覆盖其指定维度；不要额外探索无关方向。
- `analyze_profit` 只在盈利能力与问题相关且字段支持时使用。
- `create_visualization` 只在用户要求图表或图表能直接帮助解释已完成分析时使用。
- `generate_report` 不属于默认诊断流程；只有用户明确要求报告时使用。

## Hypothesis Testing

1. 根据已有 Observation 提出信息价值最高的下一步可验证假设。
2. 选择最相关且 DataCapabilities 支持的 Tool。
3. 使用 ToolResult 判断数据是否支持假设。
4. 支持时，将结果作为诊断证据，并准确限定证据范围。
5. 不支持时，不得强行得出该结论；降低或排除该假设优先级。
6. 若存在其他可验证解释且能显著增加信息，继续调查。
7. 若当前字段无法继续区分原因，停止并说明尚未验证的部分。

禁止“先猜原因，再寻找数据强行证明”。结构变化只说明同期关联时，不得写成已确认因果。

## 证据等级与 Final Answer

最终回答应区分：

- 已确认事实：ToolResult 直接给出的指标、变化和字段能力。
- 数据支持的判断：由一个或多个 Observation 支持的驱动因素判断。
- 尚未验证的可能性：当前数据不足以确认的解释，必须明确标为可能性或数据限制。

推荐回答顺序：核心结论 → 数据证据 → 原因判断 → 必要建议 → 数据限制。不要泄露隐藏推理过程。

## 数据不足与反证

如果销售额下降、订单稳定、客单价下降，但 `analyze_product` 没有发现足以支持商品结构变化的证据，不得声称“商品结构导致客单价下降”。应降低该假设优先级。

当可用字段缺少 `discount`、`coupon`、`promotion`、`actual_payment` 等进一步解释客单价的字段时，应停止并说明：可以确认变化主要与客单价下降有关，但当前商品数据未发现足够的结构性解释，现有字段无法进一步确认具体原因。

## 停止条件

满足任一条件即停止继续调用 Tool：

1. 已有足够证据回答用户问题。
2. 用户明确要求的维度已经覆盖。
3. DataCapabilities 不支持进一步验证。
4. 剩余假设依赖当前不存在的字段。
5. 继续调用 Tool 不会显著增加信息。
6. 已达到 Agent 的 `MAX_AGENT_STEPS`。

本 Skill 不是固定 Pipeline。允许根据 Observation 形成 `trend → channel`、`trend → product`、`trend → repurchase → user` 等不同路径；不得机械执行全部工具。
