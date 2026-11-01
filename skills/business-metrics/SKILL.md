---
name: business-metrics
description: >
  Resolve business metric definitions and ambiguous performance criteria such as
  best product or channel, high-value users, repurchase and revenue versus profit.
  Use when choosing or explaining the measurement basis changes the answer;
  ordinary queries with an explicit metric do not need this skill.
---

# Business Metrics（经营指标口径）

把经营表达转换成当前工具能够计算的指标，回答时说明衡量标准。数值只引用白名单工具 Observation；本 Skill 不计算指标，也不增加筛选或分析能力。

## 先确定该算什么

- 当前明确口径优先，其次使用相关上下文中的口径；不能继承与当前问题无关的旧指标。
- “卖得最多”按销量，“销售额最高”按销售额，“最赚钱”优先看利润额；利润额与毛利率分别回答。
- “表现最好”没有唯一标准。普通了解表现时，可调用一次相关工具，简短比较已返回的指标并说明采用的标准，不必让用户先选指标。
- 当用户要据此做单一取舍（例如只保留一个商品、只给一个渠道预算），且量、额、利润等标准会改变选择，使用已有澄清流程只确认一个关键标准；不擅自给指标加权。
- 数据缺少利润或渠道字段时，调用对应工具说明限制；不要追问一个无法计算的指标，也不要用销售额替代利润。
- 用户只问定义时，简短解释该定义与工具实际口径，不扩展成全部经营诊断。复杂变化原因交给 business-diagnosis。

## 与当前工具一致的口径

| 经营表达 | 工具与当前口径 |
| --- | --- |
| 销量 / 销售额 | analyze_product：quantity 为销售件数合计；revenue 为 price × quantity 合计，未扣退款、优惠或税费，不称净收入。销量排名与销售额排名分开。 |
| 订单量 / 客单价 | 完整订单号时去重订单计数，否则按行估算；客单价是销售额 / 订单量，不是每用户消费额。 |
| 利润 / 毛利率 | analyze_profit：优先使用每行 profit，其次 unit_cost × quantity，否则 cost/total_cost 按每行总成本；毛利率为商品汇总利润 / 汇总收入，不平均行毛利率。未包含全部经营费用时不称净利润。 |
| 高价值用户 | analyze_user 当前按累计消费额识别用户，年级按总销售贡献；不是用户利润、人均消费或预测 LTV。频次与最近活跃证据使用 analyze_repurchase，不能冒充完整 RFM 评分。 |
| 复购 | analyze_repurchase：购买次数超过一次的成交用户 / 全部成交用户；订单号不完整时只能按行估算。不是留存率或新客 cohort 复购率。最近购买距数据最大日期计算，不是距今天。 |
| 渠道表现 | analyze_channel：分别看成交用户、订单、销售额、客单价和渠道内重复购买占比。默认第一名按销售额；成交用户数不是新客数或转化率。 |
| CAC / ROI | 有成本字段才引用工具结果；当前 CAC 分母是渠道成交用户，ROI 未扣商品成本，不称严格新客 CAC 或利润 ROI。 |
| 最近变化 | analyze_trend：以数据最大日期为终点，最近七天对比前七天；不是自然周或系统当天。数据不足两个周期时保留限制；变化率与百分点分开。 |

## 执行与证据边界

- Tool 参数被接受不代表计算已经应用。当前 entity/channel/top_n 不实际过滤；trend 的 time_range 不改变固定周期，user/repurchase/channel 的 metric 不改变原始排名。
- 用户指定了当前工具不支持的日期或人群筛选时，说明无法按该范围计算；不得悄悄返回全量结果当作已筛选结果，也不反复追问更多无效参数。
- 优先依据 Observation 的结构化 metrics 与证据回答。商品 evidence 仍按销售额排序；利润 conclusion 按利润额、渠道 conclusion 按销售额，不能当作用户请求的其他指标第一名。
- 只能引用已返回的数据；紧凑 Observation 可能截断。可以比较已返回数值并指出其中的领先项，但比较商品毛利率或渠道复购表现时限定为“当前返回结果中”，未验证覆盖全部对象前不宣称全局最优。不新增计算比例或分数，不补造缺失对象。
- 同一分析工具一次返回多个指标，无需为每个 metric 重复调用。图表采用同一口径；现有图表不支持该口径时说明限制。
- 末尾只保留影响结论的口径说明与限制。避免把“最好”扩写成未经验证的因果判断或大段指标教学。
