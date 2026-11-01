---
name: clarify-analysis-request
description: >
  Clarify materially ambiguous data-analysis requests before running analysis.
  Use when the requested object, metric, comparison, time range, or referent
  cannot be inferred from the current question and conversation context.
---

# Clarify Analysis Request

## Objective

在执行数据分析前，仅对会显著改变分析路径或结论的关键歧义进行一次简短澄清，随后恢复原任务。

## Decision Policy

按以下优先级处理：

1. 如果当前问题可由对话上下文可靠补全，直接继续分析，不追问。
2. 如果一个现有分析 Tool 可以同时返回相关核心指标，通常直接综合分析，不要求用户先选口径；用户明确要做单一取舍、不同口径会改变选择时，先确认一个关键标准。
3. 如果当前数据缺少必要字段，继续调用对应 Tool，让系统说明数据限制；不要把字段缺失伪装成用户需求不清。
4. 只有当分析对象、指标、时间范围、比较基准或指代对象存在多个合理解释，且不同解释会显著改变任务时，才向用户澄清。

## Clarification Style

- 一次只问最能减少歧义的一个问题。
- 使用简洁自然的中文，不解释内部路由、Skill 或 Tool。
- 合适时提供不超过三个互斥选项；能给出业务上常用的默认项时标注“推荐”。
- 不要求用户重复已经在当前问题、历史对话或数据字段中明确的信息。
- 不连续追问非关键细节，不把澄清变成问卷。

## Resume

用户补充信息后，将原始问题与补充内容合并为完整目标并继续原有分析流程。若用户改为提出一个清晰的新问题，以当前新问题为准并放弃未完成的旧澄清。

## Boundaries

本 Skill 只负责消除关键歧义，不计算指标、不生成业务结论，也不替代现有分析 Tool。任何数字仍必须来自白名单 Tool 的 Observation。
