# Prism（棱镜）开发与能力说明

本指南补充首页的使用介绍，说明当前代码怎样执行分析、各模块的职责及数据口径。Prism 面向符合订单字段约定的经营数据；当前实现是本地 Streamlit 应用，尚未覆盖通用企业数据建模、持久化或生产环境权限管理。

GitHub 仓库名为 `prism`。当前应用内显示名称、环境变量和内部 Python 包仍沿用 Datalume / `datalume` 命名；下列源码路径与启动命令保持兼容。

## 运行与体验

在项目根目录安装运行依赖并启动：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell 的虚拟环境激活命令为 `.venv\Scripts\Activate.ps1`。依赖使用版本区间，当前没有锁文件；排查安装问题时应同时记录 Python、Streamlit、Pandas 和 Altair 版本。

应用初始显示空工作区。点击“新建数据库”，上传 [data/demo_orders.csv](../data/demo_orders.csv)，即可体验商品、用户、渠道、复购、利润和趋势等分析。这个示例包含扩展分析需要的订单号、成本与渠道字段；[data/orders.csv](../data/orders.csv) 是较小的回归测试样例。

可尝试的问题：

```text
哪个商品销量最高？帮我画出来。
最近销售为什么下降？
哪个渠道表现最好？
复购情况怎么样？
哪个商品利润最高？
帮我生成经营诊断报告。
```

### 基础分析与模型模式

未配置可用的 DeepSeek 客户端时，应用仍能运行，通过确定性意图识别和 Pandas 分析生成基础回答。它与模型驱动的多步诊断不是同一种运行模式。

启用模型模式时，复制 [环境变量示例](../.env.example)，在本机编辑 `.env`：

```bash
cp .env.example .env
```

| 配置 | 当前默认值或作用 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 默认为空；填写自己的 Key 才能调用服务 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash`，可按账号可用模型修改 |
| `DEEPSEEK_TIMEOUT` | 单次请求超时，默认 20 秒 |
| `DATALUME_AGENT_ENABLED` | 默认 `true`；关闭时使用原有单步分析入口 |
| `DATALUME_AGENT_MAX_STEPS` | 默认 4，限制在 1–8 次工具执行 |

默认模型名是代码配置，不代表对服务端模型可用性的验证。多步 Agent、请求规划和自然语言生成依赖网络及可用 API；关闭多步 Agent 并不自动关闭单步模型调用。希望只运行确定性分析时，可显式传入空 Key：

```bash
DEEPSEEK_API_KEY= python -m streamlit run app.py
```

模型调用会发送用户问题、相关上下文、数据能力和紧凑分析结果，其中可能包含商品、渠道、用户标识及业务数字；当前接口不直接发送完整 CSV 或 DataFrame。使用自己的数据前，应了解所配置模型服务的数据处理方式。真实 Key 只放在环境变量或本地 `.env`，不要写入 README、截图或提交内容。

## Agent、Skill 与 Tool

三个层次分别承担不同职责：

| 层次 | 当前职责 | 代码或文件入口 |
| --- | --- | --- |
| Agent | 解析请求、澄清关键歧义、选择业务 Skill 和工具、依据 Observation 决定继续或结束 | [agent_runtime.py](../datalume/agent_runtime.py) |
| Skill | 提供经营分析方法、指标口径与请求澄清规范，不新增计算能力 | [skills/](../skills/) |
| Tool | 验证参数并执行预定义分析、选择图表或生成报告 | [agent_tools.py](../datalume/agent_tools.py) |
| 分析引擎 | 通过 Python / Pandas 清洗、分组和计算实际指标 | [data_processor.py](../datalume/data_processor.py)、[analysis_engine.py](../datalume/analysis_engine.py) |

### 一次分析怎样执行

1. 上传内容经过字段校验与清洗，计算基础指标，并形成 `DataCapabilities`。
2. 请求规划结合用户问题、有限上下文、可用字段、工具摘要和 Skill 规则，决定是否需要一次关键澄清，或直接继续。
3. 若需要澄清，保存原始问题与补充信息；用户回答后恢复原请求。用户提出新的明确问题时，放弃旧澄清。
4. 主分析循环注入选中的业务 Skill；模型只能请求白名单工具。
5. 工具返回包含 `summary`、`metrics`、`evidence`、`limitations` 等字段的紧凑 Observation。
6. 模型依据结果选择下一步，或给出回答；显式要求图表、报告时调用对应工具。

例如“最近销售为什么下降？”可以先查看趋势，区分订单量与客单价变化，再选择商品或渠道方向验证。工具路径由模型和已有 Observation 决定，没有固定要求依次跑完所有维度。当前工具返回的颗粒度有限，方法论上的“提出假设”不代表每个假设都已有对应数据或验证工具。

### 工具白名单

当前注册 9 个工具：

| 工具 | 作用 | 关键条件与边界 |
| --- | --- | --- |
| `analyze_product` | 计算商品销量、销售额、贡献率及排名 | 销量与销售额分别表达；销售额证据仍按销售额排序 |
| `analyze_user` | 查看用户结构、消费贡献及高消费用户 | “高价值”采用累计消费或群体总销售贡献，不是预测 LTV |
| `analyze_repurchase` | 统计购买频次、重复购买用户、人群差异及基础 RFM 证据 | 完整订单号才可区分同订单多行；不是完整 RFM 评分模型 |
| `analyze_profit` | 按可用成本或利润字段汇总商品利润与毛利率 | 缺少可用字段时返回限制，不以销售额代替利润 |
| `analyze_channel` | 查看渠道成交用户、订单、销售额、客单价与重复购买占比 | 需要渠道字段；成本指标另需可用获客成本字段 |
| `analyze_trend` | 比较最近 7 天与前 7 天，拆解订单、客单价等变化 | 固定以数据最大日期为终点，不是任意日期比较 |
| `analyze_strategy` | 基于已计算的基础指标给出策略建议 | 建议本身不是已验证的业务效果 |
| `create_visualization` | 为已完成分析选择内置 Altair 图表 | 不执行模型生成的 Python 或图表代码 |
| `generate_report` | 生成可下载的确定性 Markdown 经营诊断报告 | 按明确报告请求调用，能力仍受字段与内置计算限制 |

Registry 会拒绝未知工具，并检查参数类型、枚举及允许字段。它没有向模型提供通用 Shell、任意 Python、文件删除或 HTTP 请求工具。

### 三个 Skill

| Skill | 作用 | 入口 |
| --- | --- | --- |
| `business-diagnosis` | 确认现象、拆解指标、提出假设、用工具证据支持或排除假设，判断何时停止 | [经营诊断](../skills/business-diagnosis/SKILL.md) |
| `business-metrics` | 区分销量、销售额、利润、毛利率等口径，说明分母、时间范围和工具限制 | [指标口径](../skills/business-metrics/SKILL.md) |
| `clarify-analysis-request` | 仅澄清会改变分析路径或结论的关键歧义，补充后恢复原问题 | [请求澄清](../skills/clarify-analysis-request/SKILL.md) |

普通“表现最好”问题可以比较工具一次返回的已有指标，不必一律先问卷式澄清。若用户需要据此做单一取舍，而销量、销售额或利润会改变选择，再确认关键标准；缺少无法计算的字段时，优先说明数据限制。

### Skill 发现与加载

[SkillLoader](../datalume/skill_loader.py) 从 `skills/*/SKILL.md` 发现 Skill：

- `list_metadata()` 只读取 YAML front matter，暴露 `name` 和 `description`。
- `load(name)` 读取指定正文，校验名称、metadata 和正文。
- 请求规划阶段会读取 `clarify-analysis-request` 与 `business-metrics` 的短策略；歧义判断、口径规则与业务 Skill 选择共用一次模型规划请求。
- 主分析循环每轮最多注入一个选中的业务 Skill：经营诊断或指标口径；明确的简单指标查询可以不加载业务 Skill。

因此，“发现 metadata 后再选业务 Skill”适用于业务指令选择，但不能概括成所有正文都只在选中后才读取。增加 Skill 也不自动增加工具能力；当前没有任意技能脚本执行器或完整的多 Skill 编排系统。

## 停止、回退与证据边界

Agent 默认最多执行 4 次工具动作，可通过环境变量设为 1–8。一次模型返回多个工具请求时，每次工具执行都会消耗步数；这不是“最多 4 次模型请求”。已成功执行的工具会从本轮后续可用工具中移除。

系统 Prompt 要求在覆盖用户明确目标后结束，不扩展无关维度；也要求业务数字来自 Observation、保留字段限制、不将同期变化直接解释为因果。代码会拒绝模型未调用任何数据工具就尝试结束的情况；若没有可支持的工具结果或达到上限，会用已有结果形成尽力回答。

这些约束减少了编造机会，但当前没有逐条结论或逐个数字的证据校验器。模型最终文本在存在支持结果时仍会被接纳，不能据此声称完全消除幻觉。重要分析应回看公开工具轨迹、计算口径和数据限制。

回退路径如下：

1. 请求规划失败时，兼容原有 Skill 选择流程；模型 Skill 选择不可用时使用保守的确定性选择。
2. Skill 未选中或加载失败时，继续不带业务 Skill 的基础 Agent。
3. 多步 Tool Calling 失败时，回退到原有单步 LLM 分析。
4. 单步模型解析或生成失败、未配置可用模型时，使用确定性意图识别与 Pandas 结果。

公开轨迹保留工具、执行状态、耗时、停止原因、回退层级和业务 Skill 使用情况，便于查看发生了什么；它不是隐藏思维链。模型客户端的当前默认超时为 20 秒、SDK 重试次数为 1；尚无任务级总耗时或费用预算。

## 数据导入与字段能力

### 必填字段

CSV 需使用 UTF-8；CSV 表头或选中 Excel 工作表首行需要包含下列字段名：

| 字段 | 含义与约束 |
| --- | --- |
| `user_id` | 用户标识；文本标识和前导零应保持稳定 |
| `product` | 商品名称 |
| `price` | 非负单价 |
| `quantity` | 大于 0 的整数件数 |
| `date` | 可解析的日期或时间 |
| `gender` | 用户性别标签，当前属于必填字段 |
| `grade` | 用户年级或对应分组标签，当前属于必填字段 |

当前不是任意 Excel 的自动字段映射工具。缺少字段、核心字段为空、非法日期或非法数值会产生错误提示；完全空白行与完全重复行会被移除，并计入数据质量报告。重复行删除是当前实现的清洗规则，不保证它们都是业务重复订单：没有稳定订单明细标识时，两次内容相同的真实交易也可能被合并。

Excel 支持 `.xlsx` 和 `.xls`，每次选择一张工作表；不支持跨表 Join、自动整理合并单元格或多行表头，不重算公式。公式单元格依赖文件中的已有缓存值，没有有效缓存时可能按缺失数据处理。

### 可选字段与 DataCapabilities

能力检测依赖字段完整性与数值可用性，不仅检查列是否存在。支持的别名以 [data_processor.py](../datalume/data_processor.py) 为准。

| 可选字段 | 启用的能力 |
| --- | --- |
| 完整 `order_id` | 去重订单计数、按用户统计独立订单次数 |
| `profit`、`unit_cost` 或 `total_cost` / `cost` | 商品利润与毛利率 |
| 完整 `channel` / `source` / `acquisition_channel` | 渠道聚合分析 |
| 渠道字段及可用 `acquisition_cost` | 当前实现的渠道成本指标 |

缺少成本时不能判断利润；缺少渠道时不能给渠道排名；订单号不完整时，订单和重复购买次数退化为“每行一笔订单”估算。能力标记不验证订单号的业务唯一性，也不替代业务人员确认数据粒度。

## 当前计算口径

以下描述实际实现，不等同于通用于所有企业的会计或经营指标标准。

| 指标 | 当前计算方式 | 解释限制 |
| --- | --- | --- |
| 销售额 | `sum(price × quantity)` | 未扣退款、优惠或税，不自动等于净收入 |
| 销量 | `sum(quantity)` | 仅支持正整数数量，不支持退货冲减 |
| 订单量 | 完整订单号时 `nunique(order_id)`，否则数据行数 | 一单多行时必须提供稳定订单号 |
| 客单价 | 销售额 / 订单量 | 不是每用户消费额；按行估算订单时口径也会受影响 |
| 商品贡献率 | 商品销售额 / 总销售额 | 描述销售贡献，不等于利润贡献或因果解释 |
| 利润 | 优先使用每行 `profit`；其次销售额减 `unit_cost × quantity`；再其次减每行 `total_cost` / `cost` | `cost` 按每行总成本处理；未含全部费用时不称净利润 |
| 商品毛利率 | 商品汇总利润 / 商品汇总销售额 | 不平均行毛利率；仍取决于上传成本字段的定义 |
| 重复购买用户占比 | 购买次数大于 1 的成交用户 / 全部成交用户 | 非新客 cohort 复购率，非留存率；无订单号时只能按行估算 |
| 基础 RFM 证据 | 最近购买、频次、累计消费额 | 最近购买距数据最大日期计算，不是距今天；没有完整评分与分层模型 |
| 当前“CAC” | 渠道获客成本合计 / 渠道成交用户数 | 分母不是新客，不能当作严格新客获客成本 |
| 当前“ROI” | `(渠道销售额 − 渠道获客成本合计) / 渠道获客成本合计` | 未扣商品成本，不是利润 ROI；成本若重复到订单行会被重复累加 |
| 最近变化 | 以数据最大日期为终点，最近 7 天对前 7 天 | 不是自然周、系统当天或任意日期范围；变化率不等于百分点变化 |

当前 `daily_sales` 按输入的 `date` 原值聚合。如果同一天包含不同时间戳，会产生多个点；希望按天看趋势的数据应先统一日期粒度。趋势工具比较窗口时会归一到日期，但稀疏日期或空比较窗口等边界仍需单独核验。

### 尚未实现的分析范围

`entity`、`channel`、`top_n` 和 `time_range` 出现在部分工具参数中，但参数被接受不代表计算已应用。当前分析执行器直接把全量数据交给分析引擎：

- 指定商品、渠道或人群不会自动过滤订单；不能把全量结果当作指定范围的结果。
- 趋势仍固定最近 7 天与前 7 天；`time_range` 不改变周期。
- 用户、复购、渠道的部分 `metric` 不改变原始排序；渠道默认第一名按销售额，利润结论按利润额。
- 紧凑 Observation 会截断部分证据和列表。比较已返回结果可以说明“这些结果中”的领先项，但不能在未覆盖全部对象时声称全局最优。
- 缺少促销、优惠券、访问量或转化等字段时，不能继续验证对应变化原因。经营诊断是描述性分析与同期证据判断，不等同于统计因果推断。

## 模块与数据流

```text
prism/
├── app.py                         # Streamlit 应用入口、上传与会话交互
├── datalume/
│   ├── agent_runtime.py           # 请求规划、澄清恢复、工具循环与回退
│   ├── agent_tools.py             # 9 个白名单工具及参数校验
│   ├── agent_models.py            # Agent 状态、步骤、Observation、响应
│   ├── deepseek_client.py         # OpenAI-compatible DeepSeek 客户端
│   ├── skill_loader.py            # Skill metadata 发现与正文加载
│   ├── ai_agent.py                # 单步 LLM 分析和确定性回退
│   ├── router.py                  # 确定性问题分类与上下文意图回退
│   ├── data_processor.py          # CSV/Excel 导入、校验、指标与能力检测
│   ├── analysis_engine.py         # 七类经营分析
│   ├── database_manager.py        # 数据工作区与状态隔离
│   ├── visualization.py           # 内置 Altair 图表
│   ├── report_generator.py        # 结构化经营报告及 Markdown 输出
│   ├── ui_components.py           # 界面样式、消息和分析结果组件
│   ├── brand_assets.py            # 品牌图片路径
│   ├── analysis.py                # 经营建议兼容入口
│   ├── insight.py                 # 基础看板洞察
│   ├── strategy.py                # 策略摘要兼容入口
│   └── assets/                    # 品牌图片
├── skills/
│   ├── business-diagnosis/SKILL.md
│   ├── business-metrics/SKILL.md
│   └── clarify-analysis-request/SKILL.md
├── data/
│   ├── demo_orders.csv            # 扩展示例经营数据
│   └── orders.csv                 # 较小回归样例
├── docs/                          # 开发说明和展示图片
├── tests/                         # 自动化测试
├── .env.example                   # 空 Key 与配置示例
├── requirements.txt               # 运行依赖
└── requirements-dev.txt           # 运行依赖及 pytest
```

工作区通过稳定 `database_id` 隔离 CSV 内容、聊天记录、Agent 上下文、分析历史与报告。创建、切换、置顶、重命名和删除管理的是本次 Streamlit 会话对象；它们没有接入持久化数据库，也没有生产环境用户认证或访问控制。

## 测试与复核

在项目根目录执行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

[tests/conftest.py](../tests/conftest.py) 的自动 fixture 会把 `DEEPSEEK_API_KEY` 设为空；自动化测试使用模拟客户端，不验证真实 DeepSeek 服务可用性，也不等同于真实模型端到端验收。

测试涵盖导入与清洗、字段能力、七类分析、白名单参数与调用、Agent 循环和回退、三个 Skill 的选择与接入、澄清恢复、工作区状态隔离、图表、报告及 Streamlit AppTest。通过数量以当前运行结果为准，本指南不固化历史数字。

涉及新字段或新指标的修改，应同时核对输入粒度、分母、周期、排序、缺失值与零分母；涉及新 Skill 的修改，应确认选择逻辑、正文加载、实际工具能力与公开轨迹一致。模拟测试通过后，真实模型表现仍需使用可追溯的数据样例单独验证。
