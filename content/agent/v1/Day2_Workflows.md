# Day 2: 核心模式 —— 工作流模式（Workflows）

> **学习时长**: 2-3 小时
> **核心问题**: 如何用 LLM 构建可靠的自动化流程？

---

## 学习目标

完成今天的学习后，你将能够：

1. **理解工作流模式** - 区分 5 种核心模式及其适用场景
2. **实现提示链** - 将任务分解为固定步骤序列
3. **实现路由** - 根据输入动态选择处理路径
4. **实现并行化** - 多个 LLM 同时处理任务
5. **实现评估-优化循环** - 迭代改进输出质量

---

## 引言：为什么需要工作流？

### 从 Agent 到 Workflow

昨天我们学习了 Agent = Model + Harness。Agent 能够自主决策，但这种灵活性是有代价的：

| Agent | Workflow |
|-------|----------|
| 路径动态 | 路径固定 |
| 难预测 | 可预测 |
| 调试复杂 | 调试简单 |
| 成本不确定 | 成本可控 |

**Anthropic 的建议**：

> 从简单开始。先用 Workflow，当 Workflow 不足够时再考虑 Agent。

### 何时用 Workflow？

```
任务复杂度
    │
    │         Agent 领域
    │        (不确定路径)
    │    ┌───────────────┐
    │    │               │
    │    │    ○○○○○     │
    │    │   ○○○○○○○    │
    │────┼───○○○○○○○────┼──── Workflow 领域
    │    │   (确定路径)  │     (可预测、可控)
    │    │               │
    │    └───────────────┘
    │
    └─────────────────────────→ 任务确定性
```

### 五种核心模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| **Augmented LLM** | LLM + 检索 + 工具 + 记忆 | 所有模式的基础 |
| **Prompt Chaining** | 固定步骤序列 | 可分解的线性任务 |
| **Routing** | 分类并路由到专门处理器 | 多类别任务 |
| **Parallelization** | 并行处理 + 聚合 | 需要多角度处理 |
| **Evaluator-Optimizer** | 生成 + 评估 + 迭代改进 | 需要高质量输出 |

---

## 模式一：Augmented LLM（增强型 LLM）

### 概念

这是所有 Workflow 和 Agent 的基础构建块：

```
Augmented LLM = LLM + Retrieval + Tools + Memory
```

不是简单的 LLM 调用，而是让模型能够：
- **主动检索**：生成搜索查询，获取相关信息
- **使用工具**：选择并调用合适的工具
- **管理记忆**：决定保留什么信息

### 架构图

```
┌─────────────────────────────────────────────────┐
│              Augmented LLM                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │Memory   │  │Tools    │  │Retrieval│         │
│  │记忆系统 │  │工具系统 │  │检索系统 │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │              │
│       └────────────┼────────────┘              │
│                    │                           │
│              ┌─────┴─────┐                     │
│              │    LLM    │                     │
│              └───────────┘                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 代码实现

详见示例文件 `augmented_llm.py`。

关键方法：

```python
class AugmentedLLM:
    def chat(self, user_input: str) -> str:
        # 1. 检索相关信息
        context = self.retrieve(user_input)
        
        # 2. 构建增强提示词
        enhanced_prompt = self.build_prompt(user_input, context)
        
        # 3. 调用 LLM
        response = self.llm.call(enhanced_prompt)
        
        # 4. 更新记忆
        self.memory.add(user_input, response)
        
        return response
```

### 与普通 LLM 调用的区别

| 普通调用 | Augmented LLM |
|----------|---------------|
| 单次请求 | 循环处理 |
| 无状态 | 有记忆 |
| 无工具 | 可调用工具 |
| 无检索 | 可检索知识 |

---

## 模式二：Prompt Chaining（提示链）

### 概念

将任务分解为固定步骤序列，每一步的输出是下一步的输入。

```
[Input] → [Step 1] → [Step 2] → [Step 3] → [Output]
              ↓           ↓
           [Gate]      [Gate]
```

**Gate（质量门）**：在每个步骤后验证输出质量，不合格则重试或终止。

### 适用场景

- ✅ 任务可以干净地分解为子任务
- ✅ 每一步需要不同的"模式"或"专业"
- ✅ 可以用延迟换取更高准确性
- ❌ 步骤之间有复杂依赖关系
- ❌ 需要根据中间结果调整路径

### 典型用例

1. **内容生产流水线**
   ```
   大纲生成 → 大纲检查 → 内容编写 → 编辑润色
   ```

2. **翻译流程**
   ```
   原文分析 → 翻译 → 校对 → 本地化调整
   ```

3. **代码生成**
   ```
   需求分析 → 设计方案 → 代码编写 → 测试生成
   ```

### 代码实现

详见示例文件 `prompt_chaining.py`。

核心结构：

```python
class PromptChain:
    def add_step(self, name, prompt_template, gate=None):
        """添加一个步骤"""
        self.steps.append(Step(name, prompt_template, gate))
    
    def run(self, input):
        """运行整个链"""
        current = input
        for step in self.steps:
            # 执行步骤
            output = self.llm.call(step.prompt.format(input=current))
            
            # 质量检查
            if step.gate:
                passed, feedback = step.gate(output)
                if not passed:
                    return {"error": feedback}
            
            current = output
        return current
```

### 质量门设计

质量门是提示链的关键组件：

```python
def outline_gate(outline: str) -> tuple[bool, str]:
    """大纲质量检查"""
    # 检查必要章节
    required = ["引言", "主体", "结论"]
    missing = [s for s in required if s not in outline]
    
    if missing:
        return False, f"缺少章节：{missing}"
    
    # 检查长度
    if len(outline) < 100:
        return False, "大纲太短，内容不够详细"
    
    # 检查结构
    if outline.count("##") < 3:
        return False, "大纲层次不够丰富"
    
    return True, "大纲符合要求"
```

### 最佳实践

1. **每个步骤职责单一**
   - 不要让一个步骤做太多事情
   - 便于调试和优化

2. **定义清晰的输入输出格式**
   - 使用结构化格式（JSON、Markdown）
   - 减少歧义

3. **添加质量检查门**
   - 自动验证输出质量
   - 早期发现问题

4. **记录中间结果**
   - 便于调试
   - 支持断点续传

5. **考虑重试策略**
   - 质量门失败时重试
   - 设置最大重试次数

---

## 模式三：Routing（路由）

### 概念

分类输入，导向专门的后续任务。

```
        [Input]
           ↓
      [Classifier]
      ↙    ↓    ↘
  [Handler A] [Handler B] [Handler C]
      ↓         ↓         ↓
   [Output]  [Output]  [Output]
```

### 适用场景

- ✅ 复杂任务有明显不同的子类别
- ✅ 需要专业化处理不同类型
- ✅ 简单问题用小模型，复杂问题用大模型
- ❌ 类别边界模糊
- ❌ 需要多类别联合处理

### 分类策略

#### 1. 基于规则

```python
def rule_based_classify(text: str) -> Category:
    """基于规则的分类"""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["退款", "refund"]):
        return Category.REFUND
    elif any(w in text_lower for w in ["bug", "错误"]):
        return Category.TECHNICAL
    else:
        return Category.GENERAL
```

**优点**：快速、可预测、无成本
**缺点**：覆盖有限、需要维护规则

#### 2. 基于 LLM

```python
def llm_classify(text: str) -> Category:
    """基于 LLM 的分类"""
    prompt = f"""
    请将以下文本分类到正确的类别：
    
    类别：
    - REFUND: 退款请求
    - TECHNICAL: 技术支持
    - COMPLAINT: 投诉
    - FEEDBACK: 反馈建议
    - GENERAL: 一般咨询
    
    文本：{text}
    
    类别：
    """
    response = llm.call(prompt)
    return parse_category(response)
```

**优点**：灵活、覆盖广、可理解上下文
**缺点**：成本高、延迟大、可能出错

#### 3. 混合策略

```python
def hybrid_classify(text: str) -> Category:
    """混合分类"""
    # 先用规则快速分类
    rule_result = rule_based_classify(text)
    
    # 高置信度直接返回
    if rule_result.confidence > 0.9:
        return rule_result.category
    
    # 低置信度调用 LLM
    return llm_classify(text)
```

**最佳实践**：规则优先，LLM 回退

### 代码实现

详见示例文件 `routing.py`。

### 路由到不同模型

路由的一个重要应用是根据复杂度选择模型：

```python
class ModelRouter:
    """根据任务复杂度路由到不同模型"""
    
    def __init__(self):
        self.small_model = "gpt-3.5-turbo"  # 快速、便宜
        self.large_model = "gpt-4"          # 强大、昂贵
    
    def classify_complexity(self, task: str) -> str:
        """评估任务复杂度"""
        # 简单特征
        simple_indicators = [
            len(task) < 100,
            task.count("？") <= 1,
            "翻译" in task,
            "总结" in task,
        ]
        
        if sum(simple_indicators) >= 2:
            return "simple"
        return "complex"
    
    def route(self, task: str) -> str:
        """路由到合适的模型"""
        complexity = self.classify_complexity(task)
        
        if complexity == "simple":
            return self.small_model
        else:
            return self.large_model
```

---

## 模式四：Parallelization（并行化）

### 概念

多个 LLM 同时处理任务，结果聚合。

### 两种变体

#### 1. Sectioning（任务分解）

将任务分解为独立子任务并行执行：

```
        [复杂任务]
      ↙    ↓    ↘
  [子任务A] [子任务B] [子任务C]
      ↓         ↓         ↓
      └─────────┼─────────┘
                ↓
           [聚合结果]
```

**案例**：文档分析
- 子任务 A：语法检查
- 子任务 B：风格分析
- 子任务 C：事实核查
- 子任务 D：结构分析

#### 2. Voting（投票）

同一任务多次执行，获得多样输出：

```
        [任务]
      ↙    ↓    ↘
  [执行1] [执行2] [执行3]
      ↓         ↓         ↓
      └─────────┼─────────┘
                ↓
           [投票/聚合]
```

**案例**：内容审核
- 多个"审核员"独立判断
- 投票决定最终结果
- 降低误判风险

### 适用场景

| 变体 | 适用场景 |
|------|----------|
| Sectioning | 任务可分解为独立部分，需要多角度分析 |
| Voting | 需要降低错误率，需要多样输出，高风险决策 |

### 代码实现

详见示例文件 `parallelization.py`。

### 并行化考量

**优点**：
- 不增加延迟（并行执行）
- 提高可靠性（多投票）
- 多角度处理

**缺点**：
- 成本倍增
- 需要聚合逻辑
- 可能结果不一致

---

## 模式五：Orchestrator-Workers（编排者-工作者）

### 概念

中央 LLM 动态分解任务，委派给 Worker，综合结果。

```
        [Input]
           ↓
      [Orchestrator]  ← 动态规划
      ↙    ↓    ↘
  [Worker 1] [Worker 2] [Worker 3]
      ↓         ↓         ↓
      └─────────┼─────────┘
                ↓
          [Orchestrator]  ← 综合结果
                ↓
            [Output]
```

**与 Parallelization 的区别**：
- Parallelization：子任务预定义
- Orchestrator-Workers：子任务动态决定

### 适用场景

- ✅ 无法预测子任务的复杂任务
- ✅ 需要根据输入动态分配工作
- ✅ 编程（文件数量和修改类型不确定）
- ❌ 简单固定流程

### 典型用例

1. **编程助手**
   - Orchestrator：分析需求，分解任务
   - Worker：写代码、写测试、写文档

2. **研究报告**
   - Orchestrator：规划研究结构
   - Worker：搜索、分析、撰写不同章节

3. **项目管理**
   - Orchestrator：评估任务，分配资源
   - Worker：执行具体工作

### 代码实现

详见示例文件 `orchestrator_workers.py`。

---

## 模式六：Evaluator-Optimizer（评估者-优化者）

### 概念

一个 LLM 生成，另一个评估反馈，循环迭代。

```
[Input] → [Generator] → [Output]
               ↑           ↓
               ← [Evaluator] ←
                  (反馈循环)
```

### 适用场景

- ✅ 有明确评估标准
- ✅ 迭代改进有明显价值
- ✅ LLM 能提供有效反馈
- ❌ 没有明确标准
- ❌ 单次输出足够好
- ❌ 成本敏感

### 典型用例

1. **文学翻译**
   - Generator：翻译文本
   - Evaluator：检查准确性、流畅性、风格
   - 循环直到满意

2. **代码优化**
   - Generator：编写代码
   - Evaluator：检查正确性、效率、可读性
   - 循环改进

3. **内容创作**
   - Generator：撰写文章
   - Evaluator：评估质量、原创性、吸引力
   - 多轮打磨

### 代码实现

详见示例文件 `evaluator_optimizer.py`。

### 评估者设计

评估者是关键，需要：

1. **明确的评估标准**

```python
EVALUATION_CRITERIA = """
评估翻译质量：

1. 准确性（40%）：是否准确传达原文意思
2. 流畅性（30%）：译文是否自然流畅
3. 风格（20%）：是否保持原文风格
4. 完整性（10%）：是否有遗漏

评分：1-10 分
"""
```

2. **具体的反馈**

```python
def evaluate(translation: str, original: str) -> dict:
    """评估翻译质量"""
    return {
        "score": 7.5,
        "passed": False,
        "feedback": """
        评估结果：
        
        - 准确性：8/10 - 大部分准确，但第三段有轻微偏差
        - 流畅性：7/10 - 整体流畅，但有几处表达略显生硬
        - 风格：8/10 - 较好保持了原文风格
        - 完整性：7/10 - 有两句话被省略
        
        建议：
        1. 第三段"..."处理不够准确，建议改为"..."
        2. 恢复被省略的两句话
        3. "..."处的表达可以更自然
        """
    }
```

### 终止条件

```python
def should_continue(evaluation: dict, iteration: int) -> bool:
    """决定是否继续迭代"""
    # 达到质量标准
    if evaluation["score"] >= 8.0:
        return False
    
    # 达到最大迭代次数
    if iteration >= 5:
        return False
    
    # 连续两次评分没有提升
    if no_improvement():
        return False
    
    return True
```

---

## 模式对比与选择

### 决策树

```
任务类型？
├─ 线性可分解？
│   └─ 是 → Prompt Chaining
│
├─ 多类别处理？
│   └─ 是 → Routing
│
├─ 可并行处理？
│   ├─ 子任务独立 → Parallelization (Sectioning)
│   └─ 需要多角度 → Parallelization (Voting)
│
├─ 动态任务分配？
│   └─ 是 → Orchestrator-Workers
│
└─ 需要迭代改进？
    └─ 是 → Evaluator-Optimizer
```

### 组合使用

实际项目中，多种模式经常组合使用：

```python
class ContentPipeline:
    """内容生产流水线（组合模式）"""
    
    def process(self, request: str):
        # 1. Routing：分类请求
        category = self.router.classify(request)
        
        # 2. Orchestrator-Workers：分配任务
        tasks = self.orchestrator.plan(request, category)
        results = self.workers.execute(tasks)
        
        # 3. Evaluator-Optimizer：迭代改进
        content = self.generator.create(results)
        for i in range(3):
            evaluation = self.evaluator.evaluate(content)
            if evaluation.passed:
                break
            content = self.generator.improve(content, evaluation.feedback)
        
        return content
```

---

## 实践练习

### 练习 1：实现一个翻译流水线

使用 Prompt Chaining 实现：
1. 原文分析
2. 翻译
3. 校对
4. 本地化调整

添加质量门验证每一步。

### 练习 2：实现一个客服路由系统

使用 Routing 实现：
- 分类：退款、技术支持、投诉、一般咨询
- 每类有专门的处理流程
- 统计路由分布

### 练习 3：实现一个代码审查系统

使用 Evaluator-Optimizer 实现：
- Generator：根据需求写代码
- Evaluator：审查代码质量
- 迭代改进

---

## 今日总结

### 核心模式回顾

```
Augmented LLM: 所有模式的基础
    ↓
Prompt Chaining: 固定步骤，线性流程
    ↓
Routing: 分类处理，专家分工
    ↓
Parallelization: 并行处理，提高效率
    ↓
Orchestrator-Workers: 动态分配，灵活应对
    ↓
Evaluator-Optimizer: 迭代改进，追求质量
```

### 关键洞察

1. **从简单开始**：先用 Workflow，不够再考虑 Agent
2. **模式可组合**：复杂系统是多种模式的组合
3. **质量门重要**：在关键节点验证输出
4. **成本与质量**：并行和迭代带来成本，需要权衡

### 明日预告

Day 3 我们将深入学习 **完整的 Agent**：
- Agent 的核心循环
- ReAct 模式详解
- 规划与自我验证
- 何时使用 Agent

---

## 阅读材料

### 必读

1. **[Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)**
   - Workflow vs Agent 的区分
   - 五种核心模式详解

### 选读

2. **[LangChain: Workflows](https://python.langchain.com/docs/expression_language/)**
   - LangChain Expression Language
   - 链式调用实现

3. **[Anthropic Cookbook: Basic Workflows](https://platform.claude.com/cookbook/patterns-agents-basic-workflows)**
   - 实际代码示例

---

## 常见问题

### Q1: Workflow 和 Agent 可以混合使用吗？

**A**: 可以。常见模式：
- Router 分流到多个 Workflow
- Workflow 中某个步骤使用 Agent
- Agent 内部使用 Workflow 处理子任务

### Q2: 如何选择模式？

**A**: 问自己：
- 任务步骤是否固定？→ Prompt Chaining
- 有明显类别区分？→ Routing
- 需要多角度处理？→ Parallelization
- 任务不确定？→ Orchestrator-Workers 或 Agent
- 需要高质量输出？→ Evaluator-Optimizer

### Q3: 并行化会增加成本吗？

**A**: 是的，但不会增加延迟。适合：
- 成本不敏感
- 延迟敏感
- 需要高可靠性

### Q4: Evaluator-Optimizer 要迭代多少次？

**A**: 建议：
- 设置最大迭代次数（如 3-5 次）
- 设置质量阈值，达到即停止
- 监控改进趋势，无提升则停止

### Q5: 如何调试 Workflow？

**A**: 
1. 记录每一步的输入输出
2. 添加详细日志
3. 单独测试每个步骤
4. 使用质量门早期发现问题

---

*Day 2 完成！明天我们将探索完整的 Agent 系统。*
