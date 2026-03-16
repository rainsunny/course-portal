# Day 3: 完整的 Agent —— 自主决策与执行

> **学习时长**: 2-3 小时
> **核心问题**: Agent 如何自主决策、规划和执行？

---

## 学习目标

完成今天的学习后，你将能够：

1. **理解 Agent 核心循环** - 掌握 Agent 的执行流程
2. **实现 ReAct 模式** - 推理与行动交替进行
3. **设计规划系统** - 让 Agent 能分解和追踪目标
4. **实现自我验证** - Agent 能检验和修正自己的工作

---

## 引言：从 Workflow 到 Agent

回顾 Day 2，Workflow 的特点是**预定义路径**。但有些任务无法预先确定步骤：

```
用户："帮我修复这个 bug"

Workflow 思维：
Step 1: 分析代码 → Step 2: 定位问题 → Step 3: 修复 → Step 4: 测试

问题：如果 Step 2 定位失败呢？如果需要先查文档呢？
```

Agent 的关键区别：**动态决策**。

```
Agent 思维：
循环 {
    思考：当前状态是什么？下一步应该做什么？
    行动：执行选定的动作
    观察：结果如何？
    判断：完成了吗？需要调整吗？
}
```

---

## 第一部分：Agent 核心循环

### 1.1 Agent 循环图解

```
┌─────────────────────────────────────────────────────┐
│                   Agent 主循环                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│    ┌─────────┐                                      │
│    │ 用户输入 │                                      │
│    └────┬────┘                                      │
│         ↓                                           │
│    ┌─────────┐                                      │
│    │ 理解任务 │ ← 初始理解                          │
│    └────┬────┘                                      │
│         ↓                                           │
│    ┌─────────┐     ┌─────────┐                      │
│    │  规划    │ ←→ │ 调整规划 │ ← 动态调整          │
│    └────┬────┘     └─────────┘                      │
│         ↓                                           │
│    ┌─────────────────────────────────┐              │
│    │         执行循环                 │              │
│    │  ┌─────────┐                    │              │
│    │  │ 思考    │                    │              │
│    │  └────┬────┘                    │              │
│    │       ↓                         │              │
│    │  ┌─────────┐                    │              │
│    │  │ 行动    │                    │              │
│    │  └────┬────┘                    │              │
│    │       ↓                         │              │
│    │  ┌─────────┐                    │              │
│    │  │ 观察    │                    │              │
│    │  └────┬────┘                    │              │
│    │       ↓                         │              │
│    │  ┌─────────┐  否                │              │
│    │  │ 完成？   │───────────────────┤              │
│    │  └────┬────┘                    │              │
│    │       │ 是                      │              │
│    └───────┼─────────────────────────┘              │
│            ↓                                        │
│       ┌─────────┐                                   │
│       │ 输出结果 │                                   │
│       └─────────┘                                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.2 ReAct 模式详解

**论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

**核心思想**：推理（Reasoning）和行动（Acting）交替进行

```
传统方式：
[问题] → [直接回答]
问题：容易产生幻觉

Chain-of-Thought：
[问题] → [思考链] → [答案]
优点：推理更可靠
问题：没有外部信息

ReAct：
[问题] → [Thought] → [Action] → [Observation] → [Thought] → ... → [Answer]
优点：结合推理和外部信息
```

### 1.3 ReAct 示例

```
问题：北京今天的天气如何？

Thought: 用户想知道北京今天的天气，我需要搜索获取实时信息。
Action: search("北京天气 今天")
Observation: 北京今天晴天，最高温度 28°C，空气质量良好。

Thought: 我已经获得了天气信息，可以回答用户了。
Action: finish("北京今天晴天，最高温度 28°C，空气质量良好。")
```

### 1.4 ReAct 的优势

| 方面 | ReAct 优势 |
|------|-----------|
| **准确性** | 通过行动获取真实信息，减少幻觉 |
| **可解释性** | 思考轨迹可见，便于理解决策过程 |
| **灵活性** | 可以根据观察调整策略 |
| **纠错能力** | 错误可以通过后续行动纠正 |

详见示例代码 `react_agent.py`。

---

## 第二部分：规划系统

### 2.1 为什么需要规划？

没有规划的 Agent：

```
Agent：我已经修复了 bug
用户：真的修复了吗？测试通过了吗？
Agent：呃...我没有运行测试
```

有规划的 Agent：

```
Agent 创建计划：
1. 分析需求 → 2. 设计方案 → 3. 实现 → 4. 测试 → 5. 验证

Agent 执行：
Step 1: 需求分析... ✓
Step 2: 设计方案... ✓
Step 3: 实现... ✓
Step 4: 运行测试... 发现 2 个失败
Step 3: 修复问题...
Step 4: 再次测试... ✓
Step 5: 最终验证... ✓
```

### 2.2 规划文件设计

```markdown
# PLAN.md

## 目标
实现用户登录功能

## 进度
- [x] 需求分析
- [x] 数据库设计
- [ ] 实现登录逻辑（当前）
- [ ] 添加错误处理
- [ ] 编写测试

## 当前任务
正在实现 login() 函数

## 发现的问题
- 需要确认密码加密方式

## 决策记录
- 2024-01-15: 决定使用 JWT 作为认证方案
```

详见示例代码 `planning_system.py`。

---

## 第三部分：自我验证

### 3.1 验证策略

#### 策略一：测试驱动

```python
def verify_with_tests(code: str) -> dict:
    # 1. 生成测试用例
    # 2. 运行测试
    # 3. 分析结果
    pass
```

#### 策略二：结果检查

```python
def verify_result(expected: str, actual: str) -> dict:
    # 判断实际结果是否符合预期
    pass
```

#### 策略三：Ralph Loop

防止 Agent 过早退出：

```python
class RalphLoop:
    """强制 Agent 继续工作直到真正完成"""
    
    def run(self, task: str) -> str:
        for loop in range(self.max_loops):
            result = self.agent.run(task)
            
            if self.agent.wants_to_finish():
                verification = self.verifier.verify(task, result)
                
                if verification["complete"]:
                    return result  # 真正完成
                else:
                    # 重置并继续
                    self.agent.reset_context()
                    self.agent.inject_task(task)
            else:
                continue
        
        return "达到最大循环次数"
```

详见示例代码 `verification_system.py`。

---

## 第四部分：何时使用 Agent？

### 4.1 决策指南

**适合使用 Agent**：
- ✅ 开放性问题，无法预测步骤数
- ✅ 需要根据中间结果动态调整
- ✅ 可信环境（沙盒）
- ✅ 可接受试错

**不适合使用 Agent**：
- ❌ 步骤固定、可预测 → 用 Workflow
- ❌ 简单任务 → 直接 LLM 调用
- ❌ 对延迟敏感
- ❌ 成本敏感

### 4.2 决策树

```
步骤是否可预测？
├─ 是 → Workflow
└─ 否 → 是否需要动态调整？
        ├─ 是 → Agent
        └─ 否 → Workflow

是否在高风险环境？
├─ 是 → 考虑人工监督
└─ 否 → Agent

成本/延迟是否可接受？
├─ 是 → Agent
└─ 否 → 优化为 Workflow
```

---

## 今日总结

### 核心概念

```
Agent 循环：理解 → 规划 → 执行循环 { 思考 → 行动 → 观察 } → 验证

ReAct：Thought → Action → Observation → 循环

规划：目标 → 计划 → 任务 → 执行 → 更新

验证：测试驱动 / 结果检查 / Ralph Loop
```

### 关键洞察

1. **Agent 不等于复杂**：简单 Agent 也可以有效
2. **规划是关键**：好的规划大大提高成功率
3. **验证不可少**：没有验证的 Agent 不可信
4. **知道何时不用 Agent**：Workflow 往往更可靠

### 明日预告

Day 4：工具与 MCP —— 如何设计和实现高质量 Agent 工具。

---

## 阅读材料

1. [ReAct 论文](https://arxiv.org/abs/2210.03629)
2. [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
3. [LangChain: Agent Architectures](https://python.langchain.com/docs/modules/agents/)

---

*Day 3 完成！明天我们将深入工具设计与 MCP 协议。*
