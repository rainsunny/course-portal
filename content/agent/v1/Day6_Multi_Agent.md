# Day 6: 多 Agent 系统

> **学习时长**: 2-3 小时
> **核心问题**: 如何让多个 Agent 协作完成复杂任务？

---

## 学习目标

1. **理解多 Agent 架构** - 为什么需要多 Agent
2. **掌握协作模式** - 路由、协作、层级、对话
3. **了解主流框架** - OpenAI Swarm、LangGraph
4. **实现多 Agent 系统** - 动手实现协作 Agent

---

## 第一部分：为什么需要多 Agent？

### 1.1 单 Agent 的局限

| 局限 | 说明 |
|------|------|
| 单次响应能力有限 | 一个 Agent 能处理的信息量有限 |
| 缺乏长期记忆 | 难以跨任务保持一致性 |
| 推理深度受限 | 复杂问题需要多轮思考 |
| 单一专业领域 | 难以同时精通多个领域 |

### 1.2 多 Agent 的思路

**核心思想**：将复杂 Agent 分解为多个专门化的 Agent

```
单 Agent（大而全）：
┌─────────────────────────┐
│      Super Agent        │
│  - 搜索                  │
│  - 编程                  │
│  - 写作                  │
│  - 分析                  │
└─────────────────────────┘
问题：Prompt 太长，难以专精

多 Agent（小而精）：
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Search  │  │  Code   │  │  Write  │
│ Agent   │  │  Agent  │  │  Agent  │
└─────────┘  └─────────┘  └─────────┘
优势：每个 Agent 专注一件事，更高效
```

---

## 第二部分：多 Agent 架构模式

### 2.1 路由模式（Triage）

```
┌─────────────┐
│ Triage Agent│ ← 分类输入
└──────┬──────┘
   ↙   ↓   ↘
[Agent A][Agent B][Agent C]
```

**适用场景**：客服系统、问题分类

**实现**：

```python
class TriageSystem:
    """路由系统"""
    
    def __init__(self):
        self.triage_agent = None
        self.specialist_agents = {}
    
    def register_specialist(self, name: str, agent):
        """注册专家 Agent"""
        self.specialist_agents[name] = agent
    
    def route(self, user_input: str):
        """路由到合适的专家"""
        # 1. 分类
        category = self.triage_agent.classify(user_input)
        
        # 2. 获取专家
        specialist = self.specialist_agents.get(category)
        
        # 3. 转交处理
        if specialist:
            return specialist.handle(user_input)
        else:
            return self.specialist_agents["general"].handle(user_input)
```

### 2.2 协作模式（Collaboration）

```
┌────────────────────────────────┐
│       Shared Workspace         │
│   (共享文件系统/黑板)           │
└───────────┬────────────────────┘
      ↙     ↓     ↘
[Agent 1][Agent 2][Agent 3]
   ↓        ↓        ↓
  读取 ←→ 写入 ←→ 读取
```

**适用场景**：编程项目、研究报告

**实现**：

```python
class CollaborationSystem:
    """协作系统"""
    
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.agents = []
    
    def add_agent(self, agent):
        """添加协作 Agent"""
        agent.workspace = self.workspace
        self.agents.append(agent)
    
    def run(self, task: str):
        """运行协作"""
        # 1. 分解任务
        subtasks = self.decompose(task)
        
        # 2. 并行执行
        for agent, subtask in zip(self.agents, subtasks):
            agent.assign(subtask)
            agent.start()
        
        # 3. 等待完成
        for agent in self.agents:
            agent.wait()
        
        # 4. 合并结果
        return self.merge_results()
```

### 2.3 层级模式（Hierarchy）

```
        [Manager Agent]
         ↙    ↓    ↘
   [Worker A][Worker B][Worker C]
```

**适用场景**：复杂项目、需要协调

**实现**：

```python
class HierarchySystem:
    """层级系统"""
    
    def __init__(self):
        self.manager = None
        self.workers = []
    
    def run(self, task: str):
        """运行层级系统"""
        # 1. Manager 分解任务
        plan = self.manager.plan(task)
        
        # 2. 分配给 Workers
        for item in plan:
            worker = self.select_worker(item)
            result = worker.execute(item)
            
            # 3. Worker 汇报给 Manager
            self.manager.receive_result(result)
        
        # 4. Manager 综合结果
        return self.manager.synthesize()
```

### 2.4 对话模式（Conversation）

```
[Agent A] ←→ [Agent B]
    ↓            ↓
 [Output] ←→ [Feedback]
```

**适用场景**：讨论、辩论、协作创作

**实现**：

```python
class ConversationSystem:
    """对话系统"""
    
    def __init__(self, agent_a, agent_b, max_turns: int = 5):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.max_turns = max_turns
    
    def converse(self, topic: str):
        """进行对话"""
        message = topic
        
        for turn in range(self.max_turns):
            # Agent A 发言
            response_a = self.agent_a.respond(message)
            
            # Agent B 回应
            response_b = self.agent_b.respond(response_a)
            
            # 检查是否达成共识
            if self.check_consensus(response_a, response_b):
                return self.synthesize(response_a, response_b)
            
            message = response_b
        
        return "未能达成共识"
```

---

## 第三部分：OpenAI Swarm 详解

### 3.1 核心概念

OpenAI Swarm 是一个轻量级多 Agent 框架：

**概念**：
- **Agent**：具有特定指令和工具的实体
- **Routine**：一组步骤和执行它们的工具
- **Handoff**：Agent 将对话移交给另一个 Agent

### 3.2 基本用法

```python
from swarm import Swarm, Agent

# 创建客户端
client = Swarm()

# 定义工具
def transfer_to_sales():
    """转交给销售"""
    return sales_agent

def transfer_to_refund():
    """转交给退款"""
    return refund_agent

# 创建 Agent
sales_agent = Agent(
    name="Sales Agent",
    instructions="你是销售助手，帮助用户购买产品。",
    functions=[transfer_to_refund]
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="你是退款助手，帮助用户处理退款。",
    functions=[transfer_to_sales]
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="收集信息，将用户导向正确的部门。",
    functions=[transfer_to_sales, transfer_to_refund]
)

# 运行
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "我想退款"}]
)
```

### 3.3 Handoff 机制

```python
def transfer_to_agent(agent_name: str):
    """创建转交函数"""
    def transfer():
        return agent_registry[agent_name]
    transfer.__name__ = f"transfer_to_{agent_name}"
    return transfer

# 使用
support_agent = Agent(
    name="Support",
    instructions="你是客服代表。",
    functions=[
        transfer_to_agent("technical"),
        transfer_to_agent("billing"),
        transfer_to_agent("general")
    ]
)
```

### 3.4 最佳实践

1. **每个 Agent 专注一件事**
2. **清晰的 Handoff 条件**
3. **避免过度嵌套**
4. **监控和调试**

---

## 第四部分：多 Agent 系统的挑战

### 4.1 发散问题

多个 Agent 可能各自为政，难以收敛：

```
Agent A：我觉得应该这样做...
Agent B：不对，应该那样做...
Agent C：你们都不对...

问题：没有收敛机制
```

**解决方案**：
- 设置协调者（Manager）
- 定义优先级
- 投票机制

### 4.2 任务分解难度

如何划分任务边界？

```python
def decompose_task(task: str) -> List[SubTask]:
    """任务分解"""
    # 1. 分析任务
    analysis = analyze(task)
    
    # 2. 识别子任务
    subtasks = identify_subtasks(analysis)
    
    # 3. 确定依赖
    dependencies = determine_dependencies(subtasks)
    
    # 4. 分配 Agent
    assignments = assign_agents(subtasks, dependencies)
    
    return assignments
```

### 4.3 维护成本

复杂架构 vs 简单提示词修改：

```
问题：模型快速迭代，复杂架构可能很快过时

建议：
- 从简单开始
- 只在必要时增加复杂度
- 保持架构可替换
```

### 4.4 成本和延迟

多 Agent = 多次 LLM 调用：

| 模式 | 调用次数 | 成本 | 延迟 |
|------|----------|------|------|
| 单 Agent | 1x | 低 | 低 |
| 路由 | 2x | 中 | 中 |
| 协作 | Nx | 高 | 高（并行时低） |
| 层级 | 多次 | 高 | 高 |

---

## 第五部分：何时使用多 Agent？

### 适合的场景

✅ **大量独立能力**
- 能力难以编码到单个提示词
- 需要专业化处理

✅ **明确的领域划分**
- 不同 Agent 负责不同领域
- 边界清晰

✅ **需要专业化处理**
- 编程 Agent + 写作 Agent + 分析 Agent
- 每个领域有专门知识

### 不适合的场景

❌ **简单任务**
- 单 Agent 就能完成

❌ **对延迟敏感**
- 多次调用增加延迟

❌ **成本敏感**
- 多次调用成本高

❌ **模型能力可能提升**
- 今天需要多 Agent，明天可能单 Agent 就够

---

## 实践练习

### 练习 1：实现路由系统

创建一个客服路由系统：
- Triage Agent：分类
- Sales Agent：销售
- Support Agent：技术支持
- Refund Agent：退款

### 练习 2：实现协作系统

创建一个编程协作系统：
- Planner Agent：规划
- Coder Agent：编码
- Reviewer Agent：审查
- 共享工作区

### 练习 3：实现对话系统

创建一个辩论系统：
- Agent A：支持方
- Agent B：反对方
- 进行辩论并达成结论

---

## 今日总结

### 核心模式

```
路由模式：Triage → 专家 Agent
协作模式：共享 Workspace → 并行处理
层级模式：Manager → Workers
对话模式：Agent A ↔ Agent B
```

### 关键洞察

1. **多 Agent 不是银弹**：简单任务用单 Agent
2. **清晰的责任划分**：每个 Agent 专注一件事
3. **注意成本和延迟**：多次调用有代价
4. **模型在进步**：今天的复杂架构明天可能过时

### 明日预告

Day 7：实战项目 —— 综合运用所学知识，完成一个完整的 Agent 项目。

---

## 阅读材料

1. [OpenAI Swarm](https://github.com/openai/swarm)
2. [LangGraph Multi-agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
3. [Microsoft AutoGen](https://microsoft.github.io/autogen/)

---

*Day 6 完成！明天我们将进行实战项目。*
