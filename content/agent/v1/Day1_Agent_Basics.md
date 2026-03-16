# Day 1: Agent 的本质与核心概念

> **学习时长**: 2-3 小时
> **核心问题**: 什么是 Agent？它和普通 LLM 应用有什么区别？

---

## 学习目标

完成今天的学习后，你将能够：

1. **定义 Agent** - 用第一性原理解释什么是 Agent
2. **区分 Workflow 与 Agent** - 理解两者的边界和适用场景
3. **理解 Harness 组件** - 知道为什么 LLM 需要"外套"才能成为 Agent
4. **手写 ReAct 循环** - 实现一个最简单的 Agent

---

## 第一部分：什么是 Agent？

### 1.1 从一个简单问题开始

假设你问 ChatGPT："北京今天天气怎么样？"

它会怎么回答？

```
抱歉，我无法提供实时信息。我的知识截止于...
```

为什么？因为 **LLM 本身只是一个静态的知识库**。

但如果你用的是某个"联网版"的 AI 助手，它可能会：

```
让我查一下...（搜索中）
北京今天晴天，最高温度 28°C，空气质量良好。
```

这个差别就是 Agent 的本质所在。

---

### 1.2 Agent 的第一性原理定义

**核心公式**：

```
Agent = Model + Harness
```

让我们拆解这个公式：

| 组件 | 是什么 | 能做什么 | 不能做什么 |
|------|--------|----------|------------|
| **Model（模型）** | LLM 本身 | 理解语言、推理、生成文本 | 无法持久化、无法执行、无法联网 |
| **Harness（框架）** | 外围系统 | 文件系统、工具执行、记忆管理 | 没有智能，只是基础设施 |

> **核心洞察**（Vivek Trivedy, LangChain）：
> 
> 一个原始的 LLM 不是 Agent。只有当 Harness 赋予它状态、工具执行、反馈循环、约束条件时，它才成为 Agent。

### 1.3 为什么模型需要 Harness？

让我们从模型的局限性出发，理解 Harness 的必要性：

#### 模型"不能"做的事情

```python
# 模型想做但做不到的事情：

# ❌ 无法持久化数据
user: "记住我的名字是张三"
model: "好的，我记住了"  # 但下次对话就忘了

# ❌ 无法执行操作
user: "帮我删除这个文件"
model: "我无法执行这个操作..."  # 只能说，不能做

# ❌ 无法获取实时信息
user: "今天股价多少？"
model: "我无法提供实时数据..."  # 知识有截止日期

# ❌ 无法记住历史
model: "我们之前聊过什么？"  # 超出 context window 就忘了
```

#### Harness 如何解决这些问题

| 模型局限 | Harness 解决方案 | 示例 |
|----------|------------------|------|
| 无法持久化数据 | 文件系统（Filesystem） | 保存对话、配置、状态 |
| 无法执行操作 | Bash/Code 执行工具 | 运行命令、执行代码 |
| 无法验证结果 | 沙盒环境 + 测试工具 | 自动测试、结果检查 |
| 无法获取新知识 | 搜索 + MCP 工具 | 联网搜索、API 调用 |
| 无法记住历史 | Memory 系统 | 长期记忆、摘要压缩 |
| Context 窗口有限 | Compaction + 压缩策略 | 智能摘要、信息过滤 |

---

### 1.4 一个类比：大脑 vs 身体

把 Agent 想象成一个人：

```
Model（模型） = 大脑
  - 有智能
  - 能思考、推理
  - 但无法直接与世界交互

Harness（框架） = 身体
  - 眼睛：搜索工具
  - 手：执行工具
  - 记忆：Memory 系统
  - 书本/笔记：文件系统
```

**没有 Harness 的模型**：像一个被困在罐子里的大脑，只能思考，无法行动。

**有了 Harness**：大脑有了身体，可以看、做、记、学。

---

## 第二部分：Workflow vs Agent

### 2.1 Anthropic 的定义

Anthropic 在《Building Effective Agents》中给出了清晰的区分：

| | Workflow | Agent |
|---|----------|-------|
| **定义** | LLM 和工具通过**预定义代码路径**编排的系统 | LLM **动态指导**自己的流程和工具使用 |
| **控制权** | 开发者预设 | 模型自主决策 |
| **路径** | 固定 | 动态 |
| **灵活性** | 低 | 高 |
| **可预测性** | 高 | 低 |
| **调试难度** | 低 | 高 |

### 2.2 直观对比

#### Workflow 示例：翻译流水线

```python
# 这是一个 Workflow，路径是预定义的

def translate_workflow(text, source_lang, target_lang):
    # Step 1: 检测语言（固定的第一步）
    detected = detect_language(text)
    
    # Step 2: 翻译（固定的第二步）
    translated = translate(text, source_lang, target_lang)
    
    # Step 3: 校对（固定的第三步）
    proofread = proofread(translated)
    
    return proofread

# 路径永远固定：检测 → 翻译 → 校对
# 不管输入是什么，流程都不会变
```

#### Agent 示例：开放式研究任务

```python
# 这是一个 Agent，路径由模型动态决定

async def research_agent(question):
    while not done:
        # 模型决定下一步做什么
        action = model.decide_next_action(question, current_state)
        
        if action.type == "search":
            results = search(action.query)
            current_state.add(results)
        elif action.type == "read":
            content = read(action.url)
            current_state.add(content)
        elif action.type == "synthesize":
            answer = synthesize(current_state)
            done = True
        elif action.type == "clarify":
            question = ask_user(action.question)
            # 继续循环...
    
    return answer

# 路径完全由模型决定
# 可能搜索3次、读2篇、问用户1次
# 也可能直接回答、或者搜索10次
```

### 2.3 什么时候用什么？

```
           任务确定性
               ↑
               │
    Workflow   │   Agent
    (确定性高) │   (确定性低)
               │
───────────────┼───────────────→ 任务复杂度
               │
    简单LLM调用 │   复杂Agent系统
    (简单任务) │   (复杂任务)
               │
```

**使用 Workflow 当**：
- ✅ 任务可以分解为固定步骤
- ✅ 流程可预测
- ✅ 需要高可靠性和可解释性
- ✅ 例：数据处理流水线、文档翻译、内容审核

**使用 Agent 当**：
- ✅ 任务是开放式的
- ✅ 无法预测需要多少步骤
- ✅ 需要根据中间结果动态调整
- ✅ 例：代码调试、研究助手、复杂客服

### 2.4 误区提醒

> **误区 1**：Agent 一定比 Workflow 好
> 
> **真相**：Agent 牺牲可预测性换取灵活性。如果任务本身是确定性的，用 Workflow 更好。

> **误区 2**：用 Agent 框架就等于构建 Agent
> 
> **真相**：你可以在 Agent 框架中实现 Workflow。框架只是工具，关键是设计思路。

> **误区 3**：Agent 可以完全自主
> 
> **真相**：Agent 仍需要边界条件、工具定义、安全约束。真正的自主需要更多基础设施。

---

## 第三部分：Harness 的核心组件

### 3.1 组件概览

一个完整的 Harness 通常包含以下组件：

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent Harness                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Tool System │  │   Memory    │  │ Filesystem │          │
│  │  工具系统    │  │   记忆系统   │  │  文件系统   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Sandbox    │  │  Context    │  │   Safety    │          │
│  │  沙盒环境   │  │  Context管理│  │   安全约束  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
                    ┌─────────────────┐
                    │      LLM        │
                    │     (模型)      │
                    └─────────────────┘
```

### 3.2 详细解析每个组件

#### 3.2.1 Tool System（工具系统）

**目的**：让模型能够与外部世界交互

**常见工具类型**：

| 类型 | 作用 | 示例 |
|------|------|------|
| 信息获取 | 搜索、查询 | web_search, database_query |
| 文件操作 | 读写文件 | read_file, write_file |
| 代码执行 | 运行代码 | exec_python, exec_bash |
| API 调用 | 外部服务 | send_email, create_issue |
| 状态管理 | 操作状态 | get_state, set_state |

**工具定义示例**：

```python
# 一个简单的工具定义

tools = [
    {
        "name": "search",
        "description": "搜索互联网获取信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_file",
        "description": "读取本地文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件的绝对路径"
                }
            },
            "required": ["path"]
        }
    }
]
```

#### 3.2.2 Memory System（记忆系统）

**目的**：跨会话保持状态，突破 Context 窗口限制

**记忆的层级**：

```
┌────────────────────────────────────────┐
│           短期记忆（Working Memory）    │
│         当前对话的 Context Window       │
├────────────────────────────────────────┤
│           中期记忆（Session Memory）    │
│         当前会话的对话历史              │
├────────────────────────────────────────┤
│           长期记忆（Long-term Memory）  │
│         跨会话持久化的知识/偏好         │
└────────────────────────────────────────┘
```

**记忆类型**：

1. **Semantic Memory（语义记忆）**
   - 提取的事实和知识
   - 例："用户的公司有500名员工"

2. **Episodic Memory（情景记忆）**
   - 对话中的具体事件
   - 例："上周三用户问了关于产品定价的问题"

3. **Preference Memory（偏好记忆）**
   - 用户的偏好设置
   - 例：`{"code_style": "Python, 使用 type hints"}`

#### 3.2.3 Filesystem（文件系统）

**目的**：
- 持久化数据
- 存储工作进度
- 管理项目文件

**典型用法**：

```python
# Agent 使用文件系统的场景

# 1. 存储计划
write_file("PLAN.md", """
# 任务计划
1. 分析需求
2. 设计方案
3. 实现代码
4. 测试验证
""")

# 2. 记录进度
write_file("PROGRESS.md", """
# 进度跟踪
- [x] 分析需求
- [x] 设计方案
- [ ] 实现代码  ← 当前
- [ ] 测试验证
""")

# 3. 存储中间结果
write_file("research_notes.md", """
# 研究笔记
## 关键发现
- 发现 1...
- 发现 2...
""")
```

#### 3.2.4 Sandbox（沙盒环境）

**目的**：安全地执行代码和命令

**为什么需要沙盒？**

```
❌ 没有沙盒的危险：
Agent: "让我删除所有文件来解决问题..."
→ 执行 rm -rf /
→ 💥 灾难性后果

✅ 有沙盒的保护：
Agent: "让我删除所有文件来解决问题..."
→ Sandbox: 拒绝执行危险命令
→ 或者在隔离环境中执行
→ ✅ 安全
```

**沙盒类型**：

| 类型 | 隔离程度 | 适用场景 |
|------|----------|----------|
| 容器（Docker） | 高 | 生产环境 |
| 进程级隔离 | 中 | 开发环境 |
| 权限限制 | 低 | 受信任环境 |

#### 3.2.5 Context Management（上下文管理）

**目的**：在有限的 Context Window 中保持有效工作

**核心挑战：Context Rot**

```
Context Rot 问题：
随着对话增长，模型推理能力下降

症状：
- "忘记"早期信息
- 重复之前的操作
- 推理质量下降
- 回答变得平庸
```

**解决方案**：

1. **Compaction（压缩）**
   ```python
   # 当 context 接近上限时
   if context_length > threshold:
       # 智能摘要，保留关键信息
       compressed = summarize(context)
       context = compressed
   ```

2. **Tool Call Offloading**
   ```python
   # 大型工具输出只保留摘要
   full_result = tool.execute()
   # 完整结果存文件
   save_to_file(full_result)
   # Context 中只保留摘要
   context.add(f"执行完成，结果已保存到 {file_path}")
   ```

3. **渐进式工具披露**
   ```python
   # 不一次性加载所有工具
   # 根据任务动态加载相关工具
   if task_involves("coding"):
       load_tools(["read_file", "write_file", "exec_bash"])
   elif task_involves("research"):
       load_tools(["search", "read_url"])
   ```

#### 3.2.6 Safety Constraints（安全约束）

**目的**：防止 Agent 做出危险或不当行为

**安全层级**：

```
┌─────────────────────────────────────────┐
│          用户确认层                      │
│   敏感操作需要用户明确批准               │
├─────────────────────────────────────────┤
│          工具限制层                      │
│   禁用危险工具，限制参数范围             │
├─────────────────────────────────────────┤
│          沙盒隔离层                      │
│   在隔离环境中执行                       │
├─────────────────────────────────────────┤
│          提示词约束层                    │
│   系统提示中定义行为边界                 │
└─────────────────────────────────────────┘
```

---

## 第四部分：工具的本质

### 4.1 工具定义的重要性

> **核心观点**（Anthropic）：
> 
> 工具定义应该获得与提示词工程同等的关注。
> 
> 在构建 SWE-bench Agent 时，我们花在优化工具上的时间比优化整体提示词还多。

为什么？因为工具是 Agent 与世界交互的**唯一接口**。

### 4.2 好工具的设计原则

#### 原则 1：给模型足够的"思考"空间

```python
# ❌ 不好的设计：让模型计算行号
{
    "name": "edit_file",
    "description": "编辑文件",
    "parameters": {
        "line_start": "起始行号",
        "line_end": "结束行号",
        "new_content": "新内容"
    }
}
# 模型需要：数行号 → 容易出错

# ✅ 好的设计：用自然语言定位
{
    "name": "edit_file",
    "description": "编辑文件，找到并替换指定内容",
    "parameters": {
        "path": "文件路径",
        "old_text": "要替换的原文本（必须完全匹配）",
        "new_text": "替换后的新文本"
    }
}
# 模型只需要：复制要改的内容 → 更自然
```

#### 原则 2：减少格式开销

```python
# ❌ 不好的设计：JSON 中的复杂转义
{
    "name": "write_code",
    "parameters": {
        "code": "代码字符串（需要转义引号、换行等）"
    }
}
# 模型需要处理：\" \\n \t 等转义

# ✅ 好的设计：直接传递文件路径
{
    "name": "write_file",
    "parameters": {
        "path": "文件路径",
        "content": "文件内容（原始文本，无需转义）"
    }
}
# 让 Harness 处理格式问题
```

#### 原则 3：防错设计（Poka-yoke）

```python
# ❌ 不好的设计：相对路径
{
    "name": "read_file",
    "parameters": {
        "path": "文件路径（相对于工作目录）"
    }
}
# 问题：Agent 离开根目录后路径会出错

# ✅ 好的设计：强制绝对路径
{
    "name": "read_file",
    "parameters": {
        "path": "文件的绝对路径（必须以 / 开头）"
    }
}
# 强制使用绝对路径，避免路径错误
```

### 4.3 案例：SWE-bench Agent 的工具优化

**问题**：模型在编辑文件时频繁出错

**调试过程**：

1. 分析错误日志
   ```
   错误类型分布：
   - 路径错误：45%
   - 行号计算错误：30%
   - 格式问题：15%
   - 其他：10%
   ```

2. 识别根本原因
   - 相对路径在 Agent 切换目录后失效
   - 行号计算对模型来说是负担

3. 优化工具设计
   ```python
   # 优化后
   {
       "name": "edit_file",
       "description": """精确编辑文件。
       
       注意：
       1. 必须使用绝对路径（如 /home/user/project/file.py）
       2. old_text 必须与文件内容完全匹配（包括空格和缩进）
       3. 建议先读取文件确认内容，再进行编辑
       """,
       "parameters": {
           "path": {
               "type": "string",
               "description": "文件的绝对路径"
           },
           "old_text": {
               "type": "string",
               "description": "要替换的原文（必须完全匹配）"
           },
           "new_text": {
               "type": "string",
               "description": "替换后的文本"
           }
       }
   }
   ```

4. 结果：工具使用成功率从 70% 提升到 98%

### 4.4 ACI（Agent-Computer Interface）设计

**类比**：
- HCI（人机交互）关注用户界面设计
- ACI（Agent-计算机交互）关注工具接口设计

**ACI 设计检查清单**：

```
□ 站在模型角度：工具描述是否清晰？
□ 参数名和描述是否足够明显？（像给初级开发者写文档）
□ 是否有防错设计？
□ 错误信息是否有助于模型纠正？
□ 是否测试过模型如何使用这个工具？
```

**错误信息的重要性**：

```python
# ❌ 不好的错误信息
def execute(command):
    try:
        return run(command)
    except Exception:
        return "Error"  # 模型不知道哪里错了

# ✅ 好的错误信息
def execute(command):
    try:
        return run(command)
    except FileNotFoundError:
        return f"错误：找不到命令 '{command}'。请检查是否已安装。"
    except PermissionError:
        return f"错误：没有权限执行 '{command}'。请尝试其他方法。"
    except Exception as e:
        return f"错误：{type(e).__name__}: {str(e)}。请分析错误并尝试修复。"
```

---

## 第五部分：实践 —— 手写一个简单的 Agent

### 5.1 最简 Agent 循环

让我们从零开始，手写一个最小化的 Agent 循环。

**目标**：实现一个能回答"北京今天天气"的简单 Agent。

```python
"""
最简 Agent 循环 - ReAct 模式实现

核心流程：
1. Thought（思考）：模型分析当前状态，决定下一步
2. Action（行动）：模型选择并调用工具
3. Observation（观察）：获取工具执行结果
4. 循环直到任务完成
"""

import json

# ===== 第一步：定义工具 =====

tools = [
    {
        "name": "search",
        "description": "搜索互联网获取信息。参数：query（搜索关键词）",
        "function": None  # 实际实现中这里会是真实的搜索函数
    },
    {
        "name": "finish",
        "description": "完成任务，返回最终答案。参数：answer（最终答案）",
        "function": None
    }
]

def mock_search(query):
    """模拟搜索工具"""
    if "天气" in query and "北京" in query:
        return "北京今天晴天，最高温度28°C，空气质量良好"
    return f"搜索结果：{query}"

def mock_llm(prompt):
    """模拟 LLM 响应"""
    # 在实际实现中，这里会调用真实的 LLM API
    pass

# ===== 第二步：实现 ReAct 循环 =====

def agent_loop(user_input, max_iterations=10):
    """
    Agent 主循环
    
    参数：
        user_input: 用户输入
        max_iterations: 最大迭代次数，防止无限循环
    """
    # 初始化上下文
    context = f"用户问题：{user_input}\n\n"
    
    for i in range(max_iterations):
        print(f"\n=== 迭代 {i+1} ===")
        
        # 构建提示词
        prompt = build_prompt(context)
        
        # 获取模型响应（思考 + 行动）
        response = call_llm(prompt)
        
        # 解析响应
        thought, action, action_input = parse_response(response)
        
        print(f"思考：{thought}")
        print(f"行动：{action}({action_input})")
        
        # 执行行动
        if action == "finish":
            print(f"\n最终答案：{action_input}")
            return action_input
        elif action == "search":
            observation = mock_search(action_input)
        else:
            observation = f"未知工具：{action}"
        
        print(f"观察：{observation}")
        
        # 更新上下文
        context += f"""
思考：{thought}
行动：{action}({action_input})
观察：{observation}
"""
    
    return "达到最大迭代次数，任务未完成"

# ===== 第三步：构建提示词 =====

def build_prompt(context):
    """
    构建 ReAct 风格的提示词
    """
    return f"""你是一个智能助手，可以通过搜索获取信息来回答问题。

可用工具：
- search(query): 搜索互联网获取信息
- finish(answer): 完成任务，返回最终答案

请按以下格式思考和行动：

思考：分析当前情况，决定下一步做什么
行动：工具名(参数)

{context}
思考："""

def parse_response(response):
    """
    解析模型响应，提取思考和行动
    """
    # 简化的解析逻辑
    # 实际实现中需要更健壮的解析
    lines = response.strip().split('\n')
    thought = ""
    action = ""
    action_input = ""
    
    for line in lines:
        if line.startswith("思考：") or line.startswith("Thought:"):
            thought = line.split("：", 1)[1].strip()
        elif line.startswith("行动：") or line.startswith("Action:"):
            action_part = line.split("：", 1)[1].strip()
            # 解析 action(input) 格式
            if "(" in action_part:
                action = action_part.split("(")[0].strip()
                action_input = action_part.split("(")[1].rstrip(")").strip()
    
    return thought, action, action_input

def call_llm(prompt):
    """
    调用 LLM（模拟）
    在实际实现中，这里会调用 OpenAI、Claude 等 API
    """
    # 这里我们模拟一个简单的响应
    # 实际实现时替换为真实的 API 调用
    pass

# ===== 运行示例 =====

if __name__ == "__main__":
    result = agent_loop("北京今天天气怎么样？")
```

### 5.2 完整可运行示例

让我们写一个真正可以运行的版本：

```python
"""
完整可运行的 ReAct Agent 示例

这个示例使用 OpenAI API，你可以替换为其他模型
运行前请设置：export OPENAI_API_KEY=your_key
"""

import os
import re
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== 工具定义 =====

TOOLS = {
    "search": {
        "description": "搜索互联网获取实时信息",
        "params": ["query"],
        "handler": lambda query: f"搜索 '{query}' 的结果：[模拟搜索结果]"
    },
    "calculate": {
        "description": "执行数学计算",
        "params": ["expression"],
        "handler": lambda expr: str(eval(expr))
    },
    "finish": {
        "description": "完成任务，返回最终答案",
        "params": ["answer"],
        "handler": None  # 特殊处理
    }
}

# ===== 提示词模板 =====

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来完成任务。

可用工具：
{tools_desc}

请按以下格式思考和行动：

思考：分析当前情况，决定下一步
行动：工具名(参数)

你可以多次思考和行动，直到获得足够信息完成任务。
最后使用 finish(答案) 返回最终结果。
"""

def get_tools_description():
    """生成工具描述"""
    lines = []
    for name, info in TOOLS.items():
        params = ", ".join(info["params"])
        lines.append(f"- {name}({params}): {info['description']}")
    return "\n".join(lines)

def extract_action(text):
    """从响应中提取行动"""
    # 匹配 "行动：tool(param)" 或 "Action: tool(param)"
    pattern = r'(?:行动|Action)[:：]\s*(\w+)\(([^)]*)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_agent(task, max_steps=10):
    """运行 Agent"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            tools_desc=get_tools_description()
        )},
        {"role": "user", "content": task}
    ]
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        
        assistant_message = response.choices[0].message.content
        print(f"Assistant: {assistant_message}")
        
        # 提取行动
        action, param = extract_action(assistant_message)
        
        if not action:
            messages.append({"role": "assistant", "content": assistant_message})
            continue
        
        # 检查是否完成
        if action == "finish":
            print(f"\n✅ 任务完成！答案：{param}")
            return param
        
        # 执行工具
        if action in TOOLS:
            result = TOOLS[action]["handler"](param)
            print(f"工具结果：{result}")
            
            # 添加到对话历史
            messages.append({"role": "assistant", "content": assistant_message})
            messages.append({"role": "user", "content": f"观察：{result}"})
        else:
            messages.append({"role": "assistant", "content": assistant_message})
            messages.append({"role": "user", "content": f"错误：未知工具 {action}"})
    
    print("\n⚠️ 达到最大步数，任务可能未完成")
    return None

# ===== 测试 =====

if __name__ == "__main__":
    # 测试 1：简单搜索
    print("=" * 50)
    print("测试 1：天气查询")
    print("=" * 50)
    run_agent("北京今天天气怎么样？")
    
    # 测试 2：计算任务
    print("\n" + "=" * 50)
    print("测试 2：数学计算")
    print("=" * 50)
    run_agent("计算 123 * 456 等于多少？")
```

### 5.3 关键代码解析

让我们深入理解这个实现：

#### 1. 工具定义模式

```python
TOOLS = {
    "tool_name": {
        "description": "工具描述",
        "params": ["参数列表"],
        "handler": lambda x: f"处理结果: {x}"
    }
}
```

这个模式让工具可扩展：
- 添加新工具只需在字典中添加条目
- 工具描述会自动注入到提示词
- 处理函数统一调用

#### 2. 提示词设计

```python
SYSTEM_PROMPT = """
请按以下格式思考和行动：

思考：分析当前情况，决定下一步
行动：工具名(参数)
"""
```

这体现了 **ReAct 模式** 的核心：
- Thought：让模型"显式思考"
- Action：指导模型采取行动
- 格式化输出便于解析

#### 3. 循环控制

```python
for step in range(max_steps):
    # 1. 调用 LLM
    # 2. 解析行动
    # 3. 执行工具
    # 4. 更新上下文
    # 5. 检查是否完成
```

这个循环是 Agent 的"心脏"：
- 防止无限循环（max_steps）
- 维护对话历史（messages）
- 检测终止条件（finish 动作）

---

## 第六部分：深入理解 —— 为什么 ReAct 有效？

### 6.1 ReAct 论文的核心发现

**论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

**核心思想**：推理（Reasoning）和行动（Acting）交替进行

```
传统方法：
[问题] → [直接回答] 
问题：容易产生幻觉

Chain-of-Thought：
[问题] → [思考链] → [答案]
优点：推理更可靠
问题：没有外部信息，仍可能幻觉

ReAct：
[问题] → [思考] → [行动] → [观察] → [思考] → ... → [答案]
优点：结合推理和外部信息
```

### 6.2 ReAct 的优势

1. **减少幻觉**
   - 通过行动获取真实信息
   - 不是"瞎猜"而是"查证"

2. **更好的可解释性**
   - 思考轨迹可见
   - 知道 Agent 做了什么、为什么

3. **更灵活的问题解决**
   - 可以根据观察调整策略
   - 不是一次性输出

### 6.3 ReAct 的局限

1. **多轮交互成本**
   - 每一步都需要 LLM 调用
   - 延迟和成本增加

2. **可能的发散**
   - Agent 可能"跑偏"
   - 需要良好的终止条件

3. **上下文累积**
   - 长对话会撑爆 Context
   - 需要压缩策略

---

## 实践练习

### 练习 1：扩展工具集

**任务**：为上面的 Agent 添加两个新工具

1. `read_file(path)`: 读取本地文件
2. `write_file(path, content)`: 写入文件

**提示**：
- 更新 TOOLS 字典
- 添加处理函数
- 测试：让 Agent 读取一个文件并总结

### 练习 2：实现记忆系统

**任务**：让 Agent 能记住之前对话的关键信息

**要求**：
1. 创建一个 memory 字典存储关键信息
2. 在提示词中注入记忆
3. 让 Agent 能更新记忆

**提示**：
```python
memory = {}

def update_memory(key, value):
    memory[key] = value
    # 持久化到文件
    with open("memory.json", "w") as f:
        json.dump(memory, f)

def load_memory():
    global memory
    try:
        with open("memory.json") as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = {}
```

### 练习 3：分析一个真实 Agent

**任务**：找一个开源 Agent 项目，分析它的架构

建议项目：
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)
- [OpenAI Assistants](https://platform.openai.com/docs/assistants/overview)

分析要点：
1. 它如何定义工具？
2. 它的 Agent 循环是怎样的？
3. 它如何管理 Context？

---

## 今日总结

### 核心概念回顾

```
Agent = Model + Harness

Model: 有智能，但无行动能力
Harness: 给模型"身体"，让它能与世界交互

Workflow vs Agent:
- Workflow: 预定义路径，可预测
- Agent: 动态决策，灵活但不可预测

Harness 核心组件:
- Tool System: 与世界交互的接口
- Memory: 跨会话保持状态
- Filesystem: 持久化数据
- Sandbox: 安全执行
- Context Management: 管理 Context Window
- Safety: 安全约束

ReAct 模式:
思考 → 行动 → 观察 → 循环
```

### 关键洞察

1. **工具设计 = 提示词工程**：不要忽视工具定义的重要性
2. **从简单开始**：先跑通最小循环，再逐步优化
3. **ACI 设计**：像重视 HCI 一样重视 Agent-计算机交互

### 明日预告

Day 2 我们将深入学习 **工作流模式（Workflows）**：
- Prompt Chaining（提示链）
- Routing（路由）
- Parallelization（并行化）
- Orchestrator-Workers（编排者-工作者）
- Evaluator-Optimizer（评估者-优化者）

---

## 阅读材料

### 必读

1. **[LangChain: The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/)**
   - 理解 Harness 的核心概念
   - 为什么模型需要"外套"

2. **[Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)**
   - Workflow vs Agent 的区分
   - 工具设计原则

### 选读

3. **[ReAct 论文](https://arxiv.org/abs/2210.03629)**
   - ReAct 模式的原始论文
   - 实验设计和结果分析

4. **[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)**
   - 工具调用的技术实现
   - API 使用方法

---

## 常见问题

### Q1: Agent 和普通 LLM 应用的区别是什么？

**A**: 普通 LLM 应用通常是一次性问答，而 Agent 是一个循环系统，能够：
- 调用工具获取信息
- 根据结果调整策略
- 多轮交互直到完成任务

### Q2: 什么时候应该用 Agent 而不是 Workflow？

**A**: 问自己：
- 能否预测任务的所有步骤？→ Workflow
- 步骤数和类型不确定？→ Agent
- 需要高可靠性？→ Workflow
- 需要灵活性？→ Agent

### Q3: ReAct 模式的"思考"有什么用？

**A**: "思考"步骤的价值：
1. **可解释性**：你能看到 Agent 的推理过程
2. **调试**：出错时能追踪到具体步骤
3. **质量**：强制模型"显式思考"比直接输出更可靠
4. **上下文**：思考成为下一步的上下文

### Q4: 工具越多越好吗？

**A**: 不是。工具过多的问题：
- 工具描述占用 Context
- 模型选择困难
- 可能选择不相关工具

**最佳实践**：
- 按需加载工具
- 相关工具分组
- 定期清理无用工具

### Q5: 如何防止 Agent 无限循环？

**A**: 多层保护：
1. **最大迭代次数**：硬性限制
2. **重复检测**：检测相同的行动
3. **进度评估**：让模型评估是否在进步
4. **用户确认**：长时间任务请求用户确认

### Q6: Context 窗口满了怎么办？

**A**: 策略组合：
1. **Compaction**：智能摘要历史
2. **截断**：只保留最近 N 轮
3. **文件存储**：重要信息存文件
4. **渐进披露**：只加载必要工具

---

## 扩展阅读：真实世界的 Agent 案例

### 案例 1：Claude Code

Claude Code 是 Anthropic 的编程 Agent：

**核心设计**：
- 工具：read_file, write_file, edit_file, exec_bash
- 文件系统：持久化工作进度
- 安全：沙盒执行代码
- 规划：PLAN.md 跟踪任务

**关键洞察**：
- 工具设计花了大量时间优化
- 绝对路径避免了很多错误
- 规划文件帮助跨越 Context 窗口

### 案例 2：AutoGPT

AutoGPT 是早期的自主 Agent：

**核心设计**：
- 目标分解：将大目标分解为子任务
- 记忆：本地存储对话历史
- 自我反思：评估自己的输出

**教训**：
- 容易"跑偏"，需要人工监督
- Context 限制是主要瓶颈
- 简单任务用简单方法更好

### 案例 3：SWE-bench Agent

Anthropic 的 SWE-bench Agent 解决 GitHub Issue：

**核心设计**：
- Bash 工具：运行命令和测试
- 文件编辑：精确定位修改
- 自我验证：运行测试确认修复

**工具优化历程**：
- 相对路径 → 绝对路径（减少路径错误）
- 行号编辑 → 文本匹配编辑（减少计算错误）
- 模糊描述 → 详细说明（减少误用）

---

## 动手挑战

### 挑战 1：实现一个问答 Agent

**目标**：创建一个能回答问题的 Agent，至少支持：
- 搜索工具（可以模拟）
- 计算工具
- 完成动作

**验收标准**：
1. 能正确处理"今天天气"类问题
2. 能正确处理"123*456"类计算
3. 有清晰的思考和行动日志

### 挑战 2：添加记忆功能

**目标**：让 Agent 记住用户说过的话

**提示**：
```python
class MemoryAgent:
    def __init__(self):
        self.memory = {}
        self.memory_file = "agent_memory.json"
        self.load_memory()
    
    def remember(self, key, value):
        self.memory[key] = value
        self.save_memory()
    
    def recall(self, key):
        return self.memory.get(key)
    
    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)
    
    def load_memory(self):
        try:
            with open(self.memory_file) as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = {}
```

### 挑战 3：分析你的 Agent 足迹

**目标**：记录 Agent 的每一步，分析效率

**要求**：
1. 记录每一步的时间戳
2. 记录每个工具调用的结果
3. 统计：平均步数、最常用工具、失败率

---

## 今日作业

### 必做

1. **运行示例代码**：确保理解 ReAct 循环
2. **回答问题**：用自己的话解释 Agent = Model + Harness

### 选做

3. **扩展 Agent**：添加一个新工具
4. **记录学习笔记**：总结今天学到的 3 个最重要的概念

### 进阶

5. **阅读论文**：浏览 ReAct 论文的 Introduction 和 Method 部分
6. **代码重构**：将示例代码改造成更模块化的结构

---

*Day 1 完成！明天我们将深入工作流模式。*
-