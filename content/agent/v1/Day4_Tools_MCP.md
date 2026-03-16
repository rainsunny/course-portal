# Day 4: 工具与 MCP

> **学习时长**: 2-3 小时
> **核心问题**: 如何设计和实现高质量的 Agent 工具？

---

## 学习目标

1. **理解工具本质** - 工具是 Agent 与世界的接口
2. **掌握设计原则** - ACI 设计，防错设计
3. **了解 MCP 协议** - 模型上下文协议
4. **实现自定义工具** - 动手实现高质量工具

---

## 第一部分：工具的本质

### 1.1 为什么工具设计很重要？

> **Anthropic 的观点**：
> 工具定义应该获得与提示词工程同等的关注。
> 在构建 SWE-bench Agent 时，我们花在优化工具上的时间比优化整体提示词还多。

### 1.2 工具是 Agent 的"手"

```
Model（大脑）→ 有智能，但无法行动
    ↓
Tool（手）→ 让智能能够作用于世界
```

工具定义了 Agent 能做什么、怎么做。

---

## 第二部分：工具设计原则

### 原则一：给模型足够的思考空间

```python
# ❌ 不好：让模型计算行号
{
    "name": "edit_file",
    "parameters": {
        "line_start": {"type": "integer", "description": "起始行号"},
        "line_end": {"type": "integer", "description": "结束行号"},
        "content": {"type": "string"}
    }
}
# 模型需要：打开文件 → 数行号 → 编辑

# ✅ 好：用文本匹配
{
    "name": "edit_file",
    "parameters": {
        "old_text": {"type": "string", "description": "要替换的文本（必须完全匹配）"},
        "new_text": {"type": "string", "description": "替换后的文本"}
    }
}
# 模型只需要：复制要改的内容 → 粘贴 → 修改
```

### 原则二：减少格式开销

```python
# ❌ 不好：需要复杂转义
{
    "name": "write_code",
    "parameters": {
        "code": {"type": "string", "description": "代码（需要转义引号、换行）"}
    }
}

# ✅ 好：直接传递文件路径
{
    "name": "write_file",
    "parameters": {
        "path": {"type": "string", "description": "文件路径"},
        "content": {"type": "string", "description": "文件内容（原始文本）"}
    }
}
```

### 原则三：防错设计（Poka-yoke）

```python
# ❌ 不好：相对路径容易出错
{
    "name": "read_file",
    "parameters": {
        "path": {"type": "string", "description": "文件路径（相对于工作目录）"}
    }
}
# 问题：Agent 切换目录后路径失效

# ✅ 好：强制绝对路径
{
    "name": "read_file",
    "parameters": {
        "path": {
            "type": "string",
            "description": "文件的绝对路径（必须以 / 开头）"
        }
    }
}
```

### 原则四：提供有用的错误信息

```python
# ❌ 不好：无用的错误
def execute(command):
    try:
        return run(command)
    except Exception:
        return "Error"

# ✅ 好：有帮助的错误
def execute(command):
    try:
        return run(command)
    except FileNotFoundError:
        return f"错误：找不到命令 '{command}'。请检查是否已安装。"
    except PermissionError:
        return f"错误：没有权限执行 '{command}'。尝试使用 sudo。"
    except Exception as e:
        return f"错误：{type(e).__name__}: {str(e)}"
```

---

## 第三部分：ACI 设计

### Agent-Computer Interface

类比 HCI（人机交互），ACI 关注 Agent-计算机交互。

**HCI 检查清单** → **ACI 检查清单**：

| HCI 关注 | ACI 关注 |
|----------|----------|
| 用户界面是否直观？ | 工具描述是否清晰？ |
| 用户能理解如何操作吗？ | 模型能理解如何使用吗？ |
| 有帮助提示吗？ | 有使用示例吗？ |
| 错误信息有帮助吗？ | 错误信息能指导纠正吗？ |

### ACI 设计检查清单

```
□ 站在模型角度：工具描述是否清晰？
□ 参数名和描述是否足够明显？（像给初级开发者写文档）
□ 是否有防错设计？
□ 错误信息是否有助于模型纠正？
□ 是否测试过模型如何使用你的工具？
□ 边界情况是否考虑周全？
```

---

## 第四部分：案例研究 - SWE-bench Agent

### 问题：文件编辑工具经常出错

**错误分析**：
- 路径错误：45%
- 行号计算错误：30%
- 格式问题：15%

### 解决方案

1. **强制绝对路径**
```python
def validate_path(path: str) -> str:
    if not path.startswith('/'):
        raise ValueError(f"路径必须是绝对路径（以 / 开头），收到：{path}")
    return path
```

2. **文本匹配替代行号**
```python
def edit_file(path: str, old_text: str, new_text: str):
    """精确编辑文件
    
    参数：
        path: 文件的绝对路径
        old_text: 要替换的文本（必须与文件内容完全匹配）
        new_text: 替换后的文本
    
    注意：
        - old_text 必须完全匹配，包括空格和缩进
        - 如果 old_text 出现多次，会替换所有出现
        - 建议先读取文件确认内容，再进行编辑
    """
    pass
```

3. **详细的使用说明**
```python
"""
精确编辑文件工具

使用方法：
1. 首先使用 read_file 读取文件内容
2. 从读取的内容中复制要修改的部分（确保完全匹配）
3. 修改复制的内容得到 new_text
4. 使用 edit_file 进行编辑

示例：
    # 读取文件
    content = read_file("/home/user/app.py")
    
    # 假设内容是：
    # def hello():
    #     print("world")
    
    # 编辑：将 "world" 改为 "hello"
    edit_file(
        path="/home/user/app.py",
        old_text='print("world")',
        new_text='print("hello")'
    )

常见错误：
- old_text 不匹配（注意空格、缩进、换行）
- 路径不是绝对路径
"""
```

### 结果

工具使用成功率从 70% 提升到 98%。

---

## 第五部分：MCP 协议

### 什么是 MCP？

**Model Context Protocol (MCP)** 是 Anthropic 开源的标准协议，用于连接 AI 系统与数据源。

### 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  AI Client  │ ←→  │ MCP Server  │ ←→  │  Data Source│
│  (Claude)   │     │             │     │ (GitHub,    │
└─────────────┘     └─────────────┘     │  Drive, DB) │
                                        └─────────────┘
```

### 核心概念

1. **MCP Server**：暴露数据或功能
2. **MCP Client**：连接到 Server 的 AI 应用
3. **Resources**：可读的数据
4. **Tools**：可执行的操作
5. **Prompts**：预定义的提示模板

### 实现 MCP Server

```python
from mcp import MCPServer, Tool, Resource

# 创建 Server
server = MCPServer("my-tools")

# 定义工具
@server.tool()
def search_web(query: str) -> str:
    """搜索互联网获取信息
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果摘要
    """
    # 实现搜索逻辑
    return f"搜索结果：{query}"

@server.tool()
def read_file(path: str) -> str:
    """读取文件内容
    
    Args:
        path: 文件的绝对路径
    
    Returns:
        文件内容
    """
    with open(path, 'r') as f:
        return f.read()

# 定义资源
@server.resource("file://{path}")
def file_resource(path: str) -> str:
    """文件资源"""
    return read_file(path)

# 启动 Server
server.start()
```

### 现有 MCP 服务器

- Google Drive
- Slack
- GitHub
- Git
- Postgres
- Puppeteer

---

## 实践练习

### 练习 1：设计文件操作工具集

设计一套文件操作工具：
- read_file
- write_file
- edit_file
- list_files
- delete_file

要求：
- 遵循 ACI 设计原则
- 有完善的错误处理
- 包含使用示例

### 练习 2：实现一个简单的 MCP Server

实现一个 MCP Server，提供：
- 搜索工具
- 文件读写工具
- 计算工具

### 练习 3：工具使用测试

设计测试用例，验证工具设计是否合理：
- 正常使用场景
- 边界情况
- 错误处理

---

## 今日总结

### 核心原则

```
1. 给模型思考空间 - 不要让模型做格式计算
2. 减少格式开销 - 让模型专注于内容
3. 防错设计 - 让错误更难发生
4. 有用错误信息 - 指导模型纠正
```

### ACI vs HCI

| HCI | ACI |
|-----|-----|
| 用户界面 | 工具接口 |
| 用户体验 | Agent 体验 |
| 直观设计 | 清晰描述 |

### 明日预告

Day 5：Memory 与 Context 管理 —— 如何让 Agent 记住和遗忘。

---

## 阅读材料

1. [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Model Context Protocol](https://modelcontextprotocol.io/)
3. [Anthropic: Tool Use](https://docs.anthropic.com/claude/docs/tool-use)

---

*Day 4 完成！明天我们将深入 Memory 与 Context 管理。*
