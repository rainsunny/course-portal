# Day 7: 实战项目与最佳实践

> **学习时长**: 2-3 小时
> **核心问题**: 如何综合运用所学，构建生产级 Agent？

---

## 学习目标

1. **综合运用知识** - 完成一个完整项目
2. **掌握最佳实践** - Anthropic 的 Agent 开发原则
3. **了解调试技巧** - 如何调试 Agent
4. **展望未来** - Agent 技术的发展趋势

---

## 第一部分：项目选择

### 项目选项

#### 项目 A：代码助手 Agent

**功能**：
- 读取代码仓库
- 理解代码结构
- 执行修改并验证

**技术要点**：
- 文件系统操作
- 代码解析
- 测试执行

#### 项目 B：研究助手 Agent

**功能**：
- 搜索网络信息
- 提取关键内容
- 生成研究报告

**技术要点**：
- 搜索工具
- 网页抓取
- 内容整合

#### 项目 C：客服 Agent

**功能**：
- 路由不同类型问题
- 访问知识库
- 执行操作（退款、下单等）

**技术要点**：
- 路由系统
- 知识库检索
- 多 Agent 协作

---

## 第二部分：项目实战 —— 代码助手

我们以代码助手为例，展示完整开发流程。

### 2.1 需求分析

```
功能需求：
1. 理解用户请求
2. 分析代码仓库
3. 定位修改位置
4. 执行修改
5. 运行测试验证
6. 输出结果

非功能需求：
1. 安全：不执行危险操作
2. 可靠：验证修改结果
3. 可解释：展示决策过程
```

### 2.2 架构设计

```
┌─────────────────────────────────────────────────┐
│              Code Assistant Agent               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Analyzer│  │  Coder  │  │ Tester  │         │
│  │  Agent  │  │  Agent  │  │  Agent  │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
│       │            │            │              │
│       └────────────┼────────────┘              │
│                    │                           │
│              ┌─────┴─────┐                     │
│              │  Toolkit  │                     │
│              │ - read    │                     │
│              │ - write   │                     │
│              │ - edit    │                     │
│              │ - execute │                     │
│              └───────────┘                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 2.3 核心实现

```python
"""
代码助手 Agent 完整实现
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import os
import subprocess

# ============================================================
# 工具系统
# ============================================================

class CodeToolkit:
    """代码工具集"""
    
    def __init__(self, workspace: str):
        self.workspace = workspace
    
    def read_file(self, path: str) -> str:
        """读取文件"""
        full_path = os.path.join(self.workspace, path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_file(self, path: str, content: str):
        """写入文件"""
        full_path = os.path.join(self.workspace, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def edit_file(self, path: str, old_text: str, new_text: str) -> bool:
        """编辑文件（文本匹配）"""
        content = self.read_file(path)
        if old_text not in content:
            return False
        new_content = content.replace(old_text, new_text)
        self.write_file(path, new_content)
        return True
    
    def list_files(self, directory: str = "") -> List[str]:
        """列出文件"""
        full_path = os.path.join(self.workspace, directory)
        files = []
        for root, dirs, filenames in os.walk(full_path):
            for filename in filenames:
                rel_path = os.path.relpath(
                    os.path.join(root, filename), 
                    self.workspace
                )
                files.append(rel_path)
        return files
    
    def execute(self, command: str) -> Dict:
        """执行命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out"
            }
    
    def run_tests(self, test_command: str = "pytest") -> Dict:
        """运行测试"""
        return self.execute(test_command)


# ============================================================
# Agent 核心
# ============================================================

class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    CODING = "coding"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Agent 结果"""
    success: bool
    message: str
    changes: List[Dict]
    test_results: Optional[Dict] = None


class CodeAssistantAgent:
    """代码助手 Agent"""
    
    SYSTEM_PROMPT = """你是一个代码助手，帮助用户理解和修改代码。

## 可用工具

1. read_file(path) - 读取文件内容
2. write_file(path, content) - 写入文件
3. edit_file(path, old_text, new_text) - 编辑文件（精确匹配替换）
4. list_files(directory) - 列出目录文件
5. execute(command) - 执行命令
6. run_tests() - 运行测试

## 工作流程

1. 理解用户请求
2. 分析相关代码
3. 制定修改计划
4. 执行修改
5. 运行测试验证
6. 汇报结果

## 安全规则

- 只修改用户明确指定的文件
- 不执行危险命令（rm -rf, sudo 等）
- 修改前先备份重要文件
- 每次修改后验证结果

## 重要原则

- 使用绝对路径
- edit_file 的 old_text 必须完全匹配
- 修改前先读取确认
"""
    
    def __init__(self, workspace: str, llm_client=None):
        self.workspace = workspace
        self.toolkit = CodeToolkit(workspace)
        self.llm = llm_client
        self.state = AgentState.IDLE
        self.history: List[Dict] = []
    
    def process(self, request: str) -> AgentResult:
        """处理用户请求"""
        self.state = AgentState.ANALYZING
        self.history = []
        changes = []
        
        try:
            # 1. 理解请求
            plan = self._understand(request)
            self.history.append({"step": "understand", "plan": plan})
            
            # 2. 分析代码
            self.state = AgentState.ANALYZING
            analysis = self._analyze(plan)
            self.history.append({"step": "analyze", "analysis": analysis})
            
            # 3. 执行修改
            self.state = AgentState.CODING
            for modification in plan.get("modifications", []):
                change = self._modify(modification)
                if change:
                    changes.append(change)
            self.history.append({"step": "modify", "changes": changes})
            
            # 4. 运行测试
            self.state = AgentState.TESTING
            if plan.get("run_tests", True):
                test_results = self.toolkit.run_tests()
                self.history.append({"step": "test", "results": test_results})
                
                if not test_results["success"]:
                    self.state = AgentState.FAILED
                    return AgentResult(
                        success=False,
                        message="测试失败",
                        changes=changes,
                        test_results=test_results
                    )
            
            # 5. 完成
            self.state = AgentState.COMPLETED
            return AgentResult(
                success=True,
                message="任务完成",
                changes=changes
            )
            
        except Exception as e:
            self.state = AgentState.FAILED
            return AgentResult(
                success=False,
                message=f"执行失败：{str(e)}",
                changes=changes
            )
    
    def _understand(self, request: str) -> Dict:
        """理解请求，制定计划"""
        # 实际实现中调用 LLM
        # 这里返回模拟计划
        return {
            "goal": request,
            "modifications": [],
            "run_tests": True
        }
    
    def _analyze(self, plan: Dict) -> Dict:
        """分析代码"""
        # 列出相关文件
        files = self.toolkit.list_files()
        return {
            "files": files,
            "relevant_files": []
        }
    
    def _modify(self, modification: Dict) -> Optional[Dict]:
        """执行修改"""
        action = modification.get("action")
        
        if action == "edit":
            success = self.toolkit.edit_file(
                modification["path"],
                modification["old_text"],
                modification["new_text"]
            )
            return {
                "action": "edit",
                "path": modification["path"],
                "success": success
            }
        
        elif action == "write":
            self.toolkit.write_file(
                modification["path"],
                modification["content"]
            )
            return {
                "action": "write",
                "path": modification["path"],
                "success": True
            }
        
        return None


# ============================================================
# 使用示例
# ============================================================

def example_usage():
    """使用示例"""
    # 创建 Agent
    agent = CodeAssistantAgent(workspace="/path/to/project")
    
    # 处理请求
    result = agent.process("修复 login.py 中的 token 过期问题")
    
    # 输出结果
    if result.success:
        print(f"✅ {result.message}")
        print(f"修改：{result.changes}")
    else:
        print(f"❌ {result.message}")


if __name__ == "__main__":
    example_usage()
```

### 2.4 测试验证

```python
"""
测试代码助手 Agent
"""

import pytest
import tempfile
import os

class TestCodeAssistantAgent:
    
    @pytest.fixture
    def agent(self):
        """创建测试 Agent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, 'w') as f:
                f.write("def hello():\n    print('world')\n")
            
            agent = CodeAssistantAgent(tmpdir)
            yield agent
    
    def test_read_file(self, agent):
        """测试文件读取"""
        content = agent.toolkit.read_file("test.py")
        assert "hello" in content
    
    def test_edit_file(self, agent):
        """测试文件编辑"""
        success = agent.toolkit.edit_file(
            "test.py",
            "print('world')",
            "print('hello')"
        )
        assert success
        
        content = agent.toolkit.read_file("test.py")
        assert "hello" in content
    
    def test_execute(self, agent):
        """测试命令执行"""
        result = agent.toolkit.execute("echo test")
        assert result["success"]
        assert "test" in result["stdout"]
```

---

## 第三部分：最佳实践

### 3.1 Anthropic 的 Agent 开发原则

#### 原则一：保持简单

```
开始：
┌─────────────────┐
│  简单 Prompt    │
└─────────────────┘
        ↓ 测试
┌─────────────────┐
│  添加工具       │
└─────────────────┘
        ↓ 测试
┌─────────────────┐
│  复杂 Agent     │
└─────────────────┘

而不是：
开始：
┌─────────────────┐
│  复杂多 Agent   │
└─────────────────┘
（难以调试，成本高）
```

#### 原则二：透明度优先

```python
# ✅ 好：展示决策过程
def process(self, request: str):
    print(f"[理解] 用户请求：{request}")
    plan = self._understand(request)
    print(f"[规划] 执行计划：{plan}")
    
    print(f"[分析] 分析代码...")
    analysis = self._analyze(plan)
    print(f"[分析] 发现：{analysis}")
    
    print(f"[执行] 修改文件...")
    changes = self._modify(plan)
    print(f"[执行] 完成：{changes}")

# ❌ 不好：黑盒执行
def process(self, request: str):
    return self.llm.call(request)
```

#### 原则三：精心设计 ACI

```
ACI 设计检查清单：
□ 工具描述是否清晰？
□ 参数说明是否详细？
□ 是否有防错设计？
□ 错误信息是否有帮助？
□ 是否测试过模型使用？
```

### 3.2 调试技巧

#### 技巧一：查看完整轨迹

```python
class DebugAgent:
    """带调试的 Agent"""
    
    def __init__(self, agent):
        self.agent = agent
        self.trace: List[Dict] = []
    
    def run(self, request: str):
        """运行并记录轨迹"""
        # 记录每一步
        for step in self.agent.run_steps(request):
            self.trace.append({
                "step": step.name,
                "input": step.input,
                "output": step.output,
                "thoughts": step.thoughts
            })
            print(f"\n[{step.name}]")
            print(f"输入：{step.input}")
            print(f"思考：{step.thoughts}")
            print(f"输出：{step.output[:200]}...")
        
        return self.trace[-1]["output"]
```

#### 技巧二：隔离问题

```python
def diagnose(self, request: str):
    """诊断问题"""
    # 1. 测试工具
    print("=== 测试工具 ===")
    for tool_name, tool in self.tools.items():
        try:
            result = tool.test()
            print(f"✅ {tool_name}: OK")
        except Exception as e:
            print(f"❌ {tool_name}: {e}")
    
    # 2. 测试 Prompt
    print("\n=== 测试 Prompt ===")
    prompt = self.build_prompt(request)
    print(f"Prompt 长度：{len(prompt)}")
    print(f"Prompt 预览：{prompt[:500]}...")
    
    # 3. 测试 LLM
    print("\n=== 测试 LLM ===")
    try:
        response = self.llm.call("测试：返回 OK")
        print(f"LLM 响应：{response}")
    except Exception as e:
        print(f"LLM 错误：{e}")
```

#### 技巧三：A/B 测试工具

```python
def ab_test_tools(self, request: str):
    """A/B 测试工具定义"""
    
    # 版本 A
    tools_a = {
        "edit": {
            "description": "编辑文件",
            "parameters": {"path": "文件路径", "text": "文本"}
        }
    }
    
    # 版本 B
    tools_b = {
        "edit": {
            "description": "精确编辑文件。先读取文件，复制要修改的部分，然后修改。",
            "parameters": {
                "path": {"type": "string", "description": "文件的绝对路径"},
                "old_text": {"type": "string", "description": "要替换的原文（必须完全匹配）"},
                "new_text": {"type": "string", "description": "替换后的文本"}
            }
        }
    }
    
    # 测试
    results_a = self.run_with_tools(request, tools_a)
    results_b = self.run_with_tools(request, tools_b)
    
    return {
        "a": {"success_rate": results_a.success_rate},
        "b": {"success_rate": results_b.success_rate}
    }
```

---

## 第四部分：未来趋势

### 4.1 模型与 Harness 的共同进化

```
趋势：
- Claude Code、Codex 在训练中包含 Harness
- 模型越来越擅长 Harness 中的操作
- 但也可能导致对特定 Harness 的过拟合

启示：
- 保持架构灵活
- 不要过度依赖特定实现
- 关注模型能力提升
```

### 4.2 Harness 的演进方向

```
未来 Harness：
┌─────────────────────────────────────────┐
│ - 数百个 Agent 并行工作                 │
│ - 共享代码库                            │
│ - 自我分析 trace                        │
│ - 动态组装工具和 Context                │
└─────────────────────────────────────────┘
```

---

## 实践练习

### 练习 1：完成代码助手

完善代码助手 Agent：
- 添加 LLM 集成
- 实现完整的理解-规划-执行循环
- 添加测试用例

### 练习 2：调试练习

给定一个有问题的 Agent：
- 分析问题原因
- 修复问题
- 优化工具定义

### 练习 3：项目改进

改进你的项目：
- 添加记忆系统
- 实现规划文件
- 添加自我验证

---

## 课程总结

### 核心要点回顾

```
Day 1: Agent = Model + Harness
Day 2: 五种工作流模式
Day 3: ReAct 循环 + 规划 + 验证
Day 4: 工具设计 = 提示词工程
Day 5: Memory 系统 + Context 管理
Day 6: 多 Agent 架构
Day 7: 实战 + 最佳实践
```

### 关键原则

1. **从简单开始**：不要一开始就构建复杂系统
2. **理解本质**：Agent = Model + Harness
3. **选择正确模式**：Workflow vs Agent 取决于任务
4. **重视工具设计**：ACI 与 HCI 同等重要
5. **管理 Context**：Context Rot 是真实挑战
6. **评估驱动**：用数据指导改进

### 成功公式

```
成功的 Agent = 清晰的目标 + 简单的设计 + 精心设计的工具 + 充分的测试
```

> 成功不在于构建最复杂的系统，而在于构建适合你需求的系统。
> 
> —— Anthropic

---

## 进一步学习

### 推荐阅读

1. [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [LangChain: Agent Architectures](https://python.langchain.com/docs/modules/agents/)
3. [OpenAI Swarm](https://github.com/openai/swarm)
4. [ReAct Paper](https://arxiv.org/abs/2210.03629)

### 实践项目

1. **个人助手**：日程管理、邮件处理、信息检索
2. **代码助手**：代码审查、bug 修复、文档生成
3. **研究助手**：文献调研、报告生成、数据分析

---

*恭喜完成 7 天 Agent 开发课程！🎉*

*现在，去构建你自己的 Agent 吧！*
