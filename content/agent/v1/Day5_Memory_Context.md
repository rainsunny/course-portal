# Day 5: Memory 与 Context 管理

> **学习时长**: 2-3 小时
> **核心问题**: 如何让 Agent 记住重要信息，同时不爆掉 Context？

---

## 学习目标

1. **理解记忆系统** - 不同类型的记忆及其作用
2. **掌握 Context 管理** - 解决 Context Rot 问题
3. **实现持久化记忆** - 跨会话保持状态
4. **设计记忆压缩策略** - 智能摘要和遗忘

---

## 第一部分：为什么需要记忆？

### 1.1 模型的局限

LLM 本身只有：
- **权重**：训练时的知识（有截止日期）
- **Context 窗口**：有限的短期记忆

记忆系统解决：
- 跨会话持久化
- 知识截止问题
- Context 窗口限制

### 1.2 记忆类型

```
┌─────────────────────────────────────────────────────┐
│                    Memory 类型                       │
├──────────────────┬──────────────────────────────────┤
│     短期记忆      │           长期记忆               │
├──────────────────┼──────────────────────────────────┤
│ - 会话内上下文    │ - 跨会话持久化                   │
│ - 工作记忆        │ - 提取关键洞察                   │
│ - 有 token 限制   │ - 压缩存储                       │
└──────────────────┴──────────────────────────────────┘
```

---

## 第二部分：Context Rot 问题

### 2.1 什么是 Context Rot？

**Context Rot**：随着对话增长，模型推理能力下降

**症状**：
- "忘记"早期信息
- 重复之前的操作
- 推理质量下降
- 回答变得平庸

### 2.2 解决方案

#### 方案一：Compaction（压缩）

```python
class ContextCompactor:
    """Context 压缩器"""
    
    def __init__(self, max_tokens: int = 100000, threshold: float = 0.8):
        self.max_tokens = max_tokens
        self.threshold = threshold
    
    def should_compact(self, current_tokens: int) -> bool:
        """是否需要压缩"""
        return current_tokens > self.max_tokens * self.threshold
    
    def compact(self, messages: List[Dict]) -> List[Dict]:
        """压缩消息历史"""
        if not self.should_compact(self.count_tokens(messages)):
            return messages
        
        # 1. 保留系统消息
        system_messages = [m for m in messages if m["role"] == "system"]
        
        # 2. 压缩历史消息
        history_messages = [m for m in messages if m["role"] != "system"]
        summary = self.summarize(history_messages)
        
        # 3. 保留最近消息
        recent_messages = history_messages[-5:]  # 保留最近 5 条
        
        # 4. 组合
        return system_messages + [
            {"role": "system", "content": f"[历史摘要]\n{summary}"}
        ] + recent_messages
    
    def summarize(self, messages: List[Dict]) -> str:
        """摘要消息"""
        # 实际实现中调用 LLM
        content = "\n".join(m["content"] for m in messages)
        return f"之前对话的关键点：{content[:500]}..."
```

#### 方案二：Tool Call Offloading

```python
# 大型工具输出只保留摘要
def handle_tool_result(result: str, max_display: int = 500) -> str:
    """处理工具结果"""
    if len(result) > max_display:
        # 保存完整结果到文件
        file_path = save_to_file(result)
        # 返回摘要
        return f"结果已保存到 {file_path}，摘要：{result[:max_display]}..."
    return result
```

#### 方案三：渐进式工具披露

```python
class ProgressiveToolLoader:
    """渐进式加载工具"""
    
    def __init__(self):
        self.all_tools = {}
        self.loaded_tools = set()
    
    def register_category(self, category: str, tools: List[str]):
        """注册工具分类"""
        self.all_tools[category] = tools
    
    def load_for_task(self, task: str) -> List[str]:
        """根据任务加载相关工具"""
        # 分析任务需要哪些工具
        if "代码" in task or "文件" in task:
            self.load_category("coding")
        if "搜索" in task or "查询" in task:
            self.load_category("search")
        
        return list(self.loaded_tools)
```

---

## 第三部分：长期记忆系统

### 3.1 记忆类型

#### Semantic Memory（语义记忆）

存储事实和知识：

```python
class SemanticMemory:
    """语义记忆 - 存储事实"""
    
    def __init__(self, storage_path: str = "memory/semantic.json"):
        self.storage_path = storage_path
        self.facts: Dict[str, Any] = {}
        self.load()
    
    def remember(self, key: str, value: Any, metadata: Dict = None):
        """记住一个事实"""
        self.facts[key] = {
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def recall(self, key: str) -> Optional[Any]:
        """回忆一个事实"""
        entry = self.facts.get(key)
        return entry["value"] if entry else None
    
    def search(self, query: str) -> List[Dict]:
        """搜索相关事实"""
        results = []
        for key, entry in self.facts.items():
            if query.lower() in key.lower() or query.lower() in str(entry["value"]).lower():
                results.append({"key": key, **entry})
        return results
```

#### Episodic Memory（情景记忆）

存储对话事件：

```python
class EpisodicMemory:
    """情景记忆 - 存储事件"""
    
    def __init__(self, storage_path: str = "memory/episodic.json"):
        self.storage_path = storage_path
        self.episodes: List[Dict] = []
        self.load()
    
    def record(self, event: str, participants: List[str] = None, 
               context: Dict = None):
        """记录一个事件"""
        episode = {
            "event": event,
            "participants": participants or [],
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        self.episodes.append(episode)
        self.save()
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """获取最近的事件"""
        return self.episodes[-n:]
    
    def find_by_time(self, start: datetime, end: datetime) -> List[Dict]:
        """按时间查找事件"""
        results = []
        for episode in self.episodes:
            ts = datetime.fromisoformat(episode["timestamp"])
            if start <= ts <= end:
                results.append(episode)
        return results
```

#### Preference Memory（偏好记忆）

存储用户偏好：

```python
class PreferenceMemory:
    """偏好记忆 - 存储用户偏好"""
    
    def __init__(self, storage_path: str = "memory/preferences.json"):
        self.storage_path = storage_path
        self.preferences: Dict[str, Dict] = {}
        self.load()
    
    def set_preference(self, category: str, key: str, value: Any):
        """设置偏好"""
        if category not in self.preferences:
            self.preferences[category] = {}
        
        self.preferences[category][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """获取偏好"""
        return self.preferences.get(category, {}).get(key, {}).get("value", default)
    
    def get_category(self, category: str) -> Dict:
        """获取某个类别的所有偏好"""
        return self.preferences.get(category, {})
```

### 3.2 统一记忆管理

```python
class MemorySystem:
    """统一记忆管理系统"""
    
    def __init__(self, workspace: str = "."):
        self.semantic = SemanticMemory(f"{workspace}/memory/semantic.json")
        self.episodic = EpisodicMemory(f"{workspace}/memory/episodic.json")
        self.preferences = PreferenceMemory(f"{workspace}/memory/preferences.json")
    
    def process_interaction(self, user_input: str, agent_response: str):
        """处理交互，提取并存储记忆"""
        # 1. 提取事实
        facts = self.extract_facts(user_input)
        for key, value in facts.items():
            self.semantic.remember(key, value)
        
        # 2. 记录事件
        self.episodic.record(
            event=f"用户: {user_input[:100]}... -> Agent: {agent_response[:100]}...",
            participants=["user", "agent"]
        )
        
        # 3. 更新偏好
        preferences = self.extract_preferences(user_input)
        for category, prefs in preferences.items():
            for key, value in prefs.items():
                self.preferences.set_preference(category, key, value)
    
    def get_context_for_prompt(self, max_items: int = 10) -> str:
        """获取用于注入 Prompt 的记忆上下文"""
        lines = ["## 关于用户的记忆\n"]
        
        # 语义记忆
        facts = list(self.semantic.facts.items())[:max_items // 2]
        if facts:
            lines.append("### 已知事实")
            for key, entry in facts:
                lines.append(f"- {key}: {entry['value']}")
        
        # 偏好记忆
        prefs = self.preferences.preferences
        if prefs:
            lines.append("\n### 用户偏好")
            for category, items in prefs.items():
                for key, entry in items.items():
                    lines.append(f"- {category}/{key}: {entry['value']}")
        
        # 最近事件
        episodes = self.episodic.get_recent(3)
        if episodes:
            lines.append("\n### 最近对话")
            for ep in episodes:
                lines.append(f"- {ep['event'][:80]}...")
        
        return "\n".join(lines)
    
    def extract_facts(self, text: str) -> Dict[str, Any]:
        """从文本中提取事实（简化版）"""
        facts = {}
        # 实际实现中调用 LLM
        if "我叫" in text:
            import re
            match = re.search(r"我叫(\w+)", text)
            if match:
                facts["用户姓名"] = match.group(1)
        return facts
    
    def extract_preferences(self, text: str) -> Dict[str, Dict]:
        """从文本中提取偏好（简化版）"""
        preferences = {}
        # 实际实现中调用 LLM
        if "喜欢" in text:
            preferences["general"] = {"喜好": text}
        return preferences
```

---

## 第四部分：记忆压缩与遗忘

### 4.1 记忆压缩策略

```python
class MemoryCompressor:
    """记忆压缩器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def compress_episode(self, episode: Dict) -> str:
        """压缩单个事件"""
        prompt = f"""
请将以下对话事件压缩为简短摘要：

事件：{episode['event']}
时间：{episode['timestamp']}

摘要（一句话）：
"""
        return self.llm.call(prompt).strip()
    
    def compress_episodes(self, episodes: List[Dict], target_length: int = 500) -> str:
        """压缩多个事件"""
        if not episodes:
            return ""
        
        # 如果总长度已经在目标内，直接返回
        total = sum(len(ep["event"]) for ep in episodes)
        if total < target_length:
            return "\n".join(ep["event"] for ep in episodes)
        
        # 否则调用 LLM 压缩
        episodes_text = "\n".join(
            f"- {ep['timestamp']}: {ep['event']}"
            for ep in episodes
        )
        
        prompt = f"""
请将以下多个对话事件压缩为一个简短的摘要：

{episodes_text}

摘要（{target_length}字符以内）：
"""
        return self.llm.call(prompt).strip()[:target_length]
    
    def selective_forget(self, episodes: List[Dict], importance_threshold: float = 0.3) -> List[Dict]:
        """选择性遗忘不重要的事件"""
        # 计算每个事件的重要性
        important_episodes = []
        for ep in episodes:
            importance = self.calculate_importance(ep)
            if importance >= importance_threshold:
                important_episodes.append(ep)
        
        return important_episodes
    
    def calculate_importance(self, episode: Dict) -> float:
        """计算事件重要性（简化版）"""
        score = 0.0
        
        # 包含关键词的事件更重要
        important_keywords = ["重要", "记住", "偏好", "决定", "同意"]
        for keyword in important_keywords:
            if keyword in episode["event"]:
                score += 0.2
        
        # 最近的事件更重要
        days_ago = (datetime.now() - datetime.fromisoformat(episode["timestamp"])).days
        recency_score = max(0, 1 - days_ago / 30) * 0.3  # 30天内递减
        score += recency_score
        
        return min(score, 1.0)
```

---

## 第五部分：实现 AGENTS.md 模式

许多成功的 Agent（如 Claude Code）使用文件来持久化记忆。

### AGENTS.md 结构

```markdown
# AGENTS.md - Agent 记忆

## 关于用户
- 姓名：张三
- 职业：软件工程师
- 偏好：Python，不喜欢 JavaScript

## 项目上下文
- 当前项目：Agent 开发课程
- 技术栈：Python, LangChain
- 风格：详细的代码注释

## 重要决策
- 2024-01-15: 决定使用 ReAct 模式
- 2024-01-16: 选择 MCP 作为工具协议

## 待办事项
- [ ] 完成 Day 4 课程
- [ ] 实现记忆压缩

## 学习记录
- 2024-01-15: 学习了 Agent 核心循环
- 2024-01-16: 实现了 ReAct Agent
```

### 实现

```python
class AgentsMdMemory:
    """AGENTS.md 模式的记忆系统"""
    
    TEMPLATE = """# AGENTS.md - Agent 记忆

## 关于用户
{user_info}

## 项目上下文
{project_context}

## 重要决策
{decisions}

## 待办事项
{todos}

## 学习记录
{learnings}
"""
    
    def __init__(self, workspace: str = "."):
        self.file_path = f"{workspace}/AGENTS.md"
        self.data = {
            "user_info": [],
            "project_context": [],
            "decisions": [],
            "todos": [],
            "learnings": []
        }
        self.load()
    
    def load(self):
        """从文件加载"""
        import os
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 解析 Markdown
            self._parse(content)
    
    def save(self):
        """保存到文件"""
        content = self.TEMPLATE.format(
            user_info=self._format_list(self.data["user_info"]),
            project_context=self._format_list(self.data["project_context"]),
            decisions=self._format_list(self.data["decisions"]),
            todos=self._format_list(self.data["todos"], checkbox=True),
            learnings=self._format_list(self.data["learnings"])
        )
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def add_user_info(self, info: str):
        """添加用户信息"""
        self.data["user_info"].append(info)
        self.save()
    
    def add_decision(self, decision: str, date: str = None):
        """添加决策记录"""
        date = date or datetime.now().strftime("%Y-%m-%d")
        self.data["decisions"].append(f"- {date}: {decision}")
        self.save()
    
    def add_todo(self, todo: str):
        """添加待办"""
        self.data["todos"].append(todo)
        self.save()
    
    def complete_todo(self, todo: str):
        """完成待办"""
        if todo in self.data["todos"]:
            self.data["todos"].remove(todo)
            self.save()
    
    def add_learning(self, learning: str, date: str = None):
        """添加学习记录"""
        date = date or datetime.now().strftime("%Y-%m-%d")
        self.data["learnings"].append(f"- {date}: {learning}")
        self.save()
    
    def _format_list(self, items: List[str], checkbox: bool = False) -> str:
        """格式化列表"""
        if not items:
            return "- 暂无"
        if checkbox:
            return "\n".join(f"- [ ] {item}" for item in items)
        return "\n".join(f"- {item}" for item in items)
    
    def _parse(self, content: str):
        """解析 Markdown（简化版）"""
        # 实际实现需要更复杂的解析
        pass
```

---

## 实践练习

### 练习 1：实现 Context 压缩器

实现一个 Context 压缩器：
- 检测 Context 是否接近上限
- 智能摘要历史消息
- 保留最近和重要的消息

### 练习 2：实现记忆提取器

实现一个记忆提取器：
- 从对话中提取事实
- 提取用户偏好
- 自动分类存储

### 练习 3：实现 AGENTS.md 系统

实现一个 AGENTS.md 系统：
- 自动更新
- 手动编辑接口
- 注入到 Prompt

---

## 今日总结

### 核心概念

```
记忆类型：
- Semantic Memory（语义记忆）：事实和知识
- Episodic Memory（情景记忆）：对话事件
- Preference Memory（偏好记忆）：用户偏好

Context 管理：
- Compaction：智能压缩
- Tool Call Offloading：输出存文件
- Progressive Disclosure：渐进式加载工具

持久化：
- 文件存储（AGENTS.md）
- 数据库存储
- 向量存储
```

### 关键洞察

1. **记忆不是越多越好**：需要压缩和遗忘
2. **Context 是有限资源**：要珍惜使用
3. **文件是最好的持久化**：简单、可读、版本控制

### 明日预告

Day 6：多 Agent 系统 —— 如何让多个 Agent 协作。

---

## 阅读材料

1. [AWS: AgentCore Long-term Memory](https://aws.amazon.com/blogs/machine-learning/building-smarter-ai-agents-agentcore-long-term-memory-deep-dive/)
2. [LangChain: Memory](https://python.langchain.com/docs/modules/memory/)
3. [Anthropic: Context Windows](https://docs.anthropic.com/claude/docs/context-windows)

---

*Day 5 完成！明天我们将探索多 Agent 系统。*
