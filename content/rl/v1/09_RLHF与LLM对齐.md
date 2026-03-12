# Day 9：RLHF与LLM训练

> **核心问题**：如何让大语言模型与人类意图对齐？

## 引言：从ChatGPT看RLHF的力量

2022年11月，ChatGPT 横空出世，震惊世界。它的核心能力——**理解人类意图、生成有用回复、拒绝有害请求**——并非来自更多的预训练数据，而是来自一项关键技术：**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**。

**RLHF 的本质**：让模型从人类的偏好中学习，而非仅仅从文本中学习。

```
预训练 → 学习"什么是合理的文本"
    ↓
SFT（监督微调）→ 学习"如何回答问题"
    ↓
RLHF → 学习"什么是好的回答"
```

今天，我们深入理解 RLHF 的完整流程，以及它的替代方案 DPO 和 DeepSeek R1 的创新。

---

## 1. 为什么需要RLHF？

### 1.1 预训练的局限

**预训练目标**：预测下一个 token

$$\mathcal{L}_{pretrain} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

**问题**：这个目标不关心：
- 回答是否有帮助
- 信息是否准确
- 是否遵守安全规范

**例子**：
```
用户：如何制作炸弹？

预训练模型（可能继续）：
"需要以下材料：硝酸铵、燃料油..." ← 危险！
```

### 1.2 监督微调的不足

**SFT（Supervised Fine-Tuning）**：在高质量问答数据上微调

**问题**：
1. **数据瓶颈**：高质量标注数据昂贵
2. **主观性**：什么是"好的回答"因人而异
3. **覆盖率**：难以覆盖所有场景

**关键洞察**：**人类更容易比较两个回答的好坏，而非写出完美回答。**

### 1.3 RLHF的核心思想

**RLHF 三步流程**：

```
Step 1: SFT（监督微调）
    高质量问答数据 → 微调预训练模型

Step 2: RM（奖励模型训练）
    人类偏好数据 → 训练奖励模型

Step 3: PPO（强化学习优化）
    奖励模型 → RL优化策略模型
```

**核心创新**：用人类偏好训练奖励模型，用奖励模型指导策略优化。

---

## 2. RLHF完整流程

### 2.1 第一阶段：监督微调（SFT）

**目标**：让模型学会基本的对话能力。

**数据**：高质量问答对

$$\mathcal{D}_{SFT} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$$

**训练**：标准语言模型训练

$$\mathcal{L}_{SFT} = -\sum_{(x,y) \in \mathcal{D}_{SFT}} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$$

**实践要点**：
- 数据质量比数量更重要
- 通常几千到几万条高质量数据足够
- 多样化指令类型

```python
class SFTModel(nn.Module):
    """监督微调模型"""
    
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
    
    def forward(self, input_ids, attention_mask, labels):
        """前向传播"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
    
    def generate(self, prompt, max_length=512, temperature=1.0, top_p=0.9):
        """生成回复"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_sft(model, train_data, epochs=3, lr=2e-5, batch_size=8):
    """SFT 训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            loss, _ = model(input_ids, attention_mask, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")
    
    return model
```

### 2.2 第二阶段：奖励模型训练（RM）

**目标**：训练一个模型来预测人类偏好。

**数据收集**：给定提示 x，让模型生成多个回答 $\{y_1, y_2, ..., y_k\}$，让人类标注员排序。

**偏好数据**：

$$\mathcal{D}_{pref} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^{N}$$

其中 $y_w$ 是更好的回答（winner），$y_l$ 是较差的回答（loser）。

**奖励模型**：$r_\phi(x, y)$ 输入提示和回答，输出标量奖励。

**训练目标**：Bradley-Terry 模型

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

**直觉**：奖励模型应该给好的回答更高的分数。

```python
class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.model = base_model
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        """前向传播，返回奖励值"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取最后一个 token 的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        last_token_hidden = last_hidden_state[:, -1, :]
        
        # 输出奖励值
        reward = self.value_head(last_token_hidden)
        return reward
    
    def get_reward(self, text):
        """获取单个文本的奖励"""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            reward = self.forward(inputs['input_ids'], inputs['attention_mask'])
        return reward.item()


def train_reward_model(model, preference_data, epochs=1, lr=1e-5):
    """训练奖励模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in preference_data:
            prompt = batch['prompt']
            chosen = batch['chosen']
            rejected = batch['rejected']
            
            # 获取奖励
            chosen_inputs = model.tokenizer(prompt + chosen, return_tensors='pt')
            rejected_inputs = model.tokenizer(prompt + rejected, return_tensors='pt')
            
            chosen_reward = model(chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
            rejected_reward = model(rejected_inputs['input_ids'], rejected_inputs['attention_mask'])
            
            # Bradley-Terry 损失
            loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(preference_data):.4f}")
    
    return model
```

**训练技巧**：
1. **归一化**：奖励值通常归一化到均值 0
2. **正则化**：防止奖励模型过度自信
3. **数据增强**：同一提示的多个排序组合

### 2.3 第三阶段：PPO强化学习优化

**目标**：使用奖励模型指导策略优化。

**策略模型**：$\pi_\theta(y|x)$，即经过 SFT 的语言模型。

**目标函数**：

$$\mathcal{L}_{PPO} = \mathbb{E}_{(x,y) \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

其中：
- $r_\phi(x, y)$：奖励模型的评分
- $\pi_{ref}$：参考模型（通常是 SFT 后的模型）
- $\beta$：KL 惩罚系数

**KL 惩罚的作用**：
- 防止策略偏离参考模型太远
- 避免语言模型退化

```python
class PPOForRLHF:
    """用于 RLHF 的 PPO"""
    
    def __init__(
        self,
        policy_model,          # 策略模型（待优化）
        reference_model,       # 参考模型（固定）
        reward_model,          # 奖励模型
        value_model,           # 价值模型
        tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 冻结参考模型和奖励模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
    
    def generate_with_logprobs(self, prompts, max_length=512):
        """生成回复并计算 log probs"""
        # 编码输入
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        
        # 生成
        outputs = self.policy_model.generate(
            inputs['input_ids'],
            max_length=max_length,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # 计算策略模型的 log probs
        policy_logprobs = self._compute_logprobs(
            self.policy_model, 
            outputs.sequences
        )
        
        # 计算参考模型的 log probs
        with torch.no_grad():
            ref_logprobs = self._compute_logprobs(
                self.reference_model, 
                outputs.sequences
            )
        
        # 解码回复
        responses = self.tokenizer.batch_decode(
            outputs.sequences, 
            skip_special_tokens=True
        )
        
        return responses, policy_logprobs, ref_logprobs
    
    def _compute_logprobs(self, model, sequences):
        """计算序列的 log probs"""
        outputs = model(sequences)
        logits = outputs.logits[:, :-1, :]  # 去掉最后一个
        labels = sequences[:, 1:]  # 去掉第一个
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        return token_logprobs.sum(dim=-1)
    
    def compute_rewards(self, prompts, responses):
        """计算奖励"""
        texts = [p + r for p, r in zip(prompts, responses)]
        rewards = self.reward_model.get_rewards(texts)
        return rewards
    
    def compute_kl_penalty(self, policy_logprobs, ref_logprobs):
        """计算 KL 散度惩罚"""
        kl = policy_logprobs - ref_logprobs
        return kl
    
    def update(self, batch):
        """PPO 更新"""
        prompts = batch['prompts']
        responses = batch['responses']
        old_logprobs = batch['logprobs']
        ref_logprobs = batch['ref_logprobs']
        rewards = batch['rewards']
        values = batch['values']
        
        # 重新计算 log probs
        new_logprobs = self._compute_logprobs(
            self.policy_model,
            self.tokenizer(responses, return_tensors='pt')['input_ids']
        )
        
        # 计算比率
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 计算优势
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # KL 惩罚
        kl_penalty = self.compute_kl_penalty(new_logprobs, ref_logprobs)
        
        # PPO 目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() + self.kl_coef * kl_penalty.mean()
        
        # 价值损失
        new_values = self.value_model(prompts, responses)
        value_loss = 0.5 * (new_values - rewards).pow(2).mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss
        
        return loss, policy_loss, value_loss
    
    def train(self, prompts, epochs=1, batch_size=8):
        """训练循环"""
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=1e-6)
        
        for epoch in range(epochs):
            # 生成回复
            responses, policy_logprobs, ref_logprobs = self.generate_with_logprobs(prompts)
            
            # 计算奖励
            rewards = self.compute_rewards(prompts, responses)
            
            # 计算价值
            values = self.value_model(prompts, responses)
            
            # 分批更新
            for i in range(0, len(prompts), batch_size):
                batch = {
                    'prompts': prompts[i:i+batch_size],
                    'responses': responses[i:i+batch_size],
                    'logprobs': policy_logprobs[i:i+batch_size],
                    'ref_logprobs': ref_logprobs[i:i+batch_size],
                    'rewards': rewards[i:i+batch_size],
                    'values': values[i:i+batch_size]
                }
                
                loss, policy_loss, value_loss = self.update(batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        return self.policy_model
```

### 2.4 RLHF完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        RLHF 完整流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  预训练模型 (LLM)                                            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │  Step 1 │ SFT（监督微调）                                 │
│  │         │ 高质量问答数据 → 基础对话能力                    │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │  Step 2 │ RM（奖励模型训练）                              │
│  │         │ 人类偏好数据 → 奖励模型                          │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                                │
│  │  Step 3 │ PPO（强化学习优化）                             │
│  │         │ 奖励模型 → 优化策略模型                          │
│  └────┬────┘                                                │
│       │                                                     │
│       ▼                                                     │
│  对齐后的模型（ChatGPT 风格）                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. DPO：直接偏好优化

### 3.1 RLHF的问题

**RLHF 的复杂性**：
1. 需要训练三个模型：策略模型、奖励模型、价值模型
2. PPO 训练不稳定，需要精心调参
3. 奖励模型可能被"欺骗"
4. 计算成本高

**问题**：能不能跳过奖励模型，直接从偏好数据学习？

### 3.2 DPO的核心思想

**DPO（Direct Preference Optimization）**：直接从偏好数据优化策略，无需显式奖励模型。

**关键洞察**：从 Bradley-Terry 模型出发，可以直接推导出最优策略。

**推导过程**：

从奖励模型的目标出发：

$$p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

最优策略与奖励的关系：

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

其中 $Z(x)$ 是配分函数，与策略无关。

代入 Bradley-Terry 模型，得到 **DPO 损失**：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**直觉**：
- 增加好回答（$y_w$）的概率
- 减少差回答（$y_l$）的概率
- 用 KL 散度约束偏离程度

### 3.3 DPO实现

```python
class DPO:
    """Direct Preference Optimization"""
    
    def __init__(
        self,
        policy_model,      # 待优化的策略模型
        reference_model,   # 参考模型（SFT 后）
        tokenizer,
        beta=0.1,          # KL 惩罚系数
        learning_rate=1e-6
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.beta = beta
        
        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), 
            lr=learning_rate
        )
    
    def compute_logprobs(self, model, input_ids, attention_mask):
        """计算序列的 log probs"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        return -outputs.loss  # 负对数似然
    
    def dpo_loss(self, policy_chosen_logprobs, policy_rejected_logprobs,
                 ref_chosen_logprobs, ref_rejected_logprobs):
        """计算 DPO 损失"""
        # 对数比率
        chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = policy_rejected_logprobs - ref_rejected_logprobs
        
        # DPO 损失
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -torch.nn.functional.logsigmoid(logits).mean()
        
        return loss
    
    def train_step(self, batch):
        """单步训练"""
        # 编码
        chosen_inputs = self.tokenizer(
            [p + c for p, c in zip(batch['prompts'], batch['chosen'])],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        rejected_inputs = self.tokenizer(
            [p + r for p, r in zip(batch['prompts'], batch['rejected'])],
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # 策略模型的 log probs
        policy_chosen_logprobs = self.compute_logprobs(
            self.policy_model,
            chosen_inputs['input_ids'],
            chosen_inputs['attention_mask']
        )
        policy_rejected_logprobs = self.compute_logprobs(
            self.policy_model,
            rejected_inputs['input_ids'],
            rejected_inputs['attention_mask']
        )
        
        # 参考模型的 log probs（不计算梯度）
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_logprobs(
                self.reference_model,
                chosen_inputs['input_ids'],
                chosen_inputs['attention_mask']
            )
            ref_rejected_logprobs = self.compute_logprobs(
                self.reference_model,
                rejected_inputs['input_ids'],
                rejected_inputs['attention_mask']
            )
        
        # DPO 损失
        loss = self.dpo_loss(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs
        )
        
        return loss
    
    def train(self, preference_data, epochs=1, batch_size=4):
        """训练循环"""
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(preference_data), batch_size):
                batch = {
                    'prompts': preference_data['prompts'][i:i+batch_size],
                    'chosen': preference_data['chosen'][i:i+batch_size],
                    'rejected': preference_data['rejected'][i:i+batch_size]
                }
                
                loss = self.train_step(batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(preference_data) // batch_size)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        return self.policy_model
```

### 3.4 DPO vs RLHF

| 特性 | RLHF | DPO |
|------|------|-----|
| **奖励模型** | 需要训练 | 不需要 |
| **模型数量** | 3个（策略、奖励、价值） | 2个（策略、参考） |
| **训练稳定性** | PPO 不稳定 | 稳定 |
| **计算成本** | 高 | 低 |
| **理论保证** | 最优策略 | 最优策略 |
| **适用场景** | 大规模训练 | 中小规模、快速迭代 |

**DPO 的优势**：
1. 简单：不需要训练奖励模型
2. 稳定：没有 PPO 的不稳定性
3. 高效：计算成本低

**DPO 的局限**：
1. 对偏好数据质量要求高
2. 可能不如 RLHF 灵活

---

## 4. DeepSeek R1的创新

### 4.1 传统RLHF的瓶颈

**问题**：
1. **依赖大量标注数据**：偏好数据昂贵
2. **SFT 冷启动**：需要高质量 SFT 数据
3. **训练流程复杂**：多阶段训练
4. **奖励模型偏差**：可能与人类真实偏好不一致

### 4.2 DeepSeek R1的核心创新

**DeepSeek R1**（2025）提出了一种突破性的方法：**纯强化学习，无需 SFT 冷启动**。

**核心思想**：
1. **直接从预训练模型开始 RL 训练**
2. **让模型"涌现"推理能力**
3. **使用"结果奖励"而非"过程监督"**

```
传统 RLHF：
预训练 → SFT → RM 训练 → PPO → 对齐模型

DeepSeek R1：
预训练 → 直接 RL → 推理模型
```

### 4.3 DeepSeek R1的训练流程

**第一阶段：冷启动（少量数据）**

只用极少量（几千条）高质量数据让模型学会基本格式。

**第二阶段：强化学习**

**奖励设计**：
- **准确性奖励**：答案是否正确
- **格式奖励**：输出格式是否规范
- **语言一致性**：是否使用统一语言

**关键创新**：不奖励"思考过程"，只奖励"最终答案"。

```python
class DeepSeekR1Reward:
    """DeepSeek R1 的奖励函数"""
    
    def __init__(self, answer_checker, format_checker, language_checker):
        self.answer_checker = answer_checker      # 检查答案正确性
        self.format_checker = format_checker       # 检查格式
        self.language_checker = language_checker   # 检查语言一致性
    
    def compute_reward(self, prompt, response, ground_truth):
        """计算奖励"""
        # 准确性奖励（主要）
        accuracy_reward = self.answer_checker(response, ground_truth)
        
        # 格式奖励（辅助）
        format_reward = self.format_checker(response)
        
        # 语言一致性奖励（辅助）
        language_reward = self.language_checker(prompt, response)
        
        # 总奖励
        total_reward = (
            1.0 * accuracy_reward +
            0.2 * format_reward +
            0.1 * language_reward
        )
        
        return total_reward


class AccuracyReward:
    """准确性奖励：检查答案是否正确"""
    
    def __init__(self, use_verification=True):
        self.use_verification = use_verification
    
    def check(self, response, ground_truth):
        """检查答案正确性"""
        # 提取最终答案
        predicted_answer = self.extract_answer(response)
        
        # 比较
        if predicted_answer == ground_truth:
            return 1.0
        elif self.partial_match(predicted_answer, ground_truth):
            return 0.5
        else:
            return 0.0
    
    def extract_answer(self, response):
        """从响应中提取最终答案"""
        # 查找 <answer> 标签
        import re
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
    
    def partial_match(self, predicted, ground_truth):
        """部分匹配检查"""
        # 简化实现：检查关键词
        predicted_words = set(predicted.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        overlap = predicted_words & ground_truth_words
        return len(overlap) / len(ground_truth_words) > 0.5


class FormatReward:
    """格式奖励：检查输出格式"""
    
    def check(self, response):
        """检查格式是否规范"""
        score = 0.0
        
        # 检查是否有思考过程
        if '<think>' in response and '</think>' in response:
            score += 0.5
        
        # 检查是否有最终答案
        if '<answer>' in response and '</answer>' in response:
            score += 0.5
        
        return score


class LanguageConsistencyReward:
    """语言一致性奖励"""
    
    def check(self, prompt, response):
        """检查语言一致性"""
        # 简化实现：检查主要语言是否一致
        prompt_lang = self.detect_language(prompt)
        response_lang = self.detect_language(response)
        
        if prompt_lang == response_lang:
            return 1.0
        else:
            return 0.0
    
    def detect_language(self, text):
        """检测语言（简化版）"""
        # 统计中英文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return 'chinese'
        else:
            return 'english'
```

### 4.4 涌现的推理能力

**DeepSeek R1 的惊人发现**：通过纯 RL 训练，模型自发涌现出：

1. **长链式推理**：模型学会"多想想"
2. **自我反思**：模型会检查自己的推理
3. **推理规划**：模型会规划解题步骤

```
用户：小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？

DeepSeek R1 的思考过程：
```
让我一步步分析：
1. 小明最初有 5 个苹果
2. 给了小红 2 个：5 - 2 = 3 个
3. 又买了 3 个：3 + 3 = 6 个
所以答案是 6 个苹果。

等等，让我再验证一下：
- 初始：5 个
- 给出后：5 - 2 = 3 个 ✓
- 买入后：3 + 3 = 6 个 ✓

答案正确。
```
<answer>6</answer>
```

这种"涌现"不是预设的，而是模型为了提高正确率自发学习的策略！

### 4.5 DeepSeek R1 vs 传统方法

| 特性 | 传统 RLHF | DeepSeek R1 |
|------|-----------|-------------|
| **SFT 数据** | 需要大量 | 极少量/不需要 |
| **奖励模型** | 需要训练 | 简单规则 |
| **训练复杂度** | 多阶段 | 单阶段 |
| **推理能力** | 需要特殊设计 | 自然涌现 |
| **可解释性** | 较低 | 较高（有思考过程） |

### 4.6 实现DeepSeek R1风格的训练

```python
class DeepSeekR1Trainer:
    """DeepSeek R1 风格的训练器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        reward_fn,
        kl_coef=0.1,
        learning_rate=1e-6,
        max_new_tokens=1024,
        temperature=0.7
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 参考模型（冻结）
        self.reference_model = copy.deepcopy(model)
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def generate_with_thinking(self, prompts):
        """生成带思考过程的回复"""
        # 添加思考提示
        thinking_prompts = [
            p + "\n\n请一步步思考：" for p in prompts
        ]
        
        inputs = self.tokenizer(
            thinking_prompts, 
            return_tensors='pt', 
            padding=True
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        responses = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return responses
    
    def compute_policy_loss(self, prompts, responses, rewards):
        """计算策略损失"""
        # 编码
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts, return_tensors='pt', padding=True)
        
        # 计算策略模型的 log probs
        outputs = self.model(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['input_ids']
        )
        policy_logprobs = -outputs.loss
        
        # 计算参考模型的 log probs
        with torch.no_grad():
            ref_outputs = self.reference_model(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids']
            )
            ref_logprobs = -ref_outputs.loss
        
        # KL 惩罚
        kl_penalty = policy_logprobs - ref_logprobs
        
        # 总损失：奖励 - KL
        loss = -(rewards.mean() - self.kl_coef * kl_penalty)
        
        return loss
    
    def train_step(self, prompts, ground_truths):
        """单步训练"""
        # 生成回复
        responses = self.generate_with_thinking(prompts)
        
        # 计算奖励
        rewards = []
        for p, r, gt in zip(prompts, responses, ground_truths):
            reward = self.reward_fn.compute_reward(p, r, gt)
            rewards.append(reward)
        rewards = torch.tensor(rewards)
        
        # 计算损失
        loss = self.compute_policy_loss(prompts, responses, rewards)
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), rewards.mean().item()
    
    def train(self, dataset, epochs=1, batch_size=4):
        """训练循环"""
        for epoch in range(epochs):
            total_loss = 0
            total_reward = 0
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                prompts = batch['prompts']
                ground_truths = batch['answers']
                
                loss, reward = self.train_step(prompts, ground_truths)
                total_loss += loss
                total_reward += reward
            
            n_batches = len(dataset) // batch_size
            print(f"Epoch {epoch + 1}")
            print(f"  Loss: {total_loss / n_batches:.4f}")
            print(f"  Avg Reward: {total_reward / n_batches:.4f}")
        
        return self.model
```

---

## 5. 实践：从零实现RLHF

### 5.1 数据准备

```python
# SFT 数据格式
sft_data = {
    "prompts": [
        "解释什么是机器学习？",
        "如何学习编程？",
        ...
    ],
    "responses": [
        "机器学习是人工智能的一个分支...",
        "学习编程的建议如下：1. 选择一门语言...",
        ...
    ]
}

# 偏好数据格式
preference_data = {
    "prompts": [
        "什么是量子计算？",
        ...
    ],
    "chosen": [
        "量子计算是利用量子力学原理进行计算的技术...",
        ...
    ],
    "rejected": [
        "量子计算就是很快的计算。",
        ...
    ]
}
```

### 5.2 完整训练流程

```python
def train_rlhf_pipeline(
    base_model,
    tokenizer,
    sft_data,
    preference_data,
    rl_data
):
    """完整的 RLHF 训练流程"""
    
    print("=" * 50)
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 50)
    
    sft_model = train_sft(base_model, sft_data, epochs=3)
    
    print("\n" + "=" * 50)
    print("Stage 2: Reward Model Training")
    print("=" * 50)
    
    reward_model = train_reward_model(sft_model, preference_data, epochs=1)
    
    print("\n" + "=" * 50)
    print("Stage 3: PPO Training")
    print("=" * 50)
    
    # 初始化 PPO
    ppo = PPOForRLHF(
        policy_model=sft_model,
        reference_model=copy.deepcopy(sft_model),
        reward_model=reward_model,
        value_model=copy.deepcopy(reward_model),  # 简化：用奖励模型作为价值模型
        tokenizer=tokenizer,
        kl_coef=0.1
    )
    
    # PPO 训练
    aligned_model = ppo.train(rl_data['prompts'], epochs=1)
    
    print("\n" + "=" * 50)
    print("RLHF Training Complete!")
    print("=" * 50)
    
    return aligned_model
```

### 5.3 评估对齐效果

```python
def evaluate_alignment(model, tokenizer, test_cases):
    """评估模型对齐效果"""
    results = {
        'helpfulness': [],
        'harmlessness': [],
        'honesty': []
    }
    
    for case in test_cases:
        response = generate_response(model, tokenizer, case['prompt'])
        
        # 评估帮助性
        if 'helpful_criteria' in case:
            helpful_score = evaluate_helpfulness(response, case['helpful_criteria'])
            results['helpfulness'].append(helpful_score)
        
        # 评估无害性
        if 'harmful_prompt' in case:
            is_safe = check_safety(response)
            results['harmlessness'].append(is_safe)
        
        # 评估诚实性
        if 'fact_check' in case:
            honest_score = evaluate_honesty(response, case['fact_check'])
            results['honesty'].append(honest_score)
    
    # 计算平均分数
    summary = {
        k: sum(v) / len(v) if v else 0 
        for k, v in results.items()
    }
    
    return summary
```

---

## 6. 思考题

### 概念理解

1. **RLHF 中的奖励模型为什么要用 Bradley-Terry 模型？有没有其他选择？**

2. **DPO 如何避免显式训练奖励模型？它的理论保证是什么？**

3. **DeepSeek R1 为什么能涌现推理能力？这与传统的 RLHF 有什么本质区别？**

### 数学推导

4. **推导 Bradley-Terry 模型的损失函数：**
   
   $$\mathcal{L} = -\mathbb{E}[\log \sigma(r(x, y_w) - r(x, y_l))]$$

5. **从最优奖励与策略的关系推导 DPO 损失函数：**
   
   $$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \text{const}$$

6. **证明 DPO 损失的最优解对应最优策略。**

### 编程实践

7. **实现一个完整的 DPO 训练流程，在小型模型（如 GPT-2）上验证效果。**

8. **比较 RLHF 和 DPO 在相同数据集上的训练效率和最终效果。**

9. **实现 DeepSeek R1 风格的奖励函数，观察模型是否能涌现推理能力。**

---

## 7. 拓展阅读

### 经典论文

- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS. —— InstructGPT/ChatGPT
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS. —— DPO
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. —— DeepSeek R1
- Stiennon, N., et al. (2020). *Learning to summarize from human feedback*. NeurIPS. —— 早期 RLHF

### 推荐资源

- OpenAI Blog: *ChatGPT: Optimizing Language Models for Dialogue*
- Anthropic Blog: *Constitutional AI*
- Hugging Face TRL: RLHF 训练框架
- DeepSeek Blog: DeepSeek R1 技术报告

### 进阶主题

- **Constitutional AI**：用规则代替人类反馈
- **RLAIF**：用 AI 反馈代替人类反馈
- **IPO**：改进的 DPO 变体
- **KTO**：Kahneman-Tversky 优化

---

## 小结

**Day 9 核心要点**：

| 概念 | 定义 | 重要性 |
|------|------|--------|
| **RLHF** | 基于人类反馈的强化学习 | ⭐⭐⭐⭐⭐ |
| **奖励模型** | 从偏好数据学习奖励函数 | ⭐⭐⭐⭐⭐ |
| **PPO for RLHF** | 用于 LLM 对齐的 PPO | ⭐⭐⭐⭐⭐ |
| **KL 惩罚** | 防止策略偏离参考模型 | ⭐⭐⭐⭐ |
| **DPO** | 直接偏好优化 | ⭐⭐⭐⭐⭐ |
| **DeepSeek R1** | 纯 RL 训练，涌现推理 | ⭐⭐⭐⭐⭐ |

**核心公式速查**：

| 公式 | 表达式 |
|------|--------|
| Bradley-Terry 损失 | $\mathcal{L} = -\mathbb{E}[\log \sigma(r(x, y_w) - r(x, y_l))]$ |
| PPO-RLHF 目标 | $L = \mathbb{E}[r(x,y) - \beta \log \frac{\pi(y\|x)}{\pi_{ref}(y\|x)}]$ |
| DPO 损失 | $\mathcal{L} = -\mathbb{E}[\log \sigma(\beta(\log\frac{\pi(y_w\|x)}{\pi_{ref}(y_w\|x)} - \log\frac{\pi(y_l\|x)}{\pi_{ref}(y_l\|x)}))]$ |
| DeepSeek R1 奖励 | $r = r_{accuracy} + \alpha r_{format} + \beta r_{language}$ |

**一句话总结**：RLHF 通过奖励模型将人类偏好转化为可学习的信号，DPO 简化了训练流程，DeepSeek R1 证明了纯 RL 可以让模型涌现推理能力——这三种方法代表了 LLM 对齐技术的演进路线。

---

**下一讲预告**：Day 10 将探讨强化学习的前沿方向，包括多智能体强化学习、博弈论基础、离线RL、元RL等。

---

*Day 9 完成！ 🎉*