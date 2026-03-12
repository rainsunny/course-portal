# Day 8：PPO与现代策略优化

> **核心问题**：如何让策略梯度训练既稳定又高效？

## 引言：策略梯度的困境

Day 6 我们学习了策略梯度方法，它可以直接优化策略，处理连续动作空间。但策略梯度有一个致命问题：**训练不稳定**。

**不稳定的原因**：

1. **步长难以控制**：同样的学习率，在不同状态下可能导致完全不同的策略变化
2. **性能崩塌**：一次糟糕的更新可能彻底破坏策略，且难以恢复
3. **样本效率低**：需要大量样本，且必须在线采集

想象你在爬一座山：

- **普通策略梯度**：每步随机迈出，可能一步迈太大掉下悬崖
- **信任区域方法**：每步确保不会比当前位置差太多，稳步前进

今天我们学习现代策略优化方法，它们解决了策略梯度的稳定性问题，成为当今最流行的深度强化学习算法。

---

## 1. 策略梯度的问题与解决思路

### 1.1 策略梯度的不稳定性

回顾策略梯度更新：

$$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$$

**问题**：步长 α 对不同状态、不同策略的影响差异巨大。

**例子**：
- 当前策略在某些状态几乎确定（π(a|s) ≈ 1），小的梯度可能导致大的策略变化
- 当前策略在某些状态很随机（π(a|s) ≈ 0.5），大的梯度可能只改变一点点

**后果**：
- 步长太小：学习太慢
- 步长太大：策略剧烈变化，性能崩塌

### 1.2 信任区域的思想

**核心洞察**：我们不应该关心梯度的大小，而应该关心**策略的实际变化**。

**策略变化度量**：KL 散度

$$D_{KL}(\pi_{old} \| \pi_{new}) = \mathbb{E}_{s \sim \pi_{old}} \left[ \sum_a \pi_{old}(a|s) \log \frac{\pi_{old}(a|s)}{\pi_{new}(a|s)} \right]$$

**信任区域约束**：

$$\max_\theta \mathbb{E}[A^{\pi_{old}}(s, a)] \quad \text{s.t.} \quad D_{KL}(\pi_{old} \| \pi_{new}) \leq \delta$$

**直觉**：在新策略与旧策略"足够相似"的前提下，最大化优势函数。

### 1.3 从约束到目标函数

**TRPO 的目标函数**：

$$L^{TRPO}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A^{\pi_{old}}(s_t, a_t) \right]$$

定义**重要性采样比率**：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

则目标函数变为：

$$L^{TRPO}(\theta) = \mathbb{E}_t [r_t(\theta) A_t]$$

**问题**：TRPO 需要计算二阶导数（Fisher 信息矩阵），计算量大。

**PPO 的解决思路**：用简单的裁剪代替复杂的约束。

---

## 2. TRPO：信任区域策略优化

### 2.1 目标函数推导

从策略梯度出发，考虑在新策略下采样的困难，使用重要性采样：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A^{\theta_{old}}(s_t, a_t) \right]$$

**局部近似**：当 θ 接近 θ_{old} 时，这个目标函数是真实目标函数的良好近似。

### 2.2 约束优化

**TRPO 的约束优化问题**：

$$\max_\theta L(\theta) = \mathbb{E}_t [r_t(\theta) A_t]$$
$$\text{s.t.} \quad \bar{D}_{KL}(\theta_{old} \| \theta) \leq \delta$$

其中 $\bar{D}_{KL}$ 是平均 KL 散度。

**求解方法**：拉格朗日对偶 + 自然梯度

### 2.3 自然梯度

**普通梯度**：参数空间中的最陡上升方向

**自然梯度**：概率分布空间中的最陡上升方向

$$\nabla_{nat} J = F^{-1} \nabla_\theta J$$

其中 F 是 Fisher 信息矩阵：

$$F = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T \right]$$

**直觉**：自然梯度考虑了参数变化对概率分布的影响，是更"真实"的梯度。

### 2.4 TRPO 算法流程

```
循环：
    1. 使用当前策略 π_θ 收集一批轨迹
    2. 计算优势函数估计 A_t（使用 GAE）
    3. 计算策略梯度 g = ∇_θ L(θ)
    4. 计算 Fisher 信息矩阵 F（或其近似）
    5. 自然梯度更新：θ = θ + α F^{-1} g
       （满足 KL 约束）
    6. 更新价值函数（拟合回报）
```

### 2.5 TRPO 的实现困难

1. **计算 Fisher 信息矩阵**：参数量大时，矩阵 O(n²) 存储，O(n³) 求逆
2. **共轭梯度**：用共轭梯度法近似求解，但需要多次迭代
3. **线搜索**：需要回溯确保满足约束

```python
class TRPO:
    """TRPO 简化实现（展示核心思想）"""
    
    def __init__(self, policy, value_fn, gamma=0.99, lambda_gae=0.95, 
                 kl_target=0.01, damping=0.1):
        self.policy = policy
        self.value_fn = value_fn
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.kl_target = kl_target
        self.damping = damping
    
    def compute_kl(self, old_log_probs, new_log_probs):
        """计算 KL 散度"""
        return (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
    
    def compute_policy_gradient(self, states, actions, advantages, old_log_probs):
        """计算策略梯度"""
        log_probs = self.policy.get_log_prob(states, actions)
        ratio = (log_probs - old_log_probs).exp()
        loss = -(ratio * advantages).mean()
        
        gradient = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=True)
        return torch.cat([g.flatten() for g in gradient])
    
    def compute_fisher_vector_product(self, v, states, actions):
        """计算 Fisher 矩阵与向量的乘积（避免存储完整矩阵）"""
        log_probs = self.policy.get_log_prob(states, actions)
        kl = self.compute_kl(log_probs.detach(), log_probs)
        
        # KL 的梯度
        kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        kl_grad = torch.cat([g.flatten() for g in kl_grad])
        
        # KL 梯度与 v 的点积
        kl_v = (kl_grad * v).sum()
        
        # 二阶导数
        kl_v_grad = torch.autograd.grad(kl_v, self.policy.parameters())
        kl_v_grad = torch.cat([g.flatten() for g in kl_v_grad])
        
        return kl_v_grad + self.damping * v
    
    def conjugate_gradient(self, b, states, actions, n_steps=10):
        """共轭梯度法求解 Fx = b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = (r * r).sum()
        
        for _ in range(n_steps):
            Fp = self.compute_fisher_vector_product(p, states, actions)
            alpha = rdotr / (p * Fp).sum()
            x += alpha * p
            r -= alpha * Fp
            new_rdotr = (r * r).sum()
            if new_rdotr < 1e-10:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr
        
        return x
    
    def update(self, trajectories):
        """TRPO 更新"""
        # 提取数据
        states, actions, rewards, next_states, dones = self.process_trajectories(trajectories)
        
        # 计算优势和旧策略概率
        advantages = self.compute_gae(rewards, states, next_states, dones)
        old_log_probs = self.policy.get_log_prob(states, actions).detach()
        
        # 计算梯度
        gradient = self.compute_policy_gradient(states, actions, advantages, old_log_probs)
        
        # 共轭梯度求解搜索方向
        step_direction = self.conjugate_gradient(gradient, states, actions)
        
        # 线搜索确定步长
        step_size = self.line_search(gradient, step_direction, states, actions, 
                                     advantages, old_log_probs)
        
        # 更新参数
        self.policy.update(step_size * step_direction)
```

---

## 3. PPO：近端策略优化

### 3.1 PPO 的核心创新

**PPO（Proximal Policy Optimization）** 的核心思想：**用简单的裁剪代替复杂的约束优化**。

**TRPO 的问题**：
- 需要计算二阶导数
- 需要共轭梯度
- 实现复杂

**PPO 的解决方案**：
- 直接修改目标函数
- 当策略变化太大时，裁剪目标函数
- 只需要一阶导数，可以用标准 SGD

### 3.2 PPO-Clip 目标函数

**核心公式**：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是裁剪参数（通常 0.1-0.2）

**裁剪的作用**：

| 情况 | A > 0（好动作） | A < 0（坏动作） |
|------|-----------------|-----------------|
| r > 1+ε | 裁剪到 1+ε，不再鼓励 | 不裁剪，抑制 |
| r < 1-ε | 不裁剪，鼓励 | 裁剪到 1-ε，不再抑制 |
| 1-ε ≤ r ≤ 1+ε | 正常更新 | 正常更新 |

**直觉**：
- 如果动作好（A > 0），增加其概率，但不超过 (1+ε) 倍
- 如果动作坏（A < 0），减少其概率，但不低于 (1-ε) 倍
- 这样确保策略不会变化太大

### 3.3 为什么 PPO 有效？

**图解 PPO-Clip**：

```
当 A > 0 时（好动作）：
                    L(θ)
                      ↑
                      │     ╱──────
                      │    ╱
                      │   ╱  ← 裁剪区域
              1+ε ────│──╱─────────
                      │ ╱
                    1─┼────────────
                    ╱ │
              1-ε ─╱──┼────────────
                  ╱   │
                ╱     │
              0───────┼────────────→ r(θ)
                      1

当 A < 0 时（坏动作）：
                    L(θ)
                      ↑
                    ╱ │
              1-ε ─╱──┼────────────
                  ╱   │
                    1─┼────────────
                      │ ╲
              1+ε ────│──╲─────────
                      │   ╲  ← 裁剪区域
                      │    ╲
                      │     ╲──────
                      0────────────→ r(θ)
                      1
```

**关键洞察**：
- 当 r(θ) 超出 [1-ε, 1+ε] 范围时，目标函数变平（梯度为 0）
- 这阻止了策略的过大更新
- 相比 TRPO 的硬约束，PPO 用软裁剪实现了类似效果

### 3.4 PPO 算法完整流程

```
算法：PPO

初始化策略参数 θ 和价值参数 φ

循环：
    1. 使用当前策略 π_θ 收集 N 个时间步的轨迹
    2. 计算优势函数估计 Â_t（使用 GAE）
    3. 计算回报 G_t
    4. 重复 K 个 epoch：
        for each minibatch:
            # 策略损失
            r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
            L^{CLIP} = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)]
            
            # 价值损失
            L^{VF} = E[(V_φ(s_t) - G_t)²]
            
            # 熵奖励（可选）
            S = E[entropy(π_θ(·|s_t))]
            
            # 总损失
            L = -L^{CLIP} + c_1 L^{VF} - c_2 S
            
            # 梯度下降
            θ, φ ← optimize(L)
    
    5. 更新旧策略：θ_old ← θ
```

### 3.5 完整 PPO 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal

class ActorCritic(nn.Module):
    """PPO 的 Actor-Critic 网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        super().__init__()
        self.continuous = continuous
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        if continuous:
            # 连续动作：输出均值和标准差
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作：输出动作概率
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # 价值函数
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std)
            value = self.critic(features)
            return action_mean, action_std, value
        else:
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value
    
    def get_action(self, state, deterministic=False):
        """获取动作和相关信息"""
        if self.continuous:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)
        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
        
        if deterministic:
            if self.continuous:
                action = action_mean
            else:
                action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions):
        """评估给定状态-动作对的概率和价值"""
        if self.continuous:
            action_mean, action_std, values = self.forward(states)
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, values = self.forward(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class PPO:
    """PPO 算法完整实现"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        continuous=False,
        lr=3e-4,
        gamma=0.99,
        lambda_gae=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        minibatch_size=64,
        normalize_advantage=True
    ):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantage = normalize_advantage
        
        # 网络
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim, continuous)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.policy.get_action(state)
        
        return action.squeeze(0).numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_state):
        """PPO 更新"""
        # 计算最后状态的价值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, _, next_value = self.policy.get_action(next_state_tensor)
            next_value = next_value.item()
        
        # 计算 GAE 和回报
        advantages = self.compute_gae(next_value)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions)) if self.policy.continuous else torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 归一化优势
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        dataset_size = len(self.states)
        
        for _ in range(self.update_epochs):
            # 随机打乱
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                
                # 获取小批量数据
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # 评估当前策略
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(mb_states, mb_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # PPO-Clip 损失
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()
                
                # 熵奖励
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 清空存储
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def train(self, env, total_timesteps=100000, horizon=2048):
        """训练"""
        timestep = 0
        episode = 0
        rewards_history = []
        
        while timestep < total_timesteps:
            state = env.reset()
            episode_reward = 0
            
            for step in range(horizon):
                # 选择动作
                action, log_prob, value = self.select_action(state)
                
                # 执行动作
                if self.policy.continuous:
                    next_state, reward, done, _ = env.step(action)
                else:
                    next_state, reward, done, _ = env.step(int(action))
                
                # 存储转移
                self.store_transition(state, action, reward, value, log_prob, done)
                
                episode_reward += reward
                timestep += 1
                state = next_state
                
                if done:
                    rewards_history.append(episode_reward)
                    episode += 1
                    
                    if episode % 10 == 0:
                        avg_reward = np.mean(rewards_history[-10:])
                        print(f"Episode {episode}, Timestep {timestep}, Avg Reward: {avg_reward:.2f}")
                    
                    state = env.reset()
                    episode_reward = 0
            
            # 更新策略
            self.update(state)
        
        return rewards_history
```

### 3.6 PPO 变体

**PPO-Penalty（KL 惩罚版本）**：

$$L^{KLPEN}(\theta) = \mathbb{E}_t [r_t(\theta) \hat{A}_t] - \beta \mathbb{E}_t [D_{KL}(\pi_{\theta_{old}}, \pi_\theta)]$$

其中 β 是自适应惩罚系数：
- 如果 KL > KL_target，增大 β
- 如果 KL < KL_target / 1.5，减小 β

```python
class PPOPenalty(PPO):
    """PPO-Penalty 实现"""
    
    def __init__(self, *args, kl_target=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_target = kl_target
        self.beta = 1.0
    
    def update(self, next_state):
        """带 KL 惩罚的更新"""
        # ...（省略前面相同的代码）
        
        for _ in range(self.update_epochs):
            # ...（省略小批量循环开始）
            
            # 计算比率
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            
            # 策略损失（无裁剪）
            policy_loss = -(ratio * mb_advantages).mean()
            
            # KL 散度（近似）
            kl = (mb_old_log_probs - new_log_probs).mean()
            
            # 总损失
            loss = policy_loss + self.beta * kl + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # ...（省略更新代码）
        
        # 自适应调整 beta
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(self.states))
            old_log_probs_tensor = torch.FloatTensor(self.log_probs)
            new_log_probs, _, _ = self.policy.evaluate_actions(states_tensor, actions)
            kl = (old_log_probs_tensor - new_log_probs).mean().item()
        
        if kl > self.kl_target * 2:
            self.beta *= 2
        elif kl < self.kl_target / 2:
            self.beta /= 2
```

---

## 4. 其他现代策略优化算法

### 4.1 SAC：软Actor-Critic

**SAC（Soft Actor-Critic）** 是一种最大熵强化学习算法。

**核心思想**：在最大化回报的同时，最大化策略的熵。

$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

其中 $\mathcal{H}(\pi(\cdot|s)) = -\sum_a \pi(a|s) \log \pi(a|s)$ 是熵，α 是温度参数。

**优势**：
1. **更好的探索**：熵最大化鼓励探索
2. **更鲁棒**：对超参数和模型误差更稳定
3. **样本效率高**：离策略学习，可重复使用经验

**SAC 的三个网络**：
- 策略网络 π_φ
- 两个 Q 网络 Q_θ₁, Q_θ₂
- 两个目标 Q 网络 Q_θ₁⁻, Q_θ₂⁻

```python
class SAC:
    """Soft Actor-Critic 实现"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy=True,
        target_entropy=None
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        
        # 策略网络
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Q 网络（两个）
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # 目标 Q 网络
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 自动温度调整
        if auto_entropy:
            if target_entropy is None:
                target_entropy = -action_dim
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=1000000)
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if deterministic:
                action = self.policy.get_mean(state)
            else:
                action, _, _ = self.policy.sample(state)
        return action.squeeze(0).numpy()
    
    def update(self, batch_size=256):
        """SAC 更新"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新 Q 网络
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next
        
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = nn.MSELoss()(q1_pred, target_q)
        q2_loss = nn.MSELoss()(q2_pred, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # 更新策略网络
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新温度
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### 4.2 TD3：双延迟深度确定性策略梯度

**TD3（Twin Delayed DDPG）** 解决 DDPG 的过高估计问题。

**三大改进**：
1. **双 Q 网络**：取两个 Q 值的最小值，减少过高估计
2. **延迟更新**：策略更新频率低于 Q 网络
3. **目标策略平滑**：给目标动作添加噪声

```python
class TD3:
    """Twin Delayed DDPG 实现"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # 策略网络
        self.actor = DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim)
        self.actor_target = DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Q 网络（两个）
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=lr
        )
        
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.total_steps = 0
    
    def select_action(self, state, noise=0.1):
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state).squeeze(0).numpy()
            if noise > 0:
                action = action + np.random.normal(0, noise, size=action.shape)
                action = np.clip(action, -self.max_action, self.max_action)
        return action
    
    def update(self, batch_size=256):
        """TD3 更新"""
        if len(self.replay_buffer) < batch_size:
            return
        
        self.total_steps += 1
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 目标策略平滑
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # 双 Q 值取最小
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * q_next
        
        # 更新 Critic
        q1_pred = self.critic1(states, actions)
        q2_pred = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(q1_pred, target_q) + nn.MSELoss()(q2_pred, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新 Actor
        if self.total_steps % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

---

## 5. 算法对比与选择指南

### 5.1 算法特点对比

| 算法 | 类型 | 动作空间 | 样本效率 | 稳定性 | 探索方式 |
|------|------|----------|----------|--------|----------|
| **TRPO** | 在策略 | 离散/连续 | 低 | 高 | 策略随机性 |
| **PPO** | 在策略 | 离散/连续 | 低 | 高 | 策略随机性 |
| **SAC** | 离策略 | 连续 | 高 | 高 | 熵最大化 |
| **TD3** | 离策略 | 连续 | 高 | 高 | 高斯噪声 |

### 5.2 选择决策树

```
动作空间类型？
├── 离散动作
│   ├── 简单任务 → PPO
│   └── 复杂任务 → PPO + 复杂网络
│
└── 连续动作
    ├── 样本效率要求高？
    │   ├── 是 → SAC 或 TD3
    │   └── 否 → PPO
    │
    └── 任务特点
        ├── 需要鲁棒性 → SAC（熵正则化）
        ├── 需要确定性策略 → TD3
        └── 一般任务 → PPO
```

### 5.3 超参数推荐

**PPO 推荐配置**：

```python
PPO_CONFIG = {
    'learning_rate': 3e-4,        # 学习率
    'gamma': 0.99,                # 折扣因子
    'lambda_gae': 0.95,           # GAE 参数
    'clip_epsilon': 0.2,          # PPO 裁剪参数
    'value_coef': 0.5,            # 价值损失系数
    'entropy_coef': 0.01,         # 熵正则化系数
    'max_grad_norm': 0.5,         # 梯度裁剪
    'update_epochs': 10,          # 每批数据更新次数
    'minibatch_size': 64,         # 小批量大小
    'horizon': 2048,              # 每次采集步数
}
```

**SAC 推荐配置**：

```python
SAC_CONFIG = {
    'learning_rate': 3e-4,        # 学习率
    'gamma': 0.99,                # 折扣因子
    'tau': 0.005,                 # 软更新系数
    'alpha': 0.2,                 # 温度参数（自动调整）
    'auto_entropy': True,         # 自动调整温度
    'batch_size': 256,            # 批大小
    'buffer_size': 1000000,       # 经验回放大小
}
```

### 5.4 实践技巧

**训练稳定性**：
1. **归一化优势函数**：`advantages = (advantages - mean) / std`
2. **梯度裁剪**：防止梯度爆炸
3. **学习率调度**：训练后期降低学习率
4. **观察 KL 散度**：监控策略变化

**样本效率**：
1. **经验回放**（离策略算法）：重复使用经验
2. **n-step return**：平衡偏差和方差
3. **优先级回放**：优先采样重要经验

**探索策略**：
1. **熵正则化**：鼓励策略多样性
2. **噪声探索**：TD3 的高斯噪声
3. **参数噪声**：直接扰动策略参数

---

## 6. 思考题

### 概念理解

1. **PPO 的裁剪机制如何保证策略不会变化太大？为什么不是硬约束而是软裁剪？**

2. **比较 PPO 和 TRPO 的优劣，为什么 PPO 在实践中更受欢迎？**

3. **SAC 的熵最大化有什么好处？温度参数 α 如何影响探索？**

### 数学推导

4. **证明：当策略变化很小时，重要性采样比率 $r_t(\theta) \approx 1 + \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (\theta - \theta_{old})$**

5. **推导 TRPO 的自然梯度更新：$\theta_{new} = \theta_{old} + \frac{1}{\lambda} F^{-1} g$，其中 F 是 Fisher 信息矩阵，g 是策略梯度。**

6. **分析 PPO-Clip 目标函数的梯度：当 $r_t > 1 + \epsilon$ 且 $A_t > 0$ 时，梯度为多少？**

### 编程实践

7. **实现 PPO 并在 CartPole 环境中训练，比较不同 clip_epsilon（0.1, 0.2, 0.3）的效果。**

8. **比较 PPO 和 SAC 在连续控制任务（如 Pendulum、HalfCheetah）中的样本效率。**

9. **实验 GAE 中不同的 λ 值（0, 0.5, 0.95, 1）对学习效果的影响。**

---

## 7. 拓展阅读

### 经典论文

- Schulman, J., et al. (2015). *Trust Region Policy Optimization (TRPO)*. ICML.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.
- Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*. ICML.
- Fujimoto, S., et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML (TD3).

### 推荐资源

- OpenAI Spinning Up: PPO, TRPO, SAC, TD3
- Stable Baselines3: 生产级实现
- CleanRL: 单文件实现，适合学习

### 进阶主题

- **PPO+GAE**：结合广义优势估计
- **SAC+自动温度**：自适应调整熵权重
- **分布式 PPO**：大规模并行训练
- **PPO+LSTM**：处理部分可观测环境

---

## 小结

**Day 8 核心要点**：

| 概念 | 定义 | 重要性 |
|------|------|--------|
| **信任区域** | 约束策略更新的范围 | ⭐⭐⭐⭐⭐ |
| **PPO-Clip** | 裁剪目标函数控制更新 | ⭐⭐⭐⭐⭐ |
| **GAE** | 广义优势估计 | ⭐⭐⭐⭐⭐ |
| **SAC** | 最大熵强化学习 | ⭐⭐⭐⭐⭐ |
| **TD3** | 双延迟DDPG | ⭐⭐⭐⭐ |

**核心公式速查**：

| 公式 | 表达式 |
|------|--------|
| PPO-Clip | $L = \mathbb{E}[\min(r\hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon)\hat{A})]$ |
| 重要性采样比率 | $r_t(\theta) = \frac{\pi_\theta(a_t\|s_t)}{\pi_{\theta_{old}}(a_t\|s_t)}$ |
| GAE | $A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$ |
| SAC 目标 | $J = \mathbb{E}[r + \alpha\mathcal{H}(\pi)]$ |
| TD3 双Q | $y = r + \gamma \min(Q_1, Q_2)$ |

**一句话总结**：现代策略优化方法通过约束策略更新幅度（TRPO、PPO）或改进探索和稳定性（SAC、TD3），解决了策略梯度训练不稳定的问题，成为当今最实用的深度强化学习算法。

---

**下一讲预告**：Day 9 将探讨强化学习在大语言模型训练中的应用——RLHF 与 DPO，这是 ChatGPT 等模型成功的关键技术。

---

*Day 8 完成！ 🎉*