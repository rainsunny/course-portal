# Day 12: 注意力机制深入 - 从RNN到Transformer的桥梁

## 核心问题

**自注意力机制的"自"是什么意思？为什么说注意力机制彻底改变了NLP？位置编码的必要性从何而来？**

---

## 一、从RNN注意力到自注意力

### 1.1 RNN注意力的回顾

**RNN+Attention的结构：**

```
编码器：h₁, h₂, ..., h_T （RNN隐藏状态）
解码器：s₁, s₂, ..., s_T' （解码器隐藏状态）

注意力计算：
e_tj = score(s_t, h_j)     # 解码器状态与编码器状态的相关性
α_tj = softmax(e_t)        # 归一化
c_t = Σ α_tj · h_j          # 上下文向量

关键：解码器状态 s_t 作为查询，编码器状态 h_j 作为键和值
```

**局限性：**

```
问题1：依赖RNN结构
├── 编码器需要RNN生成隐藏状态
├── 解码器需要RNN生成查询状态
└── 无法摆脱序列计算的限制

问题2：计算效率
├── RNN必须顺序计算
├── 无法充分利用并行化
└── 训练时间长

问题3：距离限制
├── 虽然注意力缓解了长距离依赖
├── 但RNN隐藏状态本身仍有信息损失
└── 非常长的序列仍有问题
```

### 1.2 自注意力的核心思想

**关键洞察：为什么需要RNN？**

```
RNN的作用：
├── 生成隐藏状态序列
├── 提供序列中每个位置的信息
└── 作为注意力的键和值

问题：序列元素本身不能直接作为键和值吗？

答案：可以！
├── 不需要RNN生成中间表示
├── 直接对输入序列计算注意力
└── 这就是自注意力
```

**自注意力定义：**

```
输入：X = [x₁, x₂, ..., x_n] （序列的向量表示）

自注意力：序列中的每个元素都"关注"序列中的所有其他元素

输出：Y = [y₁, y₂, ..., y_n]
其中 y_i = Attention(x_i, X, X)

意义：每个位置都可以直接看到所有其他位置
```

**对比：**

| 特性 | RNN+Attention | Self-Attention |
|------|---------------|----------------|
| 编码器 | RNN | 无需 |
| 键/值来源 | RNN隐藏状态 | 输入本身 |
| 查询来源 | 解码器RNN状态 | 输入本身 |
| 并行性 | 低（顺序计算） | 高（完全并行） |
| 路径长度 | O(n) | O(1) |
| 复杂度 | O(n) | O(n²) |

---

## 二、自注意力的数学形式

### 2.1 缩放点积注意力

**基础公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ (Query): 查询矩阵，形状 $(n, d_k)$
- $K$ (Key): 键矩阵，形状 $(n, d_k)$
- $V$ (Value): 值矩阵，形状 $(n, d_v)$
- $d_k$: 键/查询的维度

**详细计算步骤：**

```
步骤1：计算注意力分数
scores = Q @ K^T           # (n, n)
每个元素 scores[i,j] 表示位置 i 对位置 j 的关注程度

步骤2：缩放
scores = scores / sqrt(d_k)
为什么缩放？防止点积过大导致softmax梯度消失

步骤3：归一化
attention_weights = softmax(scores, dim=-1)  # (n, n)
每行是一个概率分布，和为1

步骤4：加权求和
output = attention_weights @ V   # (n, d_v)
每个位置的输出是所有值的加权和
```

**为什么需要缩放？**

```python
import torch
import math

# 假设 Q 和 K 的元素服从标准正态分布
d_k = 512
Q = torch.randn(1, 10, d_k)
K = torch.randn(1, 10, d_k)

# 点积的期望和方差
# E[q·k] = 0
# Var[q·k] = d_k （假设元素独立）

scores = Q @ K.transpose(-2, -1)
print(f"点积的方差：{scores.var().item():.2f}")  # 约等于 d_k

# 不缩放的softmax
softmax_no_scale = torch.softmax(scores, dim=-1)
print(f"不缩放的softmax最大值：{softmax_no_scale.max().item():.4f}")  # 接近1
print(f"不缩放的softmax梯度：{softmax_no_scale.grad_fn}")  # 梯度很小

# 缩放后
scores_scaled = scores / math.sqrt(d_k)
softmax_scaled = torch.softmax(scores_scaled, dim=-1)
print(f"缩放后的softmax最大值：{softmax_scaled.max().item():.4f}")  # 更平滑

# 直觉：点积值太大 → softmax接近one-hot → 梯度接近0
```

### 2.2 自注意力的完整计算

**自注意力 = Q, K, V 都来自同一输入：**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K, W^V$ 是可学习的投影矩阵。

**完整计算流程：**

```
输入：X = [x₁, x₂, ..., x_n]，形状 (n, d_model)

步骤1：线性投影
Q = X @ W_Q    # (n, d_model) @ (d_model, d_k) = (n, d_k)
K = X @ W_K    # (n, d_k)
V = X @ W_V    # (n, d_v)

步骤2：计算注意力
scores = Q @ K^T           # (n, n)
scores = scores / sqrt(d_k)
attention_weights = softmax(scores)

步骤3：加权求和
output = attention_weights @ V   # (n, d_v)

如果 d_v != d_model，还需要输出投影：
output = output @ W_O    # (n, d_model)
```

**PyTorch实现：**

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.scale = math.sqrt(d_k)
        
    def forward(self, X, mask=None):
        """
        X: (batch, n, d_model)
        mask: (batch, n, n) 可选的注意力掩码
        """
        # 线性投影
        Q = self.W_Q(X)  # (batch, n, d_k)
        K = self.W_K(X)  # (batch, n, d_k)
        V = self.W_V(X)  # (batch, n, d_v)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n, n)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 归一化
        attention_weights = torch.softmax(scores, dim=-1)  # (batch, n, n)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, n, d_v)
        
        return output, attention_weights


# 使用示例
d_model, d_k, d_v = 512, 64, 64
self_attn = SelfAttention(d_model, d_k, d_v)

X = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
output, weights = self_attn(X)

print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
```

### 2.3 自注意力的直观理解

**每个位置都在"询问"所有其他位置：**

```
句子："The cat sat on the mat"

位置3 "sat" 的自注意力计算：
├── Q₃：作为"查询"，询问"我在句子中是什么角色？"
├── K₁, K₂, K₄, K₅, K₆：作为"键"，提供"我是什么"的信息
├── 计算相关性：
│   ├── Q₃ · K₁ (The)：低相关（冠词与动词关系弱）
│   ├── Q₃ · K₂ (cat)：高相关（主语）
│   ├── Q₃ · K₄ (on)：中等相关（介词）
│   └── ...
├── V₁, V₂, ...：作为"值"，提供具体信息
└── y₃ = Σ α₃ⱼ · Vⱼ：加权组合得到"sat"的表示

结果："sat"的表示融合了"cat"（主语）的信息
```

**可视化注意力权重：**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """
    attention_weights: (seq_len, seq_len)
    tokens: 词元列表
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weights')
    plt.tight_layout()
    plt.show()

# 示例
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
# 假设的注意力权重（"sat"主要关注"cat"）
weights = torch.tensor([
    [0.7, 0.1, 0.1, 0.05, 0.03, 0.02],
    [0.2, 0.6, 0.1, 0.05, 0.03, 0.02],
    [0.1, 0.5, 0.2, 0.1, 0.05, 0.05],  # "sat"关注"cat"
    [0.1, 0.1, 0.3, 0.3, 0.1, 0.1],
    [0.05, 0.05, 0.05, 0.1, 0.6, 0.15],
    [0.05, 0.2, 0.1, 0.1, 0.15, 0.4]
])

visualize_attention(weights, tokens)
```


---

## 三、多头注意力机制

### 3.1 为什么需要多头？

**单头注意力的局限：**

```
问题：每个位置只能学习一种"关注模式"

例子：句子 "The animal didn't cross the street because it was too tired"

"it" 这个词需要关注：
├── "animal"（它指代动物）
├── "tired"（为什么累）
└── "didn't cross"（动作关系）

单头注意力：只能学习一个权重分布
├── 可能主关注"animal"
├── 其他关系被忽略
└── 信息丢失

解决方案：多头注意力
├── 每个头学习不同的关注模式
├── 头1：关注主语
├── 头2：关注属性
├── 头3：关注动作
└── 最后融合所有头的信息
```

**类比：**

```
单头注意力 = 单一视角看问题
├── 像只从"语法"角度理解句子
└── 忽略语义、情感等其他方面

多头注意力 = 多个专家从不同角度分析
├── 专家1：语法专家
├── 专家2：语义专家
├── 专家3：情感专家
└── 综合所有专家意见
```

### 3.2 多头注意力的数学形式

**公式：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**计算流程：**

```
输入：X = [x₁, ..., x_n]，形状 (n, d_model)

参数：
├── h 个注意力头
├── 每个头的维度：d_k = d_model / h
└── 总维度保持：h × d_k = d_model

每个头独立计算：
head_i = Attention(X @ W_i^Q, X @ W_i^K, X @ W_i^V)

拼接：
concat = [head_1; head_2; ...; head_h]  # (n, d_model)

输出投影：
output = concat @ W^O  # (n, d_model)
```

**参数量分析：**

```
假设 d_model = 512, h = 8, d_k = 64

单头注意力参数：
W_Q, W_K, W_V: 512 × 64 × 3 = 98,304

多头注意力参数（8头）：
每个头：512 × 64 × 3 = 98,304
8个头：98,304 × 8 = 786,432
输出投影 W_O：512 × 512 = 262,144
总计：1,048,576

注意：参数量主要取决于总维度，而不是头数！
```

### 3.3 PyTorch实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 所有头的Q, K, V投影（合并为一个矩阵效率更高）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, X, mask=None):
        """
        X: (batch, n, d_model)
        mask: (batch, n, n) 或 (batch, 1, n, n)
        """
        batch_size, seq_len, _ = X.shape
        
        # 1. 线性投影
        Q = self.W_Q(X)  # (batch, n, d_model)
        K = self.W_K(X)
        V = self.W_V(X)
        
        # 2. 分割成多个头
        # (batch, n, d_model) -> (batch, n, num_heads, d_k) -> (batch, num_heads, n, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力（每个头独立）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, num_heads, n, n)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 4. 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, num_heads, n, d_k)
        
        # 5. 拼接所有头
        # (batch, num_heads, n, d_k) -> (batch, n, num_heads, d_k) -> (batch, n, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. 输出投影
        output = self.W_O(output)
        
        return output, attention_weights


# 使用示例
d_model, num_heads = 512, 8
multihead_attn = MultiHeadAttention(d_model, num_heads)

X = torch.randn(2, 10, d_model)
output, weights = multihead_attn(X)

print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")  # (batch, num_heads, n, n)
```

### 3.4 多头注意力的可视化

```python
def visualize_multihead_attention(attention_weights, tokens, num_heads_to_show=4):
    """
    attention_weights: (num_heads, seq_len, seq_len)
    tokens: 词元列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flat
    
    for i in range(min(num_heads_to_show, len(attention_weights))):
        sns.heatmap(attention_weights[i].detach().numpy(),
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='Blues',
                    ax=axes[i],
                    cbar=True)
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key')
        axes[i].set_ylabel('Query')
    
    plt.tight_layout()
    plt.show()

# 示例：不同头关注不同模式
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
num_heads = 8
seq_len = len(tokens)

# 模拟不同头的注意力模式
attention_weights = torch.zeros(num_heads, seq_len, seq_len)

# 头0：关注前一个词（局部）
for i in range(seq_len):
    attention_weights[0, i, max(0, i-1)] = 1.0

# 头1：关注后一个词（局部）
for i in range(seq_len):
    attention_weights[1, i, min(seq_len-1, i+1)] = 1.0

# 头2：关注句子开头
attention_weights[2, :, 0] = 1.0

# 头3：关注句子结尾
attention_weights[3, :, -1] = 1.0

visualize_multihead_attention(attention_weights, tokens)
```


---

## 四、位置编码

### 4.1 为什么需要位置编码？

**自注意力的"位置盲"问题：**

```
自注意力计算：
y_i = Σ α_ij · x_j

关键：位置 i 和位置 j 的关系仅由内容决定，与位置无关

例子：
输入1：["我", "爱", "你"]
输入2：["你", "爱", "我"]

自注意力处理：
├── 两个输入的注意力模式可能完全相同
├── 因为"我"、"爱"、"你"的语义相同
└── 无法区分语序

问题：自注意力是置换不变的！
├── Permutation(X) 得到相同的注意力输出
└── 丢失了位置信息
```

**对比：**

```
RNN：
├── 顺序处理，位置隐含在处理顺序中
├── 第1个输入永远在第1步处理
└── 位置信息天然存在

CNN：
├── 卷积核在固定位置滑动
├── 不同位置的感受野不同
└── 位置信息隐含在卷积中

Self-Attention：
├── 所有位置同时处理
├── 没有内在的位置信息
└── 必须显式添加位置编码
```

### 4.2 正弦位置编码

**Transformer原论文的位置编码：**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：位置索引（0, 1, 2, ...）
- $i$：维度索引（0, 1, ..., d_model/2 - 1）

**为什么这样设计？**

```
特性1：每个位置有唯一编码
├── 不同位置的编码不同
└── 可以区分位置

特性2：相对位置关系
├── PE(pos + k) 可以表示为 PE(pos) 的线性函数
├── 模型可以学习相对位置关系
└── 这对语言处理很重要

特性3：泛化能力
├── 可以处理训练时未见过的序列长度
├── 正弦函数可以外推
└── 没有固定长度限制

特性4：值域有界
├── sin 和 cos 的值域都是 [-1, 1]
├── 不会主导词嵌入
└── 与词嵌入相加后保持合理范围
```

**数学推导：相对位置关系**

```
设 PE(pos, 2i) = sin(pos / 10000^(2i/d))
设 PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

对于位置 pos + k：
PE(pos + k, 2i) = sin((pos + k) / 10000^(2i/d))

使用三角恒等式：
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)

因此：
PE(pos + k, 2i) = PE(pos, 2i) · cos(k/...) + PE(pos, 2i+1) · sin(k/...)

这意味着：
PE(pos + k) 可以表示为 PE(pos) 的线性变换
模型可以学习这种线性关系，从而理解相对位置
```

### 4.3 PyTorch实现

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不参与训练）
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 将位置编码加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return x


# 可视化位置编码
def plot_positional_encoding(d_model=512, max_len=100):
    pe = PositionalEncoding(d_model, max_len)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe.pe[0, :, :].numpy(), aspect='auto', cmap='viridis')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

plot_positional_encoding(d_model=128, max_len=50)
```

### 4.4 其他位置编码方法

**1. 可学习位置编码：**

```python
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 可学习的位置嵌入
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)
```

**优缺点：**

| 方法 | 优点 | 缺点 |
|------|------|------|
| 正弦编码 | 可外推、无需学习 | 固定模式、可能不最优 |
| 可学习编码 | 灵活、可优化 | 不能外推、有长度限制 |

**2. 旋转位置编码（RoPE）：**

```
核心思想：通过旋转矩阵编码相对位置

优点：
├── 理论上更优雅
├── 更好的外推能力
├── 在长序列上表现更好
└── 被LLaMA、GLM等模型采用
```

**3. 相对位置编码：**

```
不编码绝对位置，而是编码相对距离

注意力分数：
e_ij = x_i^T · x_j + a_{i-j}

其中 a_{i-j} 是可学习的相对位置偏置
```


---

## 五、注意力机制的变体

### 5.1 交叉注意力（Cross-Attention）

**定义：**

```
自注意力：Q, K, V 来自同一输入
交叉注意力：Q 来自一个输入，K, V 来自另一个输入

应用场景：
├── 机器翻译：解码器Q关注编码器输出(K, V)
├── 图像描述：文本Q关注图像特征(K, V)
└── 文本摘要：摘要Q关注原文(K, V)
```

**计算过程：**

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, query_input, key_value_input, mask=None):
        """
        query_input: (batch, n_q, d_model) - 查询来源
        key_value_input: (batch, n_kv, d_model) - 键值来源
        """
        # Q来自query_input，K和V来自key_value_input
        # 这在MultiHeadAttention内部通过不同的投影实现
        # 或者显式传递：
        
        batch_size, n_q, _ = query_input.shape
        _, n_kv, _ = key_value_input.shape
        
        # 投影
        Q = self.mha.W_Q(query_input)      # (batch, n_q, d_model)
        K = self.mha.W_K(key_value_input)  # (batch, n_kv, d_model)
        V = self.mha.W_V(key_value_input)  # (batch, n_kv, d_model)
        
        # 后续计算与自注意力相同...
```

**机器翻译中的交叉注意力：**

```
编码器输出：H_enc = [h_1, h_2, ..., h_n]
解码器状态：H_dec = [s_1, s_2, ..., s_m]

交叉注意力：
Q = H_dec · W_Q    # 解码器生成查询
K = H_enc · W_K    # 编码器提供键
V = H_enc · W_V    # 编码器提供值

意义：
├── 解码器的每个位置都可以关注源语言的任意位置
├── 实现软对齐
└── 类似之前RNN+Attention的机制，但更高效
```

### 5.2 因果注意力（Causal Attention）

**问题：自回归生成中的信息泄露**

```
自回归生成：逐词生成
├── 生成第t个词时，只能看到前t-1个词
├── 不能看到未来的词
└── 否则就是"作弊"

自注意力默认行为：
├── 每个位置看到所有位置
├── 包括未来的位置
└── 训练时没问题，但会导致推理不一致

解决方案：因果掩码
├── 位置i只能关注位置1到i
├── 用掩码将未来位置的注意力分数设为负无穷
└── softmax后这些位置的权重为0
```

**掩码实现：**

```python
def create_causal_mask(seq_len):
    """
    创建因果掩码
    上三角（不包括对角线）为0，其他为1
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)  # (1, seq_len, seq_len)


# 使用示例
seq_len = 5
causal_mask = create_causal_mask(seq_len)
print(causal_mask)
# tensor([[[1., 0., 0., 0., 0.],
#          [1., 1., 0., 0., 0.],
#          [1., 1., 1., 0., 0.],
#          [1., 1., 1., 1., 0.],
#          [1., 1., 1., 1., 1.]]])

# 在注意力计算中应用
scores = Q @ K.transpose(-2, -1) / scale
scores = scores.masked_fill(causal_mask == 0, float('-inf'))
attention_weights = torch.softmax(scores, dim=-1)
# 未来位置的权重变为0
```

**因果注意力可视化：**

```
注意力矩阵（因果掩码）：

位置：  1  2  3  4  5
     ┌─────────────────┐
   1 │ 1  0  0  0  0  │  位置1只能看自己
   2 │ *  1  0  0  0  │  位置2可以看1, 2
   3 │ *  *  1  0  0  │  位置3可以看1, 2, 3
   4 │ *  *  *  1  0  │  位置4可以看1, 2, 3, 4
   5 │ *  *  *  *  1  │  位置5可以看所有
     └─────────────────┘

* 表示可以关注的权重，0表示被掩码遮蔽
```

### 5.3 高效注意力变体

**标准注意力的复杂度问题：**

```
序列长度为n时：
├── 注意力矩阵：n × n
├── 时间复杂度：O(n²)
├── 空间复杂度：O(n²)
└── 长序列时不可承受

例子：
├── n = 512: 可接受
├── n = 2048: 较慢
├── n = 8192: 非常慢
└── n = 32768: 几乎无法训练
```

**高效注意力方法：**

**1. 稀疏注意力（Sparse Attention）**

```
思想：不是所有位置都需要关注所有位置

模式：
├── 局部注意力：只关注附近的k个位置
├── 步幅注意力：每隔s个位置关注一次
├── 全局注意力：某些特殊位置关注所有位置
└── 随机注意力：随机选择一些位置关注

复杂度：O(n · k) 或 O(n · √n)
```

**2. 线性注意力（Linear Attention）**

```
核心思想：利用矩阵乘法结合律

标准注意力：
Attention(Q, K, V) = softmax(QK^T) V
= softmax(QK^T / √d) V

线性注意力：
利用 kernel 函数 φ 近似 softmax
= φ(Q) (φ(K)^T V)
= (φ(Q) φ(K)^T) V  ← O(n²d)
= φ(Q) (φ(K)^T V)  ← O(nd²)

当 n > d 时，复杂度从 O(n²d) 降为 O(nd²)
```

**3. Flash Attention**

```
思想：优化内存访问模式

标准注意力：
├── 先计算 n×n 的注意力矩阵
├── 存储到内存
├── 再与V相乘
└── 内存占用大，访问慢

Flash Attention：
├── 分块计算
├── 不存储完整的注意力矩阵
├── 在GPU高速缓存中完成计算
└── 内存占用从O(n²)降为O(n)

效果：
├── 内存效率提升
├── 计算速度提升
├── 结果完全相同（数值等价）
└── 已成为PyTorch 2.0的默认实现
```

**Flash Attention使用：**

```python
# PyTorch 2.0+ 内置支持
import torch.nn.functional as F

# 使用scaled_dot_product_attention
# 自动选择最优实现（包括Flash Attention）
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

# 手动启用Flash Attention（PyTorch 2.0+）
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V)
```

### 5.4 注意力机制的应用场景

**1. 自然语言处理**

```
├── 机器翻译：Cross-Attention连接编码器和解码器
├── 文本摘要：理解全文，生成摘要
├── 问答系统：问题关注文档的相关部分
└── 情感分析：关注情感词
```

**2. 计算机视觉**

```
Vision Transformer (ViT)：
├── 图像分块，每块作为一个token
├── 使用自注意力处理块序列
└── 性能媲美CNN

图像描述：
├── 图像特征作为K, V
├── 文本生成作为Q
└── Cross-Attention生成描述
```

**3. 多模态**

```
CLIP：
├── 图像编码器 + 文本编码器
├── 图像和文本在共享空间对比学习
└── 零样本分类

DALL-E：
├── 文本作为条件
├── 生成图像
└── 使用Cross-Attention融合文本信息
```


---

## 六、思考题

<details>
<summary>思考题1：自注意力机制的时间复杂度和空间复杂度是多少？为什么长序列任务需要特殊处理？有哪些优化方法？</summary>

**答案：**

**复杂度分析：**

```
设序列长度为n，特征维度为d

自注意力计算：
1. Q, K, V投影：O(n · d²)
2. QK^T计算：O(n² · d)
3. Softmax：O(n²)
4. 与V相乘：O(n² · d)
5. 输出投影：O(n · d²)

主导项：O(n² · d)
通常 d << n，所以简化为 O(n²)
```

**空间复杂度：**

```
主要开销：
├── 注意力矩阵：n × n
├── 中间结果存储
└── 空间复杂度：O(n²)

示例：
├── n=1024: 注意力矩阵约4MB（float32）
├── n=4096: 注意力矩阵约64MB
├── n=16384: 注意力矩阵约1GB
└── n=65536: 注意力矩阵约16GB（单层就耗尽显存）
```

**长序列问题：**

```
问题1：内存瓶颈
├── n² 增长迅速
├── GPU显存有限
└── 无法处理长文档、高分辨率图像

问题2：计算瓶颈
├── O(n²)计算量
├── 训练时间长
└── 推理延迟高

问题3：注意力稀疏
├── 实际上大多数位置对的注意力权重很小
├── 计算了很多无效的注意力
└── 效率低下
```

**优化方法：**

**1. 稀疏注意力**
```
只关注部分位置：
├── 局部窗口：O(n · w)，w是窗口大小
├── 步幅采样：O(n · n/s)，s是步幅
├── 组合模式：局部+全局+随机

代表：Longformer, BigBird, Sparse Transformer
```

**2. 线性注意力**
```
利用 kernel 技巧：
softmax(qk^T) ≈ φ(q)φ(k)^T
Attention(Q, K, V) ≈ φ(Q)(φ(K)^T V)

复杂度：O(nd²)
当 n > d 时，显著降低

代表：Linear Transformer, Performer
```

**3. Flash Attention**
```
不是改变算法，而是优化实现：
├── 分块计算，避免存储完整注意力矩阵
├── 利用GPU内存层次结构
├── IO感知的算法设计

空间：O(n)（从O(n²)降低）
速度：2-4倍提升
精度：数值等价
```

**4. 低秩近似**
```
注意力矩阵通常是低秩的：
├── 用低秩矩阵近似注意力矩阵
├── 或用随机投影降维

代表：Linformer
复杂度：O(n · k)，k是投影维度
```

**选择建议：**

| 序列长度 | 推荐方法 |
|---------|---------|
| < 512 | 标准注意力 |
| 512-2048 | Flash Attention |
| 2048-8192 | Flash Attention + 稀疏 |
| > 8192 | 线性注意力 / 分块处理 |

</details>

<details>
<summary>思考题2：为什么Transformer使用正弦位置编码而不是可学习位置编码？两种方法各有什么优缺点？</summary>

**答案：**

**正弦位置编码的设计理由：**

```
理由1：外推能力
├── 正弦函数定义在实数域
├── 可以处理任意长度的序列
├── 训练时见过长度n，推理时可以处理n+k
└── 不受最大序列长度限制

理由2：相对位置编码
├── PE(pos+k)可以表示为PE(pos)的线性函数
├── 模型可以学习相对位置关系
├── 对语言任务很重要（"前一个词"、"后两个词"）
└── 数学上优雅

理由3：无需学习
├── 不增加模型参数
├── 不受训练数据影响
└── 零成本泛化
```

**可学习位置编码的优缺点：**

```
优点：
├── 灵活性高
│   └── 可以学习任务最优的位置表示
├── 适应性强
│   └── 自动适应数据分布
└── 实现简单
    └── nn.Embedding即可

缺点：
├── 无法外推
│   └── 训练时没见过的位置无编码
│   └── 序列长度受限
├── 增加参数
│   └── 每个位置需要d_model个参数
│   └── max_len=512, d_model=512 → 26万参数
└── 数据依赖
    └── 位置编码受训练数据影响
```

**实际应用中的选择：**

```
BERT（可学习）：
├── 任务：理解任务（分类、NER等）
├── 序列长度固定：512
├── 不需要外推
└── 可学习编码足够

GPT（可学习）：
├── 任务：生成任务
├── 训练时固定长度
├── 推理时可能需要更长
└── 使用可学习编码，但有长度限制

LLaMA（RoPE）：
├── 任务：大语言模型
├── 需要处理长文本
├── RoPE有更好的外推能力
└── 结合了正弦编码的优点

实际建议：
├── 预训练模型：优先考虑RoPE或ALiBi
├── 固定长度任务：可学习编码足够
├── 需要外推：避免可学习编码
└── 资源有限：正弦编码（零参数）
```

**RoPE（旋转位置编码）的优势：**

```
结合两者优点：
├── 可外推（像正弦编码）
├── 编码相对位置（像正弦编码）
├── 在特征空间中旋转（更好的性质）
└── 被LLaMA、GLM等现代模型采用

数学形式：
├── 将位置编码为旋转矩阵
├── 查询和键在计算内积时自动包含相对位置
└── 理论上更优雅
```

</details>

<details>
<summary>思考题3：多头注意力中，为什么通常设置 d_k = d_model / h？这样设置有什么好处？</summary>

**答案：**

**设置原因：**

```
约束条件：
1. 所有头的输出需要拼接
2. 拼接后的维度应该等于d_model
3. 每个头的维度需要能整除d_model

设计选择：
├── h个头，每个头维度d_k
├── 拼接后：h × d_k
├── 要求：h × d_k = d_model
└── 因此：d_k = d_model / h
```

**好处分析：**

**1. 参数量平衡**

```
单头注意力（d_k = d_model）：
W_Q, W_K, W_V: d_model × d_model × 3
总参数：3 × d_model²

多头注意力（h头，d_k = d_model/h）：
每个头：d_model × d_k × 3 = 3 × d_model² / h
h个头：3 × d_model²
输出投影：d_model²
总参数：4 × d_model²

关键：头数不改变总参数量！
├── 参数量主要取决于d_model
├── 不因为增加头数而增加参数
└── 可以灵活调整头数
```

**2. 计算量平衡**

```
单头注意力计算量：
QK^T: O(n² × d_model)
与V相乘: O(n² × d_model)
总计: O(n² × d_model)

多头注意力计算量（每个头）：
QK^T: O(n² × d_k)
与V相乘: O(n² × d_k)

h个头总计：
h × O(n² × d_k) = h × O(n² × d_model/h)
= O(n² × d_model)

结论：计算量相同！
```

**3. 表达能力**

```
每个头学习不同的"注意力模式"：
├── 头1：关注语法关系
├── 头2：关注语义关系
├── 头3：关注位置关系
└── ...

这种设计鼓励：
├── 头之间的分工
├── 捕获不同类型的依赖
└── 增强模型表达能力
```

**4. 头数和维度的权衡**

```
假设d_model = 512

h=1, d_k=512:
├── 单头，只能学习一种模式
├── 表达能力有限
└── 但每个头维度大，信息丰富

h=8, d_k=64:
├── 8头，可以学习8种模式
├── 表达能力强
└── 但每个头维度小，信息有限

h=64, d_k=8:
├── 64头，模式多
├── 但每个头维度太小
└── 可能无法捕获足够信息

实践中的选择：
├── 通常 h ∈ {4, 8, 12, 16}
├── d_k ∈ {64, 128}
└── 平衡模式多样性和单头信息量
```

**实证发现：**

```
研究显示：
├── 有些头学习到有意义的模式（如句法、语义）
├── 有些头可能冗余
├── 剪枝部分头对性能影响不大
└── 适当增加头数有收益，但收益递减

建议：
├── 中小模型：h=8
├── 大模型：h=12-16
├── 根据实际任务调优
└── 注意d_k要能被d_model整除
```

</details>

<details>
<summary>思考题4：自注意力和卷积（CNN）有什么本质区别？为什么Transformer在视觉任务中也能取得成功？</summary>

**答案：**

**本质区别：**

```
1. 感受野
CNN：
├── 局部感受野（如3×3）
├── 通过堆叠扩大感受野
├── 感受野线性增长
└── 需要多层才能看到全局

Self-Attention：
├── 全局感受野
├── 每个位置可以直接看到所有位置
├── 感受野：整个序列/图像
└── 单层就能建立长距离依赖

2. 归纳偏置
CNN：
├── 局部性：相邻像素相关
├── 平移等变性：卷积核共享
├── 层级性：底层→高层抽象
└── 强归纳偏置，适合图像

Self-Attention：
├── 全局性：任何位置相关
├── 置换等变性：位置无关
├── 无层级：单层看到全局
└── 弱归纳偏置，需要更多数据

3. 参数效率
CNN：
├── 参数共享
├── 核大小固定
└── 参数量与输入大小无关

Self-Attention：
├── 计算量与序列长度平方相关
├── 注意力矩阵n×n
└── 长序列/高分辨率图像时开销大
```

**Transformer在视觉成功的原因：**

```
1. 全局建模能力
├── 图像中的长距离依赖
│   └── 例：猫的头和尾巴的关系
├── CNN需要多层才能建立
├── Transformer单层就能捕获
└── 对复杂场景理解有帮助

2. 自适应感受野
CNN：
├── 固定的局部感受野
├── 对所有位置一视同仁
└── 无法区分重要和不重要区域

Transformer：
├── 注意力权重自适应学习
├── 可以动态调整关注区域
├── 前景物体可能获得更多关注
└── 类似人类的视觉注意力

3. 大规模数据弥补弱归纳偏置
├── CNN的强归纳偏置适合小数据
├── Transformer需要大量数据
├── ImageNet-21K、JFT-300M等大数据集
└── 数据足够时，Transformer超越CNN

4. ViT的设计
├── 图像分块（Patch）
├── 每个Patch作为token
├── 添加位置编码
├── 纯Transformer架构
└── 在大数据集上预训练后效果优异

5. 统一架构
├── 多模态任务
│   └── 文本和图像用同样的架构
├── 迁移学习
│   └── 预训练模型易于迁移
└── 算力友好
    └── 并行计算效率高
```

**CNN vs Transformer选择建议：**

| 场景 | 推荐 | 理由 |
|------|------|------|
| 小数据集 | CNN | 强归纳偏置适合小数据 |
| 大数据集 | Transformer | 数据足够时性能更好 |
| 高分辨率图像 | CNN/局部Transformer | 全局注意力计算量大 |
| 多模态任务 | Transformer | 统一架构便于融合 |
| 实时应用 | CNN | 计算效率高 |
| 研究前沿 | Transformer | 表达能力强 |

**混合架构：**

```
结合两者优点：
├── CNN提取底层特征
├── Transformer建模全局关系
└── 如：Swin Transformer、ConvNeXt

趋势：
├── 纯CNN仍在进步（ConvNeXt）
├── 纯Transformer成为主流（ViT, Swin）
├── 混合架构有优势
└── 根据任务和数据量选择
```

</details>

---

## 七、今日要点

1. **自注意力本质**：序列中每个元素直接与所有其他元素交互，O(1)路径长度

2. **多头注意力**：多个头学习不同的关注模式，参数量不变，表达能力增强

3. **位置编码必要性**：自注意力是置换不变的，需要显式编码位置信息

4. **注意力变体**：
   - 交叉注意力：Q和K,V来自不同输入
   - 因果注意力：防止信息泄露，用于自回归生成
   - 高效注意力：解决O(n²)复杂度问题

5. **Flash Attention**：内存优化的注意力实现，已成为PyTorch默认

---

## 十、明日预告

**第13天：Transformer架构**

我们将探讨：
- Transformer的完整架构
- 编码器-解码器结构
- 层归一化和残差连接
- 前馈网络
- 训练技巧和优化策略
- BERT和GPT的区别

Transformer是现代NLP的基石，理解它对于掌握GPT、BERT等模型至关重要。
