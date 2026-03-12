# Day 11: 循环神经网络 - 处理序列的艺术

## 核心问题

**为什么全连接网络难以处理序列数据？循环神经网络的"循环"本质是什么？为什么LSTM能够解决长期依赖问题？**

---

## 一、序列数据的挑战

### 1.1 序列数据的特点

**什么是序列数据？**

```
序列数据：数据点之间存在时序或顺序关系

文本序列：
"The cat sat on the mat"
↓  ↓   ↓   ↓  ↓   ↓
单词之间存在语法和语义依赖

时间序列：
股票价格: [100, 102, 98, 105, 103, ...]
         t1   t2   t3   t4   t5
每个时间点依赖于历史数据

语音信号：
音频波形 → 声学特征序列 → 文字
连续信号，需要上下文理解

视频序列：
帧序列 [frame1, frame2, frame3, ...]
帧之间存在时间连续性
```

**序列数据的核心特征：**

| 特征 | 描述 | 例子 |
|------|------|------|
| 变长 | 序列长度不固定 | 句子长短不一 |
| 时序依赖 | 当前依赖于历史 | "下雨"依赖"天阴" |
| 上下文敏感 | 相同元素不同含义 | "bank"：银行 vs 河岸 |
| 长期依赖 | 跨越长距离的关联 | "他...（100词后）...它" |

### 1.2 全连接网络的困境

**问题1：变长输入**

```
全连接网络要求固定输入维度

句子1: "我"（1个词）
句子2: "我爱学习"（3个词）
句子3: "我爱学习深度学习"（5个词）

如何处理不同长度？
├── 填充（Padding）：浪费计算，引入噪声
├── 截断：丢失信息
└── 无法优雅处理
```

**问题2：参数爆炸**

```
假设词表大小 V = 10000，最大序列长度 L = 100

全连接网络输入维度：10000 × 100 = 1,000,000
如果隐藏层1000个神经元
参数量：1,000,000 × 1000 = 10亿参数！

问题：
├── 计算量巨大
├── 过拟合风险高
└── 训练困难
```

**问题3：无法捕捉顺序**

```
全连接网络：将输入展平后处理

"我喜欢你" → [我, 喜欢, 你] → 全连接 → 输出
"你喜欢我" → [你, 喜欢, 我] → 全连接 → 输出

如果权重矩阵固定，打乱顺序后输出可能相同！
→ 无法区分顺序，丢失时序信息
```

### 1.3 RNN的核心思想

**核心洞察：顺序处理，保持记忆**

```
RNN的关键创新：
├── 参数共享：每个时间步使用相同的权重
├── 隐藏状态：维护一个"记忆"来存储历史信息
└── 顺序处理：按时间步依次处理序列元素

类比：阅读理解
├── 你不会一次性读完所有内容
├── 而是逐词阅读，同时保持对之前内容的记忆
└── 新内容与旧记忆结合，更新理解
```

---

## 二、RNN的数学形式

### 2.1 基本RNN结构

**单时间步计算：**

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**展开形式：**

```
时间步展开：

t=1        t=2        t=3        t=4
x₁ ──→     x₂ ──→     x₃ ──→     x₄
↓          ↓          ↓          ↓
h₁ ──→     h₂ ──→     h₃ ──→     h₄
↓          ↓          ↓          ↓
y₁         y₂         y₃         y₄

箭头表示信息流动：
├── 垂直：输入到隐藏，隐藏到输出
└── 水平：隐藏状态的传递（记忆）
```

**详细计算过程：**

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """从零实现简单RNN"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 参数定义
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x, h_prev=None):
        """
        x: (seq_len, batch, input_size)
        h_prev: (batch, hidden_size) 初始隐藏状态
        """
        seq_len, batch_size, input_size = x.shape
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        for t in range(seq_len):
            # 计算当前隐藏状态
            h_t = torch.tanh(
                x[t] @ self.W_xh +    # 输入到隐藏
                h_prev @ self.W_hh +   # 隐藏到隐藏（记忆）
                self.b_h
            )
            
            # 计算输出
            y_t = h_t @ self.W_hy + self.b_y
            
            outputs.append(y_t)
            h_prev = h_t  # 更新隐藏状态
        
        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs)
        return outputs, h_t
```

### 2.2 RNN的计算图

**前向传播：**

```
输入序列 x = [x₁, x₂, x₃]

h₀ = 0（初始隐藏状态）

t=1:
h₁ = tanh(Wₓₕx₁ + Wₕₕh₀ + bₕ)
y₁ = Wₕᵧh₁ + bᵧ

t=2:
h₂ = tanh(Wₓₕx₂ + Wₕₕh₁ + bₕ)
y₂ = Wₕᵧh₂ + bᵧ

t=3:
h₃ = tanh(Wₓₕx₃ + Wₕₕh₂ + bₕ)
y₃ = Wₕᵧh₃ + bᵧ

信息流：
x₁ → h₁ → h₂ → h₃（记忆传递）
     ↓    ↓    ↓
     y₁   y₂   y₃（输出）
```

**反向传播（BPTT：时间反向传播）：**

```
损失函数：L = L₁ + L₂ + L₃

∂L/∂Wₕₕ = ∂L/∂y₃ · ∂y₃/∂h₃ · ∂h₃/∂Wₕₕ
        + ∂L/∂y₂ · ∂y₂/∂h₂ · ∂h₂/∂Wₕₕ
        + ∂L/∂y₁ · ∂y₁/∂h₁ · ∂h₁/∂Wₕₕ

关键：每个时间步都对参数有贡献！
```

### 2.3 PyTorch中的RNN

```python
import torch.nn as nn

# PyTorch内置RNN
rnn = nn.RNN(
    input_size=100,    # 输入特征维度
    hidden_size=256,   # 隐藏状态维度
    num_layers=2,      # RNN层数
    batch_first=True,  # 输入形状: (batch, seq, feature)
    bidirectional=False,  # 是否双向
    dropout=0.5        # 层间dropout
)

# 输入
x = torch.randn(32, 10, 100)  # batch=32, seq_len=10, input_size=100
h0 = torch.zeros(2, 32, 256)  # num_layers=2, batch=32, hidden_size=256

# 前向传播
output, hn = rnn(x, h0)

print(f"输出形状: {output.shape}")  # (32, 10, 256)
print(f"最终隐藏状态: {hn.shape}")   # (2, 32, 256)

# output: 所有时间步的隐藏状态
# hn: 最后一个时间步的隐藏状态
```

---

## 三、RNN的问题与挑战

### 3.1 梯度消失问题

**数学分析：**

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$$

对于时间步 $t$ 和 $t-k$：

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=t-k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t-k+1}^{t} W_{hh}^T \cdot \text{diag}(\tanh'(z_i))$$

**问题：**
- $\tanh'$ 的值在 $[0, 1]$ 之间
- $W_{hh}$ 的特征值如果小于1，连乘后会指数衰减
- $k$ 越大（距离越远），梯度越接近0

**实际影响：**

```python
import matplotlib.pyplot as plt

def gradient_magnitude(W, seq_len):
    """模拟梯度消失"""
    grad = 1.0
    magnitudes = [grad]
    for t in range(seq_len):
        # 假设 tanh' ≈ 0.5
        grad = grad * W * 0.5
        magnitudes.append(abs(grad))
    return magnitudes

# 不同的权重值
W_values = [0.9, 1.0, 1.1]

plt.figure(figsize=(10, 6))
for W in W_values:
    mags = gradient_magnitude(W, 20)
    plt.plot(mags, label=f'W = {W}')
plt.xlabel('Time Step')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow Through Time')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# W=0.9: 梯度指数衰减（消失）
# W=1.1: 梯度指数增长（爆炸）
# W=1.0: 梯度稳定（理想但不现实）
```

### 3.2 长期依赖问题

**问题示例：**

```
句子："我出生在法国...（中间很多词）...所以我说流利的法语"

任务：预测最后一个词

需要的信息：
├── "出生在法国" 在句子开头
├── 中间隔了很多词
├── RNN需要保持"法国"这个信息
└── 但梯度消失导致无法学习这个长期依赖

实际表现：
├── 短期依赖：学得很好
└── 长期依赖：几乎学不到
```

**长期依赖的本质：**

```
信息传递链：h₁ → h₂ → h₃ → ... → hₜ

每一步都是非线性变换：
h_{t+1} = tanh(W·h_t + ...)

问题：
├── tanh 压缩信息到 [-1, 1]
├── 每一步都有信息损失
├── 经过多个时间步，早期信息被"稀释"
└── 最终 h_t 几乎不包含早期信息
```

---

## 四、LSTM：长期记忆的解决方案

### 4.1 LSTM的核心思想

**关键创新：门控机制 + 细胞状态**

```
LSTM引入两个状态：
├── 细胞状态 C_t：长期记忆，信息高速公路
└── 隐藏状态 h_t：短期记忆，对外输出

三个门控：
├── 遗忘门（Forget Gate）：决定丢弃哪些长期记忆
├── 输入门（Input Gate）：决定更新哪些长期记忆
└── 输出门（Output Gate）：决定输出哪些信息
```

### 4.2 LSTM详细结构

**完整计算过程：**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{（遗忘门）}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{（输入门）}$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{（候选记忆）}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{（更新细胞状态）}$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{（输出门）}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{（隐藏状态）}$$

**图解LSTM单元：**

```
                    ┌─────────────────────────────────────┐
                    │              LSTM单元                │
                    │                                     │
    C_{t-1} ────────┼────→ [×] ──────→ [+] ──────→ C_t ──┼───→
                    │       ↑             ↑               │
                    │       f_t        i_t, C̃_t          │
                    │     遗忘门        输入门            │
    h_{t-1} ────────┼──→ [连接] ←─────────┘               │
                    │     ↓  ↓                            │
                    │    f_t i_t o_t                      │
                    │         ↓                           │
                    │       [×] ──→ h_t ──────────────────┼───→
                    │       o_t                           │
                    └─────────────────────────────────────┘
                                  ↑
    x_t ──────────────────────────┘

符号说明：
[×] : 逐元素乘法（门控）
[+] : 逐元素加法（信息更新）
[连接] : 向量拼接
σ : sigmoid函数，输出[0,1]
tanh : 双曲正切，输出[-1,1]
```

### 4.3 为什么LSTM能解决长期依赖？

**关键1：细胞状态的线性传递**

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

关键洞察：
├── 如果 f_t ≈ 1, i_t ≈ 0，则 C_t ≈ C_{t-1}
├── 信息直接传递，不经过非线性变换
├── 梯度：∂C_t/∂C_{t-1} = f_t
└── 如果 f_t ≈ 1，梯度不衰减！

对比普通RNN：
h_t = tanh(W·h_{t-1} + ...)
├── 每一步都有 tanh 非线性
├── 梯度：∂h_t/∂h_{t-1} = tanh'(...) · W
└── tanh' < 1，梯度必然衰减
```

**关键2：门控机制**

```
遗忘门 f_t：
├── 学习什么时候"忘记"无关信息
├── sigmoid 输出 [0, 1]
└── 0 = 完全遗忘，1 = 完全保留

输入门 i_t：
├── 学习什么时候"记住"新信息
├── 控制哪些新信息进入细胞状态
└── 选择性记忆

输出门 o_t：
├── 学习什么时候"输出"信息
├── 控制细胞状态对输出的影响
└── 选择性输出
```

**LSTM代码实现：**

```python
class LSTMCell(nn.Module):
    """LSTM单元实现"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 所有门的权重合并计算（效率更高）
        self.W = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
    
    def forward(self, x, state):
        """
        x: (batch, input_size)
        state: (h, c) 各为 (batch, hidden_size)
        """
        h_prev, c_prev = state
        
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_size + hidden_size)
        
        # 计算所有门（一次性计算，效率高）
        gates = combined @ self.W + self.b  # (batch, 4 * hidden_size)
        
        # 分割得到各个门
        i, f, g, o = gates.chunk(4, dim=1)  # 各 (batch, hidden_size)
        
        # 激活函数
        i = torch.sigmoid(i)    # 输入门
        f = torch.sigmoid(f)    # 遗忘门
        g = torch.tanh(g)       # 候选记忆
        o = torch.sigmoid(o)    # 输出门
        
        # 更新细胞状态
        c = f * c_prev + i * g  # 长期记忆
        
        # 计算隐藏状态
        h = o * torch.tanh(c)   # 短期记忆
        
        return h, c
```

### 4.4 GRU：LSTM的简化版本

**GRU（Gated Recurrent Unit）：**

```
GRU简化了LSTM：
├── 两个门：重置门、更新门
├── 一个状态：隐藏状态 h_t
└── 参数更少，计算更快

计算过程：
r_t = σ(W_r · [h_{t-1}, x_t])     # 重置门
z_t = σ(W_z · [h_{t-1}, x_t])     # 更新门
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])  # 候选隐藏状态
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # 更新隐藏状态
```

**图解GRU：**

```
    h_{t-1} ────→ [×] ───→ [+] ───→ h_t ───→
                  ↑ r_t    ↑
                  │        z_t, h̃_t
                  │        ↑
                  └──[连接]┘
                       ↑
    x_t ───────────────┘
```

**LSTM vs GRU：**

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 状态数量 | 2个 (h, c) | 1个 (h) |
| 参数量 | 较多 | 较少 |
| 计算速度 | 较慢 | 较快 |
| 表达能力 | 较强 | 稍弱 |
| 适用场景 | 复杂长序列 | 中等长度序列 |

**GRU代码实现：**

```python
class GRUCell(nn.Module):
    """GRU单元实现"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 重置门和更新门
        self.W_rz = nn.Parameter(torch.randn(input_size + hidden_size, 2 * hidden_size))
        self.b_rz = nn.Parameter(torch.zeros(2 * hidden_size))
        
        # 候选隐藏状态
        self.W_h = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        
        # 重置门和更新门
        rz = torch.sigmoid(combined @ self.W_rz + self.b_rz)
        r, z = rz.chunk(2, dim=1)  # 重置门、更新门
        
        # 候选隐藏状态
        h_tilde = torch.tanh(
            torch.cat([r * h_prev, x], dim=1) @ self.W_h + self.b_h
        )
        
        # 更新隐藏状态
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

---

## 五、RNN的应用模式

### 5.1 多种输入输出模式

**1. 一对一（One-to-One）**

```
输入：单个向量
输出：单个向量

应用：图像分类（不是RNN）

x ──→ [RNN] ──→ y
```

**2. 一对多（One-to-Many）**

```
输入：单个向量
输出：序列

应用：图像描述生成

图像 ──→ [RNN] ──→ "一只" ──→ "猫" ──→ "坐在" ──→ "垫子上"
        h₀       h₁        h₂      h₃
```

**3. 多对一（Many-to-One）**

```
输入：序列
输出：单个向量

应用：情感分析、文本分类

"这" ──→ "部" ──→ "电影" ──→ "很" ──→ "好" ──→ [RNN] ──→ "正面"
                                      hₙ ──→ y

关键：只使用最后一个隐藏状态
```

**4. 多对多（Many-to-Many）**

```
输入：序列
输出：序列

应用：机器翻译、命名实体识别

方式1：同步（输入输出对齐）
x₁ ──→ x₂ ──→ x₃ ──→ x₄
↓      ↓      ↓      ↓
h₁ ──→ h₂ ──→ h₃ ──→ h₄
↓      ↓      ↓      ↓
y₁     y₂     y₃     y₄

方式2：异步（编码器-解码器）
编码器：x₁ → x₂ → x₃ → h_最终
解码器：h_最终 → y₁ → y₂ → y₃
```

### 5.2 双向RNN

**动机：未来信息也很重要**

```
句子："我 喜欢 苹果 公司 的 产品"

预测"苹果"的含义：
├── 只看左边："我 喜欢 ___" → 不确定
├── 只看右边："___ 公司 的 产品" → 确定是苹果公司
└── 双向：结合左右上下文 → 更准确

单向RNN：只看到历史
双向RNN：同时看到过去和未来
```

**双向RNN结构：**

```
前向：h⃗₁ → h⃗₂ → h⃗₃ → h⃗₄
      x₁    x₂    x₃    x₄

后向：h⃖₁ ← h⃖₂ ← h⃖₃ ← h⃖₄
      x₁    x₂    x₃    x₄

合并：h₁ = [h⃗₁; h⃖₁]
      h₂ = [h⃗₂; h⃖₂]
      h₃ = [h⃗₃; h⃖₃]
      h₄ = [h⃗₄; h⃖₄]
```

**代码实现：**

```python
# PyTorch双向RNN
birnn = nn.RNN(
    input_size=100,
    hidden_size=256,
    bidirectional=True,  # 双向
    batch_first=True
)

x = torch.randn(32, 10, 100)
output, hn = birnn(x)

print(f"输出形状: {output.shape}")  # (32, 10, 512) = (batch, seq, 2*hidden)
print(f"隐藏状态: {hn.shape}")      # (2, 32, 256) = (num_directions, batch, hidden)

# output 包含前向和后向的拼接
# hn[0] 是前向最终状态，hn[1] 是后向最终状态
```

### 5.3 深层RNN

**多层RNN：**

```
第1层：x₁  →  x₂  →  x₃  →  x₄
       ↓      ↓      ↓      ↓
       h₁⁽¹⁾ → h₂⁽¹⁾ → h₃⁽¹⁾ → h₄⁽¹⁾
       ↓      ↓      ↓      ↓
第2层：h₁⁽²⁾ → h₂⁽²⁾ → h₃⁽²⁾ → h₄⁽²⁾
       ↓      ↓      ↓      ↓
第3层：h₁⁽³⁾ → h₂⁽³⁾ → h₃⁽³⁾ → h₄⁽³⁾
       ↓      ↓      ↓      ↓
       y₁     y₂     y₃     y₄

每一层提取不同级别的抽象：
├── 第1层：低级特征（字符级）
├── 第2层：中级特征（词级）
└── 第3层：高级特征（句法级）
```

```python
# 多层LSTM
multi_lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=3,  # 3层
    batch_first=True,
    dropout=0.5    # 层间dropout
)

x = torch.randn(32, 10, 100)
output, (hn, cn) = multi_lstm(x)

print(f"输出形状: {output.shape}")  # (32, 10, 256)
print(f"隐藏状态: {hn.shape}")      # (3, 32, 256) = (num_layers, batch, hidden)
print(f"细胞状态: {cn.shape}")      # (3, 32, 256)
```



---

## 六、序列到序列模型

### 6.1 Seq2Seq架构

**编码器-解码器结构：**

```
编码器（Encoder）：
"我" → "爱" → "你" → [EOS]
 ↓      ↓      ↓
 h₁ →  h₂ →  h₃ → h_最终
                     ↓
                   上下文向量 c

解码器（Decoder）：
  c  →  "I"  →  "love" → "you" → [EOS]
  ↓       ↓        ↓       ↓
 h₀' →  h₁'  →   h₂'  →  h₃'
  ↓       ↓        ↓       ↓
 "I"   "love"    "you"   [EOS]

关键：c 是编码器和解码器之间的桥梁
```

**代码实现：**

```python
class Encoder(nn.Module):
    """编码器"""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        output, (h, c) = self.rnn(embedded)
        return h, c


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h, c):
        embedded = self.embedding(x)
        output, (h, c) = self.rnn(embedded, (h, c))
        output = self.fc(output)
        return output, h, c
```

### 6.2 教师强制（Teacher Forcing）

**概念：**

```
训练时的两种策略：

1. 自由运行（Free Running）：
   用模型自己的预测作为下一步输入
   └── 问题：错误会累积

2. 教师强制（Teacher Forcing）：
   用真实目标作为下一步输入
   └── 优点：训练稳定
   └── 缺点：推理时可能不一致

实际做法：
├── 训练初期：高教师强制比例
├── 训练后期：逐渐降低比例
└── 推理时：完全自由运行
```

### 6.3 Seq2Seq的问题

**固定长度上下文向量：**

```
问题：编码器将整个输入序列压缩为一个固定长度的向量 c

长句子问题：
输入："我 昨天 去 了 北京 故宫 并且 看到 很多 游客 ..."

编码器需要把所有信息压缩到 c 中
    ↓
信息瓶颈：c 容量有限，无法承载长序列的所有信息
    ↓
解码器无法准确获取需要的信息

解决思路：
├── 让解码器"看到"编码器的所有隐藏状态
└── 根据当前需要，选择性关注相关信息
    ↓
注意力机制（Attention）
```

---

## 七、注意力机制基础

### 7.1 注意力的直觉

**人类阅读的注意力：**

```
阅读句子时，你的注意力在不同词上分配：

"我 喜欢 吃 苹果"
        ↑
    当前关注点

翻译"苹果"时：
├── 中文："苹果"
├── 你会回顾源句，找到"苹果"对应的词
└── 注意力集中在"苹果"这个词上
```

**注意力机制的核心：**

```
不再使用固定的上下文向量 c

而是，解码器每一步都：
1. 查看 编码器的所有隐藏状态
2. 计算 当前解码状态与各编码状态的"相关性"
3. 加权求和 得到当前步的上下文向量

c_i = Σ α_ij * h_j

其中：
├── h_j: 编码器第j步的隐藏状态
├── α_ij: 注意力权重（相关性）
└── c_i: 解码器第i步的上下文向量
```

### 7.2 注意力的计算

**步骤：**

```
解码器第 i 步：

1. 计算注意力分数：
   e_ij = score(h_i^dec, h_j^enc)
   
   常用 score 函数：
   ├── 点积：score(h_i, h_j) = h_i^T · h_j
   ├── 加性：score(h_i, h_j) = v^T tanh(W[h_i; h_j])
   └── 缩放点积：score(h_i, h_j) = (h_i^T · h_j) / sqrt(d)

2. 计算注意力权重（softmax归一化）：
   α_ij = softmax(e_ij) = exp(e_ij) / Σ_k exp(e_ik)
   
   性质：Σ_j α_ij = 1

3. 计算上下文向量：
   c_i = Σ_j α_ij * h_j

4. 解码：
   y_i = decode(h_i^dec, c_i)
```

**图解：**

```
编码器隐藏状态：
h₁    h₂    h₃    h₄
↓     ↓     ↓     ↓
α₁₁   α₁₂   α₁₃   α₁₄   （注意力权重，解码第1步）
0.1   0.2   0.5   0.2
   
c₁ = 0.1*h₁ + 0.2*h₂ + 0.5*h₃ + 0.2*h₄

解码器第1步：结合 h₁^dec 和 c₁ 生成 y₁
```

**代码实现：**

```python
class Attention(nn.Module):
    """加性注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        
        # 扩展decoder_hidden以匹配encoder_outputs
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # 计算注意力分数
        energy = torch.tanh(self.W(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # 归一化
        attention_weights = torch.softmax(attention, dim=1)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights
```
---

## 八、思考题

<details>
<summary>思考题1：为什么RNN比全连接网络更适合处理序列数据？RNN的参数共享机制有什么优缺点？</summary>

**答案：**

**RNN适合序列数据的原因：**

1. **处理变长序列**
   - 全连接网络需要固定输入维度，需要填充或截断
   - RNN顺序处理，可处理任意长度，参数不随序列长度变化

2. **捕捉时序依赖**
   - 全连接网络展平输入后无法区分顺序
   - RNN通过隐藏状态传递信息，自然建模时序依赖

3. **参数效率**
   - 示例：词表10000，序列长度100，隐藏层256
   - 全连接网络参数：约2.56亿
   - RNN参数：约518万
   - 参数减少约50倍

**参数共享的优缺点：**

优点：参数效率高、可处理任意长度、泛化能力强
缺点：假设序列位置等价、无法建模位置特异性、递归计算难以并行

</details>

<details>
<summary>思考题2：LSTM是如何解决梯度消失问题的？</summary>

**答案：**

**普通RNN梯度消失原因：**
- 梯度传播涉及连乘：∂h_t/∂h_{t-1} = tanh' · W
- tanh' < 1，连乘后指数衰减

**LSTM解决方案：**
- 细胞状态线性传递：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
- 梯度：∂C_t/∂C_{t-1} = f_t
- 如果遗忘门 f_t ≈ 1，梯度无损传递
- 网络学习何时保持信息，何时遗忘

**关键创新：**
- 细胞状态是"信息高速公路"
- 门控机制选择性地更新和输出
- 遗忘门可以学习保持长期记忆

</details>

<details>
<summary>思考题3：什么是教师强制？它有什么问题？</summary>

**答案：**

**教师强制：**
- 训练时用真实目标作为下一步输入
- 推理时用模型预测作为下一步输入

**问题：暴露偏差**
- 模型从未见过自己的错误输出
- 推理时一旦出错，不知道如何恢复
- 训练和推理分布不一致

**解决方案：**
- 计划采样：训练中逐步降低教师强制比例
- 训练初期高比例，后期降低
- 让模型逐渐适应自己的预测

</details>

<details>
<summary>思考题4：注意力机制解决了Seq2Seq的什么问题？</summary>

**答案：**

**Seq2Seq的问题：**
- 所有信息压缩到固定长度向量
- 长序列信息丢失
- 解码器无法"回顾"源序列

**注意力机制解决方案：**
- 解码器每步查看编码器所有状态
- 动态生成上下文向量
- 选择性关注源序列不同部分

**与RNN的区别：**
- RNN：顺序传递，路径长度=T，梯度衰减
- 注意力：直接连接，路径长度=1，无衰减
- 注意力可并行计算，但复杂度O(T²)

</details>

---

## 九、今日要点

1. **RNN的本质**：通过隐藏状态传递历史信息，参数共享处理变长序列

2. **LSTM的关键**：
   - 细胞状态线性传递，解决梯度消失
   - 门控机制选择性记忆和遗忘
   - 遗忘门、输入门、输出门协同工作

3. **GRU的简化**：
   - 两个门（重置门、更新门）
   - 参数更少，计算更快
   - 性能与LSTM相当

4. **Seq2Seq模型**：
   - 编码器-解码器架构
   - 固定长度瓶颈问题
   - 教师强制训练策略

5. **注意力机制**：
   - 动态上下文向量
   - 解决长序列信息瓶颈
   - 可解释性强

---

## 十、明日预告

**第12天：注意力机制深入**

我们将探讨：
- 自注意力（Self-Attention）机制
- 多头注意力（Multi-Head Attention）
- 位置编码的设计
- Transformer的核心组件
- 注意力的变体和优化

注意力机制是现代深度学习的核心，为Transformer架构奠定基础。
