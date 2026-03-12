# Day 13: Transformer架构 - 现代深度学习的基石

## 核心问题

**Transformer为什么能够彻底改变NLP？编码器和解码器各自的作用是什么？残差连接和层归一化为什么如此重要？**

---

## 一、Transformer的整体架构

### 1.1 从RNN到Transformer的演进

**RNN+Attention的局限：**

```
架构：
├── 编码器：RNN生成隐藏状态序列
├── 解码器：RNN生成目标序列
└── 注意力：解码器关注编码器状态

局限：
├── 顺序计算：无法并行
│   └── RNN必须等前一步计算完成
├── 长距离依赖：仍然受限
│   └── RNN隐藏状态仍有信息损失
└── 训练效率：低
    └── GPU利用率不高
```

**Transformer的革命性创新：**

```
抛弃RNN，完全基于注意力：
├── 自注意力：替代RNN的序列建模
├── 并行计算：所有位置同时处理
├── 全局感受野：单层即可建立长距离依赖
└── 高效训练：充分利用GPU并行能力

结果：
├── 训练速度大幅提升
├── 长序列建模能力增强
├── 成为NLP的标准架构
└── 扩展到CV、多模态等领域
```

### 1.2 Transformer架构图解

**完整架构：**

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer                           │
├──────────────────────┬──────────────────────────────────┤
│      编码器           │           解码器                 │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │ 输入嵌入+位置   │  │  │ 输出嵌入+位置              │  │
│  └───────┬────────┘  │  └───────────┬────────────────┘  │
│          ↓           │              ↓                   │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │  多头自注意力   │  │  │  掩码多头自注意力           │  │
│  └───────┬────────┘  │  └───────────┬────────────────┘  │
│          ↓           │              ↓                   │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │ 残差+层归一化   │  │  │   残差+层归一化             │  │
│  └───────┬────────┘  │  └───────────┬────────────────┘  │
│          ↓           │              ↓                   │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │   前馈网络     │  │  │   编码器-解码器注意力       │←─┤
│  └───────┬────────┘  │  └───────────┬────────────────┘  │
│          ↓           │              ↓                   │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │ 残差+层归一化   │  │  │   残差+层归一化             │  │
│  └───────┬────────┘  │  └───────────┬────────────────┘  │
│          ↓           │              ↓                   │
│         ...×N        │  ┌────────────────────────────┐  │
│                      │  │      前馈网络              │  │
│                      │  └───────────┬────────────────┘  │
│                      │              ↓                   │
│                      │  ┌────────────────────────────┐  │
│                      │  │   残差+层归一化             │  │
│                      │  └───────────┬────────────────┘  │
│                      │              ↓                   │
│                      │             ...×N                │
│                      │              ↓                   │
│                      │  ┌────────────────────────────┐  │
│                      │  │      线性层+Softmax         │  │
│                      │  └────────────────────────────┘  │
└──────────────────────┴──────────────────────────────────┘
```

**关键组件：**

```
编码器：
├── 输入嵌入 + 位置编码
├── N层编码器层（原论文N=6）
│   ├── 多头自注意力
│   ├── 残差连接 + 层归一化
│   ├── 前馈网络（FFN）
│   └── 残差连接 + 层归一化
└── 输出传递给解码器

解码器：
├── 输出嵌入 + 位置编码
├── N层解码器层（原论文N=6）
│   ├── 掩码多头自注意力
│   ├── 残差连接 + 层归一化
│   ├── 编码器-解码器注意力（Cross-Attention）
│   ├── 残差连接 + 层归一化
│   ├── 前馈网络（FFN）
│   └── 残差连接 + 层归一化
├── 线性层
└── Softmax输出概率
```

### 1.3 数据流分析

**训练阶段：**

```
输入：
├── 源序列（编码器输入）："我爱学习"
└── 目标序列（解码器输入）："<SOS> I love learning"

编码器流程：
1. 词嵌入：["我", "爱", "学", "习"] → [e_我, e_爱, e_学, e_习]
2. 位置编码：添加位置信息
3. 自注意力：每个词关注所有词
4. 前馈网络：非线性变换
5. 输出编码器状态：H_enc = [h_1, h_2, h_3, h_4]

解码器流程：
1. 词嵌入：["<SOS>", "I", "love", "learning"] → [e_SOS, e_I, e_love, e_learning]
2. 位置编码：添加位置信息
3. 掩码自注意力：每个位置只能看到之前的位置
4. Cross-Attention：解码器状态查询编码器状态
5. 前馈网络：非线性变换
6. 线性层+Softmax：输出词表概率分布

损失计算：
├── 预测：["I", "love", "learning", "<EOS>"]
├── 目标：["I", "love", "learning", "<EOS>"]
└── 交叉熵损失
```

**推理阶段：**

```
自回归生成：
1. 输入源序列到编码器 → 得到编码器状态
2. 解码器输入：<SOS>
3. 解码器输出：概率分布 → 选择概率最高的词"I"
4. 解码器输入：<SOS> I
5. 解码器输出：概率分布 → 选择"love"
6. 解码器输入：<SOS> I love
7. 解码器输出：概率分布 → 选择"learning"
8. 解码器输入：<SOS> I love learning
9. 解码器输出：概率分布 → 选择<EOS>
10. 结束生成

关键：每一步生成一个词，新词作为下一步输入
```


---

## 二、编码器详解

### 2.1 输入嵌入层

**词嵌入：**

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        x: (batch, seq_len) - 词元索引
        返回: (batch, seq_len, d_model)
        """
        # 乘以sqrt(d_model)用于缩放
        # 原因：使嵌入的L2范数与位置编码相近
        return self.embedding(x) * math.sqrt(self.d_model)
```

**为什么需要缩放？**

```
问题：
├── 词嵌入的初始化通常均值为0，标准差较小
├── 位置编码的值域是[-1, 1]
├── 如果不缩放，词嵌入可能被位置编码"淹没"
└── 模型可能忽略词的内容信息

解决方案：
├── 乘以sqrt(d_model)
├── 使词嵌入和位置编码在同一量级
└── 模型可以平衡两者的信息

示例：d_model = 512
├── 缩放因子：sqrt(512) ≈ 22.6
├── 词嵌入范数：约22.6
├── 位置编码范数：约1
└── 两者可比
```

### 2.2 多头自注意力层

```python
class EncoderLayer(nn.Module):
    """编码器单层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差 + 层归一化
        attn_output, _ = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class Encoder(nn.Module):
    """完整编码器"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        
        # 词嵌入和位置编码
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        
        # N个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) - 词元索引
        """
        # 嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        
        # 通过N层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最终归一化
        x = self.norm(x)
        
        return x
```

### 2.3 前馈网络（Feed-Forward Network）

**结构：**

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

两层全连接：
├── 第一层：d_model → d_ff（扩展）
├── 激活函数：ReLU或GELU
└── 第二层：d_ff → d_model（压缩）

原论文参数：
├── d_model = 512
├── d_ff = 2048（4倍扩展）
└── 扩展比例：通常为4
```

**作用：**

```
每个位置独立处理：
├── 多头注意力捕获位置间关系
├── FFN处理每个位置的特征
└── 增加模型的非线性表达能力

类比：
├── 注意力：信息交流（不同位置交互）
├── FFN：信息处理（每个位置内部）
└── 类似：两个人对话后各自思考
```

**实现：**

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或 nn.GELU()
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = self.linear1(x)           # (batch, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)           # (batch, seq_len, d_model)
        return x
```

**为什么扩展4倍？**

```
直觉：
├── 扩展维度增加表达能力
├── 类似核方法映射到高维空间
└── 压缩回来保持维度一致

实验发现：
├── d_ff = 4 × d_model 效果好
├── 更大的d_ff收益递减
└── 更小的d_ff表达能力不足

参数量：
├── FFN参数：2 × d_model × d_ff = 2 × 512 × 2048 = 2M
├── 注意力参数：4 × d_model² = 4 × 512 × 512 = 1M
└── FFN参数是注意力的2倍
```

### 2.4 残差连接和层归一化

**残差连接：**

```
结构：LayerNorm(x + Sublayer(x))

作用：
├── 梯度直通通道
│   └── ∂(x + F(x))/∂x = 1 + ∂F(x)/∂x
├── 缓解梯度消失
├── 允许堆叠深层网络
└── 类似ResNet
```

**层归一化（Layer Normalization）：**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        在最后一个维度（特征维度）上归一化
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和平移
        return self.gamma * x_norm + self.beta
```

**为什么用层归一化而不是批归一化？**

| 特性 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化维度 | 批次维度 | 特征维度 |
| 对序列长度 | 敏感 | 不敏感 |
| 对批次大小 | 依赖 | 不依赖 |
| 推理行为 | 需要统计量 | 无需统计量 |
| NLP任务 | 效果差 | 效果好 |

```
NLP任务的特殊性：
├── 序列长度可变
│   └── Batch Norm需要填充到相同长度
├── 批次大小可能变化
│   └── 小批次时Batch Norm不稳定
└── 推理时批次大小可能为1
    └── Batch Norm难以处理

Layer Norm优势：
├── 每个样本独立归一化
├── 不依赖批次统计量
├── 适合序列数据
└── 训练和推理行为一致
```

**Pre-Norm vs Post-Norm：**

```
Post-Norm（原论文）：
x → Sublayer → Add → Norm → Output

Pre-Norm（现代做法）：
x → Norm → Sublayer → Add → Output

Pre-Norm优势：
├── 训练更稳定
├── 允许更深的网络
├── 梯度传播更顺畅
└── 被GPT-2/3、LLaMA等采用
```

```python
class EncoderLayerPreNorm(nn.Module):
    """Pre-Norm版本的编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm: Norm → Attention → Add
        attn_output, _ = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Pre-Norm: Norm → FFN → Add
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x
```


---

## 三、解码器详解

### 3.1 解码器的特殊性

**与编码器的区别：**

```
编码器：
├── 自注意力：双向，可以看到所有位置
├── 输入：完整的源序列
└── 用途：理解源语言

解码器：
├── 自注意力：单向（掩码），只能看到之前的位置
├── Cross-Attention：连接编码器
├── 输入：目标序列（训练时完整，推理时逐步）
└── 用途：生成目标语言
```

### 3.2 掩码多头自注意力

**因果掩码的作用：**

```
训练时：
├── 目标序列完整输入解码器
├── 如果不掩码，位置i可以看到位置i+1, i+2, ...
├── 这就是"作弊"，模型会学不到东西
└── 必须用掩码遮住未来位置

掩码示例（序列长度=4）：
┌──────────────────────┐
│ 1  0  0  0 │ 位置1只能看位置1
│ 1  1  0  0 │ 位置2可以看位置1,2
│ 1  1  1  0 │ 位置3可以看位置1,2,3
│ 1  1  1  1 │ 位置4可以看位置1,2,3,4
└──────────────────────┘
1 = 可见，0 = 掩码
```

**实现：**

```python
class DecoderLayer(nn.Module):
    """解码器单层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 掩码自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        x: 解码器输入 (batch, tgt_len, d_model)
        enc_output: 编码器输出 (batch, src_len, d_model)
        tgt_mask: 目标序列掩码（因果掩码）
        memory_mask: 源序列掩码（可选，用于填充）
        """
        # 掩码自注意力
        attn_output, _ = self.self_attn(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力
        # Q来自解码器，K和V来自编码器
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
```

### 3.3 编码器-解码器注意力

**Cross-Attention的工作原理：**

```
查询（Q）：解码器的状态
├── "我当前在生成什么"
├── 形状：(batch, tgt_len, d_model)

键（K）和值（V）：编码器的输出
├── "源语言有什么信息"
├── 形状：(batch, src_len, d_model)

计算过程：
1. 解码器位置i生成查询 q_i
2. 与编码器所有位置 k_1, k_2, ..., k_n 计算相关性
3. 得到注意力权重 α_i1, α_i2, ..., α_in
4. 加权求和编码器的值：c_i = Σ α_ij · v_j
5. c_i 包含源语言的相关信息

意义：
├── 解码器"回顾"源序列
├── 选择性地关注源语言的不同部分
├── 实现软对齐
└── 类似之前的注意力机制，但更高效
```

**可视化Cross-Attention：**

```
翻译任务："我 爱 你" → "I love you"

解码器生成"love"时：
├── 查询：解码器"love"位置的状态
├── 编码器状态：[h_我, h_爱, h_你]
├── 注意力权重：
│   ├── 对"我"：0.1
│   ├── 对"爱"：0.8  ← 主要关注
│   └── 对"你"：0.1
└── 上下文向量：主要包含"爱"的信息

解码器结合：
├── 解码器自注意力状态（知道前文"I"）
├── 上下文向量（知道源语言的"爱"）
└── 预测下一个词："you"
```

### 3.4 完整解码器实现

```python
class Decoder(nn.Module):
    """完整解码器"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        
        # 词嵌入和位置编码
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        
        # N个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        x: 目标序列 (batch, tgt_len)
        enc_output: 编码器输出 (batch, src_len, d_model)
        """
        # 嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        
        # 通过N层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        
        # 最终归一化
        x = self.norm(x)
        
        # 输出词表概率
        output = self.fc_out(x)  # (batch, tgt_len, vocab_size)
        
        return output
    
    def generate_causal_mask(self, seq_len):
        """生成因果掩码"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
```

---

## 四、完整的Transformer实现

### 4.1 整合编码器和解码器

```python
class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 d_ff, num_layers, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, 
                               num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, 
                               num_layers, dropout)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: 源序列 (batch, src_len)
        tgt: 目标序列 (batch, tgt_len)
        """
        # 编码
        enc_output = self.encoder(src, src_mask)
        
        # 解码
        output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        
        return output
    
    def encode(self, src, src_mask=None):
        """仅编码（用于推理）"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        """仅解码（用于推理）"""
        return self.decoder(tgt, enc_output, tgt_mask, src_mask)


# 使用示例
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# 训练
src = torch.randint(0, 10000, (32, 20))  # batch=32, src_len=20
tgt = torch.randint(0, 10000, (32, 15))  # batch=32, tgt_len=15
output = model(src, tgt)
print(f"输出形状: {output.shape}")  # (32, 15, 10000)
```

### 4.2 训练流程

```python
class Trainer:
    def __init__(self, model, vocab, device='cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        
        # 损失函数（忽略填充符）
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001, 
                                    betas=(0.9, 0.98), eps=1e-9)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=0.001, 
            total_steps=10000, pct_start=0.1
        )
    
    def train_step(self, src, tgt):
        """
        src: 源序列 (batch, src_len)
        tgt: 目标序列 (batch, tgt_len)，包含<SOS>
        """
        self.model.train()
        
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        
        # 解码器输入：去掉最后一个词
        tgt_input = tgt[:, :-1]
        # 目标：去掉第一个词<SOS>
        tgt_output = tgt[:, 1:]
        
        # 创建因果掩码
        tgt_mask = self.model.decoder.generate_causal_mask(tgt_input.size(1))
        tgt_mask = tgt_mask.to(self.device)
        
        # 前向传播
        output = self.model(src, tgt_input, tgt_mask=tgt_mask)
        
        # 计算损失
        loss = self.criterion(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
```

### 4.3 推理流程

```python
class Inference:
    def __init__(self, model, vocab, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
    
    def greedy_decode(self, src, max_len=50):
        """
        贪婪解码
        src: 源序列 (1, src_len)
        """
        with torch.no_grad():
            # 编码
            enc_output = self.model.encode(src)
            
            # 初始化解码器输入为<SOS>
            tgt = torch.tensor([[self.vocab.sos_idx]], device=self.device)
            
            for _ in range(max_len):
                # 创建因果掩码
                tgt_mask = self.model.decoder.generate_causal_mask(tgt.size(1))
                tgt_mask = tgt_mask.to(self.device)
                
                # 解码
                output = self.model.decode(tgt, enc_output, tgt_mask=tgt_mask)
                
                # 取最后一个位置的输出
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # 拼接
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 如果生成<EOS>，停止
                if next_token.item() == self.vocab.eos_idx:
                    break
            
            return tgt
    
    def beam_search(self, src, beam_size=5, max_len=50):
        """
        束搜索
        """
        with torch.no_grad():
            # 编码
            enc_output = self.model.encode(src)
            
            # 初始化：[序列, 对数概率]
            beams = [([self.vocab.sos_idx], 0.0)]
            
            for _ in range(max_len):
                new_beams = []
                
                for seq, score in beams:
                    if seq[-1] == self.vocab.eos_idx:
                        # 已结束，保留
                        new_beams.append((seq, score))
                        continue
                    
                    # 解码
                    tgt = torch.tensor([seq], device=self.device)
                    tgt_mask = self.model.decoder.generate_causal_mask(len(seq))
                    tgt_mask = tgt_mask.to(self.device)
                    
                    output = self.model.decode(tgt, enc_output, tgt_mask=tgt_mask)
                    
                    # 取最后一个位置的log概率
                    log_probs = torch.log_softmax(output[:, -1, :], dim=-1)
                    
                    # 取top-k
                    topk_probs, topk_indices = log_probs.topk(beam_size)
                    
                    for prob, idx in zip(topk_probs[0], topk_indices[0]):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        new_beams.append((new_seq, new_score))
                
                # 保留top-k
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # 检查是否所有序列都结束
                if all(seq[-1] == self.vocab.eos_idx for seq, _ in beams):
                    break
            
            # 返回最佳序列
            best_seq = beams[0][0]
            return torch.tensor([best_seq], device=self.device)
```


---

## 五、训练技巧和优化

### 5.1 学习率调度

**Transformer的特殊学习率调度：**

```
Warmup阶段：
├── 学习率从0线性增加到峰值
├── 持续warmup_steps步
└── 帮助训练初期稳定

衰减阶段：
├── 峰值后按步数的反比例衰减
├── 衰减公式：lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
└── 逐渐降低学习率

公式：
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

**实现：**

```python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        step = self.step_count
        return (self.d_model ** (-0.5) * 
                min(step ** (-0.5), step * (self.warmup_steps ** (-1.5))))


# 可视化学习率
def plot_lr_schedule(d_model=512, warmup_steps=4000, total_steps=20000):
    scheduler = TransformerLRScheduler(None, d_model, warmup_steps)
    lrs = []
    for _ in range(total_steps):
        scheduler.step_count += 1
        lrs.append(scheduler.get_lr())
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Transformer Learning Rate Schedule')
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label='Warmup End')
    plt.legend()
    plt.grid(True)
    plt.show()
```

**为什么需要Warmup？**

```
训练初期的问题：
├── 参数随机初始化
├── 梯度不稳定
├── 大学习率可能导致参数震荡
└── 模型可能发散

Warmup的作用：
├── 初期小步更新，让模型稳定
├── 逐渐增大学习率，加速收敛
├── 类似"热身"，避免一开始就剧烈运动
└── 对于深层网络尤其重要
```

### 5.2 标签平滑

**问题：One-hot标签过于自信**

```
标准交叉熵：
目标标签：[0, 0, 1, 0, 0]（完全确定）
模型预测：[0.1, 0.1, 0.7, 0.05, 0.05]

问题：
├── 模型被鼓励预测[0, 0, 1, 0, 0]
├── 但实际任务往往有歧义
├── 过度自信可能导致过拟合
└── 不确定性未被考虑
```

**标签平滑：**

```python
class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
    
    def forward(self, pred, target):
        """
        pred: (batch, vocab_size) - log概率
        target: (batch) - 目标索引
        """
        # 创建平滑标签
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 2))  # -2: padding和正确类
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0
        
        # 忽略padding位置
        mask = (target == self.padding_idx)
        smooth_target.masked_fill_(mask.unsqueeze(1), 0)
        
        # KL散度损失
        loss = self.criterion(pred, smooth_target)
        return loss


# 对比
# 标准标签：[0, 0, 1, 0, 0]
# 平滑标签：[ε/(V-1), ε/(V-1), 1-ε, ε/(V-1), ε/(V-1)]
#          [0.0025, 0.0025, 0.99, 0.0025, 0.0025]  (ε=0.1, V=100)
```

**标签平滑的好处：**

```
1. 防止过度自信
├── 模型不会预测概率接近1
├── 保留一定的不确定性
└── 更好地校准置信度

2. 正则化效果
├── 防止模型过度拟合训练数据
├── 提高泛化能力
└── 类似熵正则化

3. 处理标注噪声
├── 真实数据可能有标注错误
├── 标签平滑提供一定容忍
└── 模型不会对错误标签过于自信
```

### 5.3 Dropout策略

**Transformer中的Dropout位置：**

```
1. 嵌入层后
   Embedding → Dropout

2. 注意力权重后
   Attention Weights → Dropout

3. 每个子层输出后
   Sublayer → Dropout → Add → Norm

4. FFN激活后
   Linear → ReLU → Dropout → Linear
```

**实现：**

```python
class TransformerWithDropout(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 嵌入后dropout
        x = self.dropout(x)
        
        # 注意力权重dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 子层输出dropout
        sublayer_output = self.dropout(sublayer_output)
        x = self.norm(x + sublayer_output)
        
        # FFN激活后dropout
        x = self.dropout(F.relu(x))
        
        return x
```

**Dropout率的选择：**

```
原论文：dropout = 0.1
├── 在小数据集上可能需要更高
├── 在大数据集上可以降低
└── 过高会损害性能

经验值：
├── 小数据集（< 10万）：0.2 - 0.3
├── 中等数据集（10万 - 100万）：0.1 - 0.2
├── 大数据集（> 100万）：0.1或更低
└── 微调预训练模型：0.0 - 0.1
```

### 5.4 梯度裁剪

```python
# 训练循环中
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

**为什么需要梯度裁剪？**

```
问题：
├── Transformer训练初期梯度可能很大
├── 导致参数更新过大
├── 训练不稳定，可能发散
└── 尤其是深层网络

解决：
├── 限制梯度的范数
├── 如果超过阈值，缩放梯度
├── 保持梯度方向，控制大小
└── 训练更稳定

公式：
如果 ||g|| > max_norm:
    g = g × (max_norm / ||g||)
```

---

## 六、BERT与GPT：Transformer的两大变体

### 6.1 架构对比

```
Transformer（原版）：
├── 编码器 + 解码器
├── 编码器：双向自注意力
├── 解码器：单向自注意力 + Cross-Attention
└── 用途：序列到序列任务（翻译）

BERT：
├── 仅编码器
├── 双向自注意力
├── 理解任务：分类、NER、QA
└── 代表：BERT, RoBERTa, ALBERT

GPT：
├── 仅解码器
├── 单向自注意力（因果掩码）
├── 生成任务：文本生成
└── 代表：GPT-2/3/4, LLaMA
```

**架构图：**

```
BERT（编码器）：
输入：[CLS] 我 爱 学 习 [SEP] 语 言 模 型 [SEP]
        ↓   ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓   ↓
     [Embeddings + Position Encoding]
        ↓   ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓   ↓
     [Self-Attention（双向）]
        ↓   ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓   ↓
     [Feed-Forward Network]
        ↓   ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓   ↓
          ... 多层 ...
        ↓   ↓   ↓   ↓   ↓    ↓   ↓   ↓   ↓   ↓
     输出：每个位置都有表示
     [CLS]用于分类任务

GPT（解码器）：
输入：<SOS> 我 爱 学 习 语 言 模 型
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
     [Embeddings + Position Encoding]
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
     [Masked Self-Attention（单向）]
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
     [Feed-Forward Network]
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
          ... 多层 ...
        ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
     输出：预测下一个词
     我  爱  学  习  语  言  模  型  <EOS>
```

### 6.2 BERT的训练方式

**掩码语言模型（MLM）：**

```
输入：我 [MASK] 学习深度学习
目标：预测[MASK]位置的词

训练过程：
1. 随机选择15%的词进行替换
2. 其中：
   ├── 80%替换为[MASK]
   ├── 10%替换为随机词
   └── 10%保持不变
3. 只预测被替换的词
4. 损失只在被替换位置计算

例子：
原句：我喜欢学习深度学习
输入：我 [MASK] 学习深度 [MASK]
目标：[MASK] → 喜欢, [MASK] → 学习
```

**下一句预测（NSP）：**

```
输入：[CLS] 句子A [SEP] 句子B [SEP]
目标：预测B是否是A的下一句

训练过程：
1. 50%选择真实的下一句（正例）
2. 50%选择随机句子（负例）
3. 用[CLS]位置的表示进行分类

例子：
正例：[CLS] 我爱学习 [SEP] 深度学习很有趣 [SEP] → IsNext
负例：[CLS] 我爱学习 [SEP] 今天天气真好 [SEP] → NotNext
```

**BERT的代码框架：**

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, 
                               d_ff=4*d_model, num_layers)
        # MLM头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        # NSP头
        self.nsp_head = nn.Linear(d_model, 2)
    
    def forward(self, input_ids, segment_ids, masked_positions=None):
        """
        input_ids: (batch, seq_len)
        segment_ids: (batch, seq_len) - 区分句子A和B
        masked_positions: MLM位置的索引
        """
        # 编码器输出
        enc_output = self.encoder(input_ids)
        
        # MLM预测
        if masked_positions is not None:
            mlm_output = self.mlm_head(enc_output[masked_positions])
        
        # NSP预测（使用[CLS]位置的表示）
        cls_output = enc_output[:, 0, :]  # [CLS]位置
        nsp_output = self.nsp_head(cls_output)
        
        return mlm_output, nsp_output
```

### 6.3 GPT的训练方式

**因果语言模型（CLM）：**

```
输入：我 爱 学 习 深 度 学 习
目标：爱 学 习 深 度 学 习 <EOS>

训练过程：
1. 使用因果掩码，每个位置只能看之前的位置
2. 预测下一个词
3. 损失在所有位置计算

例子：
位置1：看 <SOS>，预测 "我"
位置2：看 <SOS> 我，预测 "爱"
位置3：看 <SOS> 我 爱，预测 "学"
...
```

**GPT的代码框架：**

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        # 注意：这是解码器，但没有Cross-Attention
        self.decoder = nn.ModuleList([
            DecoderLayerWithoutCrossAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        """
        # 嵌入
        x = self.embedding(input_ids)
        x = self.position_encoding(x)
        
        # 创建因果掩码
        seq_len = input_ids.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        # 通过解码器层
        for layer in self.decoder:
            x = layer(x, mask)
        
        # 预测下一个词
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_len=50):
        """生成文本"""
        for _ in range(max_len):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
```

### 6.4 BERT vs GPT 对比

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | 仅编码器 | 仅解码器 |
| 注意力 | 双向 | 单向（因果） |
| 预训练 | MLM + NSP | CLM |
| 适合任务 | 理解任务 | 生成任务 |
| 优点 | 双向理解 | 自回归生成 |
| 缺点 | 不适合生成 | 单向理解 |
| 代表模型 | BERT, RoBERTa | GPT-2/3/4, LLaMA |

**任务适配：**

```
BERT适合：
├── 文本分类
├── 命名实体识别
├── 问答系统（抽取式）
├── 语义相似度
└── 情感分析

GPT适合：
├── 文本生成
├── 对话系统
├── 代码生成
├── 创意写作
└── 问答系统（生成式）
```


---

## 七、思考题

<details>
<summary>思考题1：Transformer为什么使用层归一化而不是批归一化？Pre-Norm和Post-Norm有什么区别？</summary>

**答案：**

**层归一化 vs 批归一化：**

```
批归一化（BatchNorm）：
├── 归一化维度：批次维度
├── 计算方式：对批次中所有样本的同一特征计算均值和方差
├── 公式：BN(x) = γ × (x - μ_B) / σ_B + β
└── 依赖批次统计量

层归一化（LayerNorm）：
├── 归一化维度：特征维度
├── 计算方式：对单个样本的所有特征计算均值和方差
├── 公式：LN(x) = γ × (x - μ_L) / σ_L + β
└── 不依赖批次统计量
```

**Transformer选择LayerNorm的原因：**

```
1. 序列长度可变
   批归一化：
   ├── 需要填充到相同长度
   ├── 填充位置影响统计量
   └── 不同长度批次行为不一致
   
   层归一化：
   ├── 每个样本独立处理
   ├── 不受序列长度影响
   └── 对变长序列友好

2. 批次大小变化
   批归一化：
   ├── 小批次统计量不稳定
   ├── 训练和推理行为不同
   └── 需要维护运行时统计量
   
   层归一化：
   ├── 批次大小无关
   ├── 训练和推理完全一致
   └── 无需额外统计量

3. 序列数据特性
   批归一化：
   ├── 假设批次内样本独立同分布
   ├── 但序列数据通常不同分布
   └── 时间步之间的差异被忽略
   
   层归一化：
   ├── 对每个时间步独立归一化
   ├── 尊重时间步之间的差异
   └── 更适合序列建模
```

**Pre-Norm vs Post-Norm：**

```
Post-Norm（原论文）：
x → Sublayer → Add → Norm → Output

残差连接在归一化之前：
output = LayerNorm(x + Sublayer(x))

Pre-Norm（现代做法）：
x → Norm → Sublayer → Add → Output

归一化在子层之前：
output = x + Sublayer(LayerNorm(x))
```

**Pre-Norm的优势：**

```
1. 训练稳定性
   Post-Norm：
   ├── 子层输出直接加到残差
   ├── 如果子层输出过大，可能不稳定
   └── 深层网络容易梯度爆炸/消失
   
   Pre-Norm：
   ├── 子层输入先归一化，范围稳定
   ├── 残差连接保持梯度流
   └── 训练更稳定，尤其是深层网络

2. 梯度传播
   Post-Norm梯度：
   ∂L/∂x = ∂L/∂output × (∂Norm/∂x + ∂Sublayer/∂x)
   
   Pre-Norm梯度：
   ∂L/∂x = ∂L/∂output × (1 + ∂Sublayer/∂Norm × ∂Norm/∂x)
   
   Pre-Norm的"1"确保梯度直接流过
   类似ResNet的梯度通道

3. 深层网络
   Post-Norm：
   ├── 12层Transformer训练困难
   ├── 需要仔细调参
   └── 深层网络需要特殊初始化
   
   Pre-Norm：
   ├── 可以轻松堆叠更多层
   ├── GPT-2/3使用Pre-Norm
   ├── LLaMA使用Pre-Norm
   └── 百层Transformer也能训练
```

**实践建议：**

```
选择LayerNorm的原因：
├── 序列任务默认选择
├── 变长序列友好
├── 小批次训练稳定
└── 训练推理一致

选择Pre-Norm的原因：
├── 深层网络（>12层）必须
├── 训练更稳定
├── 无需warmup也能训练
└── 现代Transformer的标准配置

何时用Post-Norm：
├── 浅层网络（≤12层）
├── 原始Transformer论文设置
└── BERT使用Post-Norm
```

</details>

<details>
<summary>思考题2：Transformer中的残差连接有什么作用？为什么深层网络必须使用残差连接？</summary>

**答案：**

**残差连接的作用：**

```
基本形式：y = x + F(x)

其中：
├── x：输入
├── F(x)：子层（注意力或FFN）
└── y：输出

关键：直接将输入加到输出上
```

**1. 梯度传播通道**

```
前向传播：
y = x + F(x)

反向传播：
∂y/∂x = 1 + ∂F(x)/∂x

关键洞察：
├── "1"确保梯度可以直接流过
├── 即使F(x)的梯度很小，梯度仍然可以传播
├── 缓解梯度消失问题
└── 深层网络可以训练

对比无残差：
y = F(x)
∂y/∂x = ∂F(x)/∂x

如果∂F(x)/∂x很小（梯度消失）：
├── 浅层几乎收不到梯度
├── 无法有效训练
└── 深层网络不可训练
```

**2. 恒等映射学习**

```
问题：
如果最优解就是恒等映射（输出=输入），网络能学会吗？

无残差：
F(x) 需要学习恒等映射
├── 需要学习 F(x) = x
├── 多层组合后非常困难
└── 参数初始化不是恒等

有残差：
F(x) 只需要学习零映射
├── 只需要学习 F(x) = 0
├── 初始时F(x)接近0（随机初始化）
├── 很容易学会
└── 如果不需要变换，网络可以"跳过"这层

意义：
├── 网络可以选择"使用"或"跳过"每层
├── 有效深度可以自适应调整
└── 提高模型的灵活性
```

**3. 集成效果**

```
理论视角：残差网络可以看作许多浅层网络的集成

展开残差网络：
y = x_L
  = x_{L-1} + F_L(x_{L-1})
  = x_{L-2} + F_{L-1}(x_{L-2}) + F_L(x_{L-1})
  = ...

可以展开为多条路径的组合：
├── 每个残差块可以选择"使用"或"跳过"
├── 相当于2^L条路径
├── 隐式地创建了模型集成
└── 提高泛化能力
```

**为什么深层网络必须使用残差连接：**

```
实验观察（ResNet论文）：
├── 20层网络：训练误差≈测试误差（正常）
├── 56层网络（无残差）：训练误差>测试误差（优化困难！）
├── 56层网络（有残差）：训练误差<测试误差（正常）
└── 深层网络出现"退化"问题

退化问题：
├── 不是过拟合（训练误差也高）
├── 不是能力不足（深层应该能表示浅层）
└── 是优化困难（找不到好解）

残差解决退化：
├── 提供梯度通道
├── 简化恒等映射学习
├── 允许深层网络有效训练
└── Transformer、ResNet等深度模型的基础
```

**Transformer中的残差：**

```
每个子层都有残差连接：

# 自注意力层
attn_output = MultiHeadAttention(x)
x = LayerNorm(x + Dropout(attn_output))
      ↑        ↑
    残差    子层输出

# FFN层
ffn_output = FeedForward(x)
x = LayerNorm(x + Dropout(ffn_output))
      ↑        ↑
    残差    子层输出

作用：
├── 注意力层和FFN层都可以"跳过"
├── 梯度可以跨层传播
├── 堆叠多层（6层、12层、甚至100层）
└── 训练稳定
```

</details>

<details>
<summary>思考题3：BERT和GPT分别使用Transformer的哪部分？为什么BERT适合理解任务，GPT适合生成任务？</summary>

**答案：**

**架构选择：**

```
BERT：仅编码器
├── 双向自注意力
├── 每个位置可以看到所有位置
├── [CLS]位置聚合整个序列信息
└── 输出：每个位置的特征表示

GPT：仅解码器
├── 单向自注意力（因果掩码）
├── 每个位置只能看到之前的位置
├── 自回归生成
└── 输出：下一个词的概率分布
```

**BERT适合理解任务的原因：**

```
1. 双向上下文
   理解任务需要：
   ├── 命名实体识别：识别"苹果"，需要看前后文
   ├── 情感分析：理解"不坏"，需要同时看"不"和"坏"
   ├── 问答：找答案，需要理解问题和上下文
   
   BERT的双向注意力：
   ├── "苹果"可以同时看"我"和"吃"
   ├── 根据完整上下文理解词义
   └── 消歧更准确

2. 完整序列编码
   [CLS]位置的表示：
   ├── 聚合了整个序列的信息
   ├── 可以直接用于分类
   └── 不需要额外处理

3. 位置级别的表示
   每个位置都有输出：
   ├── 可以用于序列标注（NER、POS）
   ├── 可以用于抽取式问答
   └── 灵活适配各种理解任务

例子：
输入：[CLS] 我 喜欢 苹果 手机 [SEP]
BERT输出：
├── [CLS]：整个句子的表示 → 分类任务
├── 我、喜欢、苹果、手机：每个词的上下文表示
└── "苹果"的表示融合了"喜欢"和"手机"，知道是品牌
```

**GPT适合生成任务的原因：**

```
1. 自回归生成
   因果掩码：
   ├── 位置i只能看到位置1到i-1
   ├── 符合文本生成的自然顺序
   └── 训练和推理一致

   生成过程：
   输入：<SOS> 我 爱
   预测：学习
   输入：<SOS> 我 爱 学习
   预测：深度
   ... 逐词生成

2. 单向注意力
   训练时：
   ├── 预测下一个词
   ├── 不能看"未来"的词
   └── 模拟真实生成场景
   
   与理解任务的区别：
   ├── BERT：看到完整序列，猜测中间
   ├── GPT：看到部分序列，预测未来
   └── 生成任务需要后者

3. 概率建模
   语言模型目标：
   P(w_1, w_2, ..., w_n) = Π P(w_i | w_1, ..., w_{i-1})
   
   GPT的训练：
   ├── 最大化上述概率
   ├── 学习语言的统计规律
   └── 自然的生成模型

例子：
输入：人工智能正在
GPT预测：改变、发展、影响...
├── 根据前面的上下文预测下一个词
├── 不需要看后面的词
└── 符合生成任务的需求
```

**BERT vs GPT 任务适配：**

```
BERT（理解任务）：
├── 文本分类
│   └── [CLS] → 分类器 → 类别
├── 命名实体识别
│   └── 每个位置 → 分类器 → 实体标签
├── 问答系统（抽取式）
│   └── 问题+文章 → 标记答案的起止位置
├── 语义相似度
│   └── 两句拼接 → [CLS] → 相似度
└── 情感分析
    └── 句子 → [CLS] → 情感标签

GPT（生成任务）：
├── 文本续写
│   └── 开头 → 生成后续
├── 对话系统
│   └── 用户输入 → 生成回复
├── 代码生成
│   └── 注释 → 生成代码
├── 翻译（生成式）
│   └── 源语言 → 生成目标语言
└── 创意写作
    └── 提示 → 生成文章

混合任务：
├── BART：编码器+解码器
│   └── 适合翻译、摘要
├── T5：编码器+解码器
│   └── 统一的文本到文本框架
└── 现代LLM（如GPT-4）：仅解码器
    └── 通过提示工程适配各种任务
```

**现代趋势：**

```
GPT系列的崛起：
├── GPT-2：15亿参数，生成能力惊艳
├── GPT-3：1750亿参数，少样本学习
├── GPT-4：多模态，推理能力大幅提升
└── 仅解码器架构成为主流

为什么仅解码器足够？
├── 规模效应：足够大的模型可以学习双向理解
├── 提示工程：通过提示适配各种任务
├── 统一接口：生成式解决所有问题
└── 训练简单：只需预测下一个词

BERT系列的应用：
├── 仍然在理解任务上表现优秀
├── 计算效率高（编码一次）
├── 微调成本低
└── 适合资源受限场景
```

</details>

<details>
<summary>思考题4：Transformer训练中为什么使用学习率Warmup？标签平滑有什么作用？</summary>

**答案：**

**学习率Warmup的作用：**

```
问题：深层Transformer训练初期不稳定

原因：
├── 参数随机初始化
├── 深层网络梯度可能很大
├── 大学习率导致参数剧烈变化
└── 容易陷入局部最优或发散

Warmup解决方案：
├── 训练初期使用小学习率
├── 逐步增加学习率到峰值
├── 让模型先"稳定"再"加速"
└── 类似热身运动
```

**Warmup的数学解释：**

```
Transformer的梯度特性：
├── 初始化时，注意力权重接近均匀分布
├── 深层输出可能非常小或非常大
├── 反向传播时梯度不稳定
└── Adam优化器的二阶矩估计不准确

Warmup期间：
├── 学习率从0逐步增加到峰值
├── Adam的动量和方差估计逐渐稳定
├── 参数更新幅度受控
└── 训练稳定后可以用大学习率

学习率调度公式：
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

阶段：
├── Warmup阶段（step < warmup_steps）：
│   └── lr ∝ step（线性增长）
├── 衰减阶段（step ≥ warmup_steps）：
│   └── lr ∝ step^(-0.5)（反比例衰减）
└── warmup_steps通常为4000步
```

**实验验证：**

```
无Warmup：
├── 训练初期loss震荡
├── 可能发散
├── 收敛慢或无法收敛
└── 最终性能可能较差

有Warmup：
├── 训练平稳
├── 快速收敛
├── 最终性能更好
└── 是Transformer训练的标准配置

消融实验：
├── Warmup = 0：难以训练
├── Warmup = 1000：有所改善
├── Warmup = 4000：最佳
└── Warmup太大：初期学习太慢
```

**标签平滑的作用：**

```
问题：One-hot标签过于自信

标准交叉熵：
目标：[0, 0, 1, 0, 0]
鼓励模型输出：[0, 0, 1, 0, 0]（概率1）

问题：
├── 真实数据可能有标注错误
├── 某些样本本身有歧义
├── 强制模型过度自信
└── 可能导致过拟合
```

**标签平滑的实现：**

```
原始标签：
y = [0, 0, 1, 0, 0]

平滑标签（ε = 0.1）：
y_smooth = [ε/(K-1), ε/(K-1), 1-ε, ε/(K-1), ε/(K-1)]
         = [0.025, 0.025, 0.9, 0.025, 0.025]

其中K是类别数

效果：
├── 不再鼓励模型输出概率为1
├── 保留一定的不确定性
└── 模型输出更"谦虚"
```

**标签平滑的好处：**

```
1. 校准置信度
   无标签平滑：
   ├── 模型预测概率接近1
   ├── 但实际准确率可能只有80%
   ├── 置信度与准确率不匹配
   
   有标签平滑：
   ├── 模型预测概率约0.9
   ├── 更接近实际准确率
   └── 置信度更好地反映不确定性

2. 正则化效果
   ├── 防止模型过度自信
   ├── 鼓励模型输出更平滑
   ├── 类似熵正则化
   └── 提高泛化能力

3. 处理标注噪声
   ├── 真实数据可能有错误标注
   ├── 标签平滑提供容忍度
   ├── 模型不会对错误标签过于自信
   └── 对噪声数据更鲁棒

4. 改善嵌入质量
   └── 嵌入空间更均匀分布
   └── 不同类别的嵌入距离更合理
   └── 提高下游任务表现
```

**实践建议：**

```
Warmup设置：
├── 标准Transformer：warmup_steps = 4000
├── 小模型/数据集：warmup_steps = 总步数的10%
├── 大模型：warmup_steps = 总步数的1-3%
└── 观察训练曲线调整

标签平滑设置：
├── 标准值：ε = 0.1
├── 小数据集：ε = 0.1-0.2
├── 大数据集：ε = 0.0-0.05
├── 有噪声数据：ε = 0.1-0.2
└── 需要校准置信度：使用标签平滑

注意事项：
├── 标签平滑会轻微降低训练准确率
├── 但通常提高测试准确率
├── 对于需要精确概率的任务，谨慎使用
└── 知识蒸馏时，教师模型标签平滑有益
```

</details>

---

## 八、今日要点

1. **Transformer架构**：编码器-解码器结构，完全基于注意力机制

2. **核心组件**：
   - 多头自注意力：捕获位置间关系
   - 前馈网络：增加非线性表达
   - 残差连接：梯度通道，支持深层网络
   - 层归一化：稳定训练，适合序列数据

3. **训练技巧**：
   - 学习率Warmup：稳定训练初期
   - 标签平滑：防止过度自信
   - 梯度裁剪：控制梯度爆炸

4. **BERT vs GPT**：
   - BERT：编码器，双向注意力，适合理解任务
   - GPT：解码器，单向注意力，适合生成任务

5. **Pre-Norm vs Post-Norm**：
   - Post-Norm：原论文设计
   - Pre-Norm：现代标准，训练更稳定

---

## 九、明日预告

**第14天：现代深度学习范式**

我们将探讨：
- 大规模预训练与微调
- 提示学习与少样本学习
- 指令微调与RLHF
- 多模态学习
- 深度学习最佳实践

这是课程的最后一天，我们将展望深度学习的未来方向和实践建议。
